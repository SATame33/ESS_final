from __future__ import annotations

import math
import warnings
import sys
import os
import contextlib
import numpy as np
import pandas as pd  # Excel 저장을 위해 추가
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Tuple, Dict, List, Optional, Callable, NamedTuple

from skopt import gp_minimize
from skopt.space import Real
import cvxpy as cp

# 경고 메시지 제어
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. Config dataclasses (SSOT)
# ============================================================

@dataclass(frozen=True)
class TimeConfig:
    num_hours: int = 8760  # 기본값: 1년 (검증용)
    dt_hours: float = 1.0  # 시간 간격
    look_ahead_hours: int = 6  # MPC 예측 지평


@dataclass(frozen=True)
class ESSConfig:
    # [Modified] 4시간 지속형 (300MW / 1200MWh)
    energy_capacity_mwh: float = 1200.0
    power_charge_mw: float = 300.0
    power_discharge_mw: float = 300.0
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    soc_min_tech: float = 0.05
    soc_max: float = 0.95
    soc_initial: float = 0.5
    degradation_cost_per_mwh: float = 1.0


@dataclass(frozen=True)
class RiskWeightConfig:
    w_alpha: float = 0.4
    w_beta: float = 0.4
    w_sigma: float = 0.2

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.w_alpha, self.w_beta, self.w_sigma


@dataclass(frozen=True)
class ReliabilityConfig:
    # [Tuning] 연간 LOLE ~20h 수준을 위해 용량 하향 (3800 -> 3500)
    firm_capacity_mw: float = 3500.0
    largest_unit_mw: float = 1000.0
    value_of_lost_load_per_mwh: float = 10000.0

    # [Dynamic Markov Parameters]
    normal_gen_mean_up_h: float = 200.0
    normal_gen_mean_down_h: float = 8.0
    stress_gen_mean_up_h: float = 24.0
    stress_gen_mean_down_h: float = 24.0

    ren_mean_up_h: float = 500.0
    ren_mean_down_h: float = 12.0


@dataclass(frozen=True)
class OptimizationWeightConfig:
    # 생존 우선 가중치 (LOLE >>> Profit)
    lole_weight: float = 100.0
    eens_weight: float = 0.1
    profit_weight: float = 1e-7


@dataclass(frozen=True)
class RiskBasedPolicyParams:
    tau1: float
    tau2: float
    s1: float
    s2: float


@dataclass(frozen=True)
class MarketRuleConfig:
    price_low_quantile: float = 0.3
    price_high_quantile: float = 0.7
    renewable_low_quantile: float = 0.3
    renewable_high_quantile: float = 0.7


@dataclass(frozen=True)
class SimulationConfig:
    time: TimeConfig = TimeConfig()
    ess: ESSConfig = ESSConfig()
    risk_weights: RiskWeightConfig = RiskWeightConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    optimization_weights: OptimizationWeightConfig = OptimizationWeightConfig()
    market_rules: MarketRuleConfig = MarketRuleConfig()

    # Bayesian Opt Search Space
    tau1_bounds: Tuple[float, float] = (0.2, 0.6)
    tau2_bounds: Tuple[float, float] = (0.5, 0.90)
    s1_bounds: Tuple[float, float] = (0.3, 0.7)
    s2_bounds: Tuple[float, float] = (0.8, 0.99)

    # Optimization Settings
    offline_mc_scenarios: int = 10
    offline_n_calls: int = 50
    random_seed: int = 42


# ============================================================
# 2. Data Structures
# ============================================================

class Strategy(Enum):
    NO_ESS = auto()
    TOU = auto()
    RENEWABLE = auto()
    RISK_BASED = auto()


class SystemScenario(NamedTuple):
    load_mw: np.ndarray
    renewable_mw: np.ndarray
    firm_capacity_mw: np.ndarray
    mssc_mw: np.ndarray
    forecast_error_std: np.ndarray
    price_mwh: np.ndarray
    is_heatwave: np.ndarray


@dataclass
class ThresholdMetrics:
    tau: float
    recall: float
    precision: float
    false_positive_rate: float
    auc: float
    emergency_resolve_rate: float


@dataclass
class SimulationResult:
    strategy: Strategy
    soc: np.ndarray
    ess_power_mw: np.ndarray
    risk_series: np.ndarray
    shortage_without_ess: np.ndarray
    shortage_with_ess: np.ndarray
    lole_hours_without_ess: float
    lole_hours_with_ess: float
    eens_mwh_without_ess: float
    eens_mwh_with_ess: float
    profit: float
    degradation_cost: float
    tau1_metrics: Optional[ThresholdMetrics] = None
    tau2_metrics: Optional[ThresholdMetrics] = None


@dataclass
class BenchmarkResults:
    case_no_ess: SimulationResult
    case_tou: SimulationResult
    case_renewable: SimulationResult
    case_risk_based: SimulationResult


@dataclass
class RiskComponents:
    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    risk: np.ndarray


@dataclass
class ESSStepResult:
    new_soc: float
    actual_power_mw: float
    degradation_cost: float
    energy_charged_mwh: float
    energy_discharged_mwh: float


@dataclass
class MarketThresholds:
    price_low: float
    price_high: float
    renewable_low: float
    renewable_high: float


# ============================================================
# 3. Utility Functions
# ============================================================

@contextlib.contextmanager
def suppress_stdout_stderr():
    """OS-level output suppression for Gurobi"""
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
    except Exception:
        yield


def safe_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0: return arr

    min_v = float(np.min(arr))
    max_v = float(np.quantile(arr, 0.99))  # Quantile Normalization

    if max_v < min_v + eps:
        return np.zeros_like(arr, dtype=float)

    return np.clip((arr - min_v) / (max_v - min_v + eps), 0.0, 1.0)


def compute_auc_pairwise(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    total = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(total / (pos.size * neg.size))


def sample_dynamic_markov_outage(
        hours: int,
        rng: np.random.Generator,
        condition_mask: np.ndarray,
        normal_up: float, normal_down: float,
        stress_up: float, stress_down: float,
        dt: float = 1.0,
        init_down: bool = False
) -> np.ndarray:
    down = init_down
    mask = np.zeros(hours, dtype=bool)

    p_fail_norm = min(1.0, dt / max(normal_up, 1e-9))
    p_repair_norm = min(1.0, dt / max(normal_down, 1e-9))
    p_fail_stress = min(1.0, dt / max(stress_up, 1e-9))
    p_repair_stress = min(1.0, dt / max(stress_down, 1e-9))

    rand_fail = rng.random(size=hours)
    rand_repair = rng.random(size=hours)

    for t in range(hours):
        mask[t] = down
        if condition_mask[t]:
            p_fail, p_repair = p_fail_stress, p_repair_stress
        else:
            p_fail, p_repair = p_fail_norm, p_repair_norm

        if not down:
            if rand_fail[t] < p_fail: down = True
        else:
            if rand_repair[t] < p_repair: down = False

    return mask


def compute_threshold_metrics(
        risk: np.ndarray,
        ground_truth_shortage: np.ndarray,
        final_shortage: np.ndarray,
        tau: float,
) -> ThresholdMetrics:
    risk = np.asarray(risk, dtype=float)
    y_true = np.asarray(ground_truth_shortage, dtype=bool)
    y_final = np.asarray(final_shortage, dtype=bool)
    y_pred = risk >= tau

    tp = float(np.sum(y_pred & y_true))
    fp = float(np.sum(y_pred & ~y_true))

    positive = float(np.sum(y_true))
    negative = float(np.sum(~y_true))
    total_called = float(np.sum(y_pred))

    recall = tp / positive if positive > 0 else 0.0
    precision = tp / total_called if total_called > 0 else 0.0
    fpr = fp / negative if negative > 0 else 0.0

    resolved = (y_true & ~y_final)
    resolved_and_called = (resolved & y_pred)
    resolve_rate = float(np.sum(resolved_and_called)) / positive if positive > 0 else 0.0

    auc = compute_auc_pairwise(y_true, risk)

    return ThresholdMetrics(tau, recall, precision, fpr, auc, resolve_rate)


# ============================================================
# 4. Core Logic: Scenario Generation
# ============================================================

def generate_synthetic_scenario(config: SimulationConfig, rng: np.random.Generator) -> SystemScenario:
    t_cfg = config.time
    r_cfg = config.reliability
    hours = t_cfg.num_hours
    hours_arr = np.arange(hours)

    # 1. Regime Switching: Heatwave
    is_heatwave = sample_dynamic_markov_outage(
        hours, rng,
        condition_mask=np.zeros(hours, dtype=bool),
        normal_up=500.0, normal_down=48.0,
        stress_up=500.0, stress_down=48.0
    )
    if np.sum(is_heatwave) < 12:
        start_idx = hours // 3
        is_heatwave[start_idx: start_idx + 36] = True

    # 2. Load
    daily = np.sin(2.0 * np.pi * (hours_arr % 24) / 24.0 - np.pi / 2.0) * 0.5 + 0.5
    base_load = 3000.0 + 500.0 * daily
    heatwave_adder = np.zeros(hours)
    heatwave_adder[is_heatwave] = rng.uniform(500.0, 800.0, size=np.sum(is_heatwave))
    load_mw = base_load + heatwave_adder + rng.normal(0.0, 30.0, size=hours)
    load_mw = np.clip(load_mw, 1000.0, None)

    # 3. Renewable
    solar = 1500.0 * np.clip(np.sin(2.0 * np.pi * ((hours_arr % 24) - 6.0) / 24.0), 0.0, None)
    wind_base = 500.0 + 150.0 * np.sin(2.0 * np.pi * hours_arr / 48.0)
    wind_factor = np.ones(hours)
    wind_factor[is_heatwave] = rng.uniform(0.1, 0.3, size=np.sum(is_heatwave))
    wind = wind_base * wind_factor + rng.normal(0.0, 30.0, size=hours)

    ren_tech_down = sample_dynamic_markov_outage(
        hours, rng, np.zeros(hours, bool),
        r_cfg.ren_mean_up_h, r_cfg.ren_mean_down_h,
        r_cfg.ren_mean_up_h, r_cfg.ren_mean_down_h
    )
    ren_availability = np.ones(hours)
    ren_availability[ren_tech_down] = 0.2
    renewable_mw = (solar + np.clip(wind, 0.0, None)) * ren_availability

    # 4. Firm Capacity
    firm_capacity_mw = np.full(hours, r_cfg.firm_capacity_mw, dtype=float)
    is_gen_outage = sample_dynamic_markov_outage(
        hours, rng,
        condition_mask=is_heatwave,
        normal_up=r_cfg.normal_gen_mean_up_h, normal_down=r_cfg.normal_gen_mean_down_h,
        stress_up=r_cfg.stress_gen_mean_up_h, stress_down=r_cfg.stress_gen_mean_down_h
    )
    firm_capacity_mw[is_gen_outage] -= r_cfg.largest_unit_mw
    firm_capacity_mw = np.maximum(firm_capacity_mw, 0.0)

    # 5. Derived
    net_load = load_mw - renewable_mw
    mssc_mw = firm_capacity_mw - r_cfg.largest_unit_mw - net_load

    base_std = 50.0
    stress_mult = np.ones(hours)
    stress_mult[is_heatwave] = 2.0
    forecast_error_std = (base_std + rng.normal(0.0, 10.0, size=hours)) * stress_mult
    forecast_error_std = np.clip(forecast_error_std, 1.0, None)

    price_base = 60.0
    reserve = (firm_capacity_mw + renewable_mw) - load_mw
    scarcity_premium = np.zeros(hours)
    low_res_mask = reserve < 800.0
    scarcity_premium[low_res_mask] = 300.0 * np.exp(-reserve[low_res_mask] / 300.0)
    scarcity_premium[is_heatwave] += 150.0
    price_mwh = np.clip(price_base + scarcity_premium + rng.normal(0.0, 5.0, size=hours), 0.0, 3000.0)

    return SystemScenario(
        load_mw, renewable_mw, firm_capacity_mw, mssc_mw, forecast_error_std, price_mwh, is_heatwave
    )


def compute_risk_series(scenario: SystemScenario, config: SimulationConfig) -> RiskComponents:
    w_alpha, w_beta, w_sigma = config.risk_weights.as_tuple()
    net_load = scenario.load_mw - scenario.renewable_mw

    alpha_raw = net_load / np.maximum(scenario.firm_capacity_mw, 1e-6)
    beta_raw = -np.clip(scenario.mssc_mw, -config.reliability.largest_unit_mw, config.reliability.largest_unit_mw)
    sigma_raw = scenario.forecast_error_std ** 2

    alpha_n = safe_normalize(alpha_raw)
    beta_n = safe_normalize(beta_raw)
    sigma_n = safe_normalize(sigma_raw)
    risk = np.clip(w_alpha * alpha_n + w_beta * beta_n + w_sigma * sigma_n, 0.0, 1.0)

    return RiskComponents(alpha_n, beta_n, sigma_n, risk)


def compute_market_thresholds(scenario: SystemScenario, config: SimulationConfig) -> MarketThresholds:
    m = config.market_rules
    return MarketThresholds(
        price_low=float(np.quantile(scenario.price_mwh, m.price_low_quantile)),
        price_high=float(np.quantile(scenario.price_mwh, m.price_high_quantile)),
        renewable_low=float(np.quantile(scenario.renewable_mw, m.renewable_low_quantile)),
        renewable_high=float(np.quantile(scenario.renewable_mw, m.renewable_high_quantile)),
    )


def apply_ess_power(soc: float, p_req: float, config: SimulationConfig) -> ESSStepResult:
    ess = config.ess
    dt = config.time.dt_hours

    if p_req > 0:
        p = min(p_req, ess.power_discharge_mw)
        max_e = max(0.0, (soc - ess.soc_min_tech) * ess.energy_capacity_mwh)
        p = min(p, max_e * ess.efficiency_discharge / dt)
    else:
        p = max(p_req, -ess.power_charge_mw)
        max_e = max(0.0, (ess.soc_max - soc) * ess.energy_capacity_mwh)
        p = max(p, -max_e / (ess.efficiency_charge * dt))

    e_charged, e_discharged = 0.0, 0.0
    soc_change = 0.0

    if p > 1e-9:
        e_out = p * dt
        soc_change = - (e_out / ess.efficiency_discharge) / ess.energy_capacity_mwh
        e_discharged = e_out
    elif p < -1e-9:
        e_in = -p * dt
        soc_change = (e_in * ess.efficiency_charge) / ess.energy_capacity_mwh
        e_charged = e_in
    else:
        p = 0.0

    new_soc = float(np.clip(soc + soc_change, ess.soc_min_tech, ess.soc_max))
    deg_cost = (e_charged + e_discharged) * ess.degradation_cost_per_mwh

    return ESSStepResult(new_soc, p, deg_cost, e_charged, e_discharged)


# ============================================================
# 5. Simulation Runner
# ============================================================

class SimulationRunner:
    def __init__(self, config: SimulationConfig, scenario: SystemScenario):
        self.config = config
        self.scenario = scenario
        self.risk_comp = compute_risk_series(scenario, config)
        self.thresholds = compute_market_thresholds(scenario, config)

        self.solvers_to_try = []
        if 'GUROBI' in cp.installed_solvers():
            self.solvers_to_try.append(cp.GUROBI)
        if 'CLARABEL' in cp.installed_solvers():
            self.solvers_to_try.append(cp.CLARABEL)
        self.solvers_to_try.extend([cp.ECOS, cp.OSQP])

    def simulate(self, strategy: Strategy, params: Optional[RiskBasedPolicyParams] = None) -> SimulationResult:
        cfg = self.config
        T = cfg.time.num_hours

        soc = np.zeros(T + 1)
        soc[0] = cfg.ess.soc_initial
        ess_power = np.zeros(T)
        shortage_no = np.zeros(T, bool)
        shortage_with = np.zeros(T, bool)

        lole_no = 0.0;
        lole_with = 0.0
        eens_no = 0.0;
        eens_with = 0.0
        profit = 0.0;
        deg_cost = 0.0

        load = self.scenario.load_mw
        ren = self.scenario.renewable_mw
        firm = self.scenario.firm_capacity_mw
        price = self.scenario.price_mwh
        risk = self.risk_comp.risk

        for t in range(T):
            supply_base = ren[t] + firm[t]
            shortage_mw_base = max(0.0, load[t] - supply_base)

            if shortage_mw_base > 1e-4:
                shortage_no[t] = True
                lole_no += 1.0
                eens_no += shortage_mw_base

            p_cmd = 0.0
            step = None

            if strategy == Strategy.NO_ESS:
                step = ESSStepResult(soc[t], 0.0, 0.0, 0.0, 0.0)
            else:
                curr_soc = soc[t]
                if strategy == Strategy.TOU:
                    p_cmd = self._strat_tou(curr_soc, price[t])
                elif strategy == Strategy.RENEWABLE:
                    p_cmd = self._strat_renewable(curr_soc, ren[t])
                elif strategy == Strategy.RISK_BASED and params:
                    p_cmd = self._strat_risk_mpc(t, curr_soc, risk, price, params)

                if shortage_mw_base > 0:
                    ess_max_p = self._get_max_discharge(curr_soc)
                    p_cmd = max(p_cmd, min(ess_max_p, shortage_mw_base))

                step = apply_ess_power(curr_soc, p_cmd, cfg)

            soc[t + 1] = step.new_soc
            ess_power[t] = step.actual_power_mw

            supply_final = supply_base + step.actual_power_mw
            shortage_mw_final = max(0.0, load[t] - supply_final)

            if shortage_mw_final > 1e-4:
                shortage_with[t] = True
                lole_with += 1.0
                eens_with += shortage_mw_final

            rev = step.energy_discharged_mwh * price[t]
            cost = step.energy_charged_mwh * price[t]
            profit += (rev - cost - step.degradation_cost)
            deg_cost += step.degradation_cost

        tau1_m, tau2_m = None, None
        if strategy == Strategy.RISK_BASED and params:
            tau1_m = compute_threshold_metrics(risk, shortage_no, shortage_with, params.tau1)
            tau2_m = compute_threshold_metrics(risk, shortage_no, shortage_with, params.tau2)

        return SimulationResult(
            strategy, soc, ess_power, risk, shortage_no, shortage_with,
            lole_no, lole_with, eens_no, eens_with, profit, deg_cost, tau1_m, tau2_m
        )

    def _get_max_discharge(self, soc):
        ess = self.config.ess
        e_avail = max(0.0, (soc - ess.soc_min_tech) * ess.energy_capacity_mwh)
        p_soc = e_avail * ess.efficiency_discharge / self.config.time.dt_hours
        return min(ess.power_discharge_mw, p_soc)

    def _get_max_charge(self, soc):
        ess = self.config.ess
        e_room = max(0.0, (ess.soc_max - soc) * ess.energy_capacity_mwh)
        p_soc = e_room / (ess.efficiency_charge * self.config.time.dt_hours)
        return min(ess.power_charge_mw, p_soc)

    def _strat_tou(self, soc, price):
        if price <= self.thresholds.price_low:
            return -self._get_max_charge(soc)
        if price >= self.thresholds.price_high:
            return self._get_max_discharge(soc)
        return 0.0

    def _strat_renewable(self, soc, ren):
        if ren >= self.thresholds.renewable_high:
            return -self._get_max_charge(soc)
        if ren <= self.thresholds.renewable_low:
            return self._get_max_discharge(soc)
        return 0.0

    def _strat_risk_mpc(self, t, soc, risk_arr, price_arr, p: RiskBasedPolicyParams):
        cfg = self.config
        ess = cfg.ess
        H = cfg.time.look_ahead_hours
        dt = cfg.time.dt_hours
        T = len(risk_arr)

        end = min(t + H, T)
        horizon_len = end - t
        if horizon_len == 0: return 0.0

        r_slice = risk_arr[t:end]
        pr_slice = price_arr[t:end]

        deadline = -1
        target_soc = ess.soc_min_tech

        high_risk_idx = np.where(r_slice >= p.tau2)[0]
        if len(high_risk_idx) > 0:
            deadline = high_risk_idx[0]
            target_soc = max(p.s2, ess.soc_min_tech)
        else:
            med_risk_idx = np.where(r_slice >= p.tau1)[0]
            if len(med_risk_idx) > 0:
                deadline = med_risk_idx[0]
                target_soc = max(p.s1, ess.soc_min_tech)

        if deadline == -1:
            return self._strat_tou(soc, pr_slice[0])

        pc = cp.Variable(horizon_len, nonneg=True)
        pd = cp.Variable(horizon_len, nonneg=True)
        s_var = cp.Variable(horizon_len + 1)

        constrs = [s_var[0] == soc]
        for k in range(horizon_len):
            constrs.append(s_var[k + 1] == s_var[k] + (pc[k] * ess.efficiency_charge - pd[
                k] / ess.efficiency_discharge) * dt / ess.energy_capacity_mwh)
            constrs.append(s_var[k + 1] >= ess.soc_min_tech)
            constrs.append(s_var[k + 1] <= ess.soc_max)
            constrs.append(pc[k] <= ess.power_charge_mw)
            constrs.append(pd[k] <= ess.power_discharge_mw)

        if deadline != -1:
            for k in range(deadline, horizon_len):
                constrs.append(s_var[k + 1] >= target_soc)

        obj_expr = cp.sum(cp.multiply(pr_slice, pd - pc)) * dt - cp.sum(pc + pd) * dt * ess.degradation_cost_per_mwh
        prob = cp.Problem(cp.Maximize(obj_expr), constrs)

        solve_success = False
        with suppress_stdout_stderr():
            for solver in self.solvers_to_try:
                try:
                    solve_args = {'verbose': False}
                    if solver == cp.GUROBI:
                        solve_args.update({"LogToConsole": 0, "OutputFlag": 0})
                    prob.solve(solver=solver, **solve_args)
                    if prob.status in ['optimal', 'optimal_inaccurate']:
                        solve_success = True
                        break
                except:
                    continue

        if not solve_success or pc.value is None:
            return 0.0
        return float(pd.value[0] - pc.value[0])


# ============================================================
# 6. Optimization (Visual Progress Bar)
# ============================================================

def make_objective(config: SimulationConfig) -> Callable[[List[float]], float]:
    total_calls = config.offline_n_calls
    counter = {"n": 0}
    bar_width = 30

    def obj_func(x):
        counter["n"] += 1
        n = counter["n"]
        pct = min(100.0, 100.0 * n / max(1, total_calls))
        filled = int(bar_width * n / max(1, total_calls))
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\r[Offline Opt] |{bar}| {n}/{total_calls} ({pct:5.1f}%)")
        sys.stdout.flush()

        tau1, tau2, s1, s2 = x
        if not (config.tau1_bounds[0] <= tau1 <= config.tau1_bounds[1]): return 1e6
        if not (config.tau2_bounds[0] <= tau2 <= config.tau2_bounds[1]): return 1e6
        if not (config.s1_bounds[0] <= s1 <= config.s1_bounds[1]): return 1e6
        if not (config.s2_bounds[0] <= s2 <= config.s2_bounds[1]): return 1e6
        if tau1 >= tau2: return 1e5 + (tau1 - tau2) ** 2
        if s1 >= s2: return 1e5 + (s1 - s2) ** 2

        params = RiskBasedPolicyParams(tau1, tau2, s1, s2)
        rng = np.random.default_rng(config.random_seed)

        total_lole = 0.0
        total_profit = 0.0

        for _ in range(config.offline_mc_scenarios):
            scen = generate_synthetic_scenario(config, rng)
            res = SimulationRunner(config, scen).simulate(Strategy.RISK_BASED, params)
            total_lole += res.lole_hours_with_ess
            total_profit += res.profit

        avg_lole = total_lole / config.offline_mc_scenarios
        avg_profit = total_profit / config.offline_mc_scenarios

        w = config.optimization_weights
        cost = w.lole_weight * avg_lole - w.profit_weight * avg_profit
        return cost

    return obj_func


def run_optimization(config: SimulationConfig):
    space = [
        Real(*config.tau1_bounds), Real(*config.tau2_bounds),
        Real(*config.s1_bounds), Real(*config.s2_bounds)
    ]
    print(">>> Starting Bayesian Optimization (Silent Mode)...")
    res = gp_minimize(
        make_objective(config),
        space,
        n_calls=config.offline_n_calls,
        random_state=config.random_seed
    )
    print()
    best = RiskBasedPolicyParams(*res.x)
    print(f">>> Best Params: Tau1={best.tau1:.2f}, Tau2={best.tau2:.2f}, S1={best.s1:.2f}, S2={best.s2:.2f}")
    return best


# ============================================================
# 7. Visualization & Excel Export
# ============================================================

def export_results_to_excel(bench: BenchmarkResults, scen: SystemScenario, params: RiskBasedPolicyParams,
                            filename="simulation_results.xlsx"):
    print(f">>> Exporting results to {filename}...")

    # 1. Summary DataFrame
    summary_data = []
    strategies = {
        "No ESS": bench.case_no_ess,
        "TOU": bench.case_tou,
        "Renewable": bench.case_renewable,
        "Risk-Based": bench.case_risk_based
    }

    for name, res in strategies.items():
        summary_data.append({
            "Strategy": name,
            "LOLE (h)": res.lole_hours_with_ess if name != "No ESS" else res.lole_hours_without_ess,
            "EENS (MWh)": res.eens_mwh_with_ess if name != "No ESS" else res.eens_mwh_without_ess,
            "Profit ($)": res.profit,
            "Degradation Cost ($)": res.degradation_cost,
            "Energy Charged (MWh)": res.energy_charged_mwh,
            "Energy Discharged (MWh)": res.energy_discharged_mwh
        })

    df_summary = pd.DataFrame(summary_data)

    # 2. Parameters DataFrame
    param_data = [{
        "Parameter": k, "Value": v
    } for k, v in asdict(params).items()]
    df_params = pd.DataFrame(param_data)

    # 3. Time Series DataFrame
    T = len(scen.load_mw)
    df_ts = pd.DataFrame({
        "Hour": np.arange(T),
        "Load (MW)": scen.load_mw,
        "Renewable (MW)": scen.renewable_mw,
        "Firm Capacity (MW)": scen.firm_capacity_mw,
        "Supply No ESS (MW)": scen.firm_capacity_mw + scen.renewable_mw,
        "Price ($/MWh)": scen.price_mwh,
        "Risk Index": bench.case_risk_based.risk_series,
        "Is Heatwave": scen.is_heatwave,
    })

    # Add strategy-specific columns
    for name, res in strategies.items():
        prefix = name.replace(" ", "_")
        df_ts[f"{prefix}_SOC"] = res.soc[:-1]
        df_ts[f"{prefix}_Power"] = res.ess_power_mw

        # 부족량 (Shortage)
        shortage = np.zeros(T)
        supply = scen.firm_capacity_mw + scen.renewable_mw + np.maximum(0, res.ess_power_mw)
        if name == "No ESS":
            supply = scen.firm_capacity_mw + scen.renewable_mw

        shortage = np.maximum(0, scen.load_mw - supply)
        df_ts[f"{prefix}_Shortage"] = shortage

    # Save to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_params.to_excel(writer, sheet_name='Parameters', index=False)
        df_ts.to_excel(writer, sheet_name='Time Series', index=False)

    print(">>> Export Complete.")


def plot_results(bench: BenchmarkResults, params: RiskBasedPolicyParams, scen: SystemScenario):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

    h = np.arange(len(scen.load_mw))
    risk = bench.case_risk_based.risk_series

    strategies = ["No ESS", "TOU", "Ren.", "Risk-Based"]
    colors = ['gray', 'tab:blue', 'tab:green', 'tab:red']

    # (1) LOLE
    plt.figure(figsize=(6, 5))
    lole = [
        bench.case_no_ess.lole_hours_without_ess,
        bench.case_tou.lole_hours_with_ess,
        bench.case_renewable.lole_hours_with_ess,
        bench.case_risk_based.lole_hours_with_ess
    ]
    bars = plt.bar(strategies, lole, color=colors)
    plt.title("Reliability: LOLE (Hours)")
    plt.ylabel("Hours")
    plt.grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.1f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()  # Display only

    # (2) EENS
    plt.figure(figsize=(6, 5))
    eens = [
        bench.case_no_ess.eens_mwh_without_ess,
        bench.case_tou.eens_mwh_with_ess,
        bench.case_renewable.eens_mwh_with_ess,
        bench.case_risk_based.eens_mwh_with_ess
    ]
    bars = plt.bar(strategies, eens, color=colors)
    plt.title("Severity: EENS (MWh)")
    plt.ylabel("MWh")
    plt.grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.0f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # (3) Profit
    plt.figure(figsize=(6, 5))
    prof = [
        bench.case_no_ess.profit,
        bench.case_tou.profit,
        bench.case_renewable.profit,
        bench.case_risk_based.profit
    ]
    bars = plt.bar(strategies, prof, color=colors)
    plt.title("Economics: Profit ($)")
    plt.ylabel("USD")
    plt.grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5000, f"${yval / 1000:.0f}k", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # Time Series Plotting Helper
    def highlight_heatwave(ax):
        hw_indices = np.where(scen.is_heatwave)[0]
        if len(hw_indices) > 0:
            from itertools import groupby
            from operator import itemgetter
            for k, g in groupby(enumerate(hw_indices), lambda ix: ix[0] - ix[1]):
                grp = list(map(itemgetter(1), g))
                ax.axvspan(grp[0], grp[-1], color='red', alpha=0.15, label='Heatwave' if k == 0 else "")

    # (4) Supply vs Load
    plt.figure(figsize=(10, 4))
    supply = scen.firm_capacity_mw + scen.renewable_mw
    plt.plot(h, scen.load_mw, 'k-', lw=1.5, label='Load')
    plt.plot(h, supply, color='tab:blue', lw=1.5, label='Supply (Gen+Ren)')
    highlight_heatwave(plt.gca())
    short_idx = np.where(bench.case_no_ess.shortage_without_ess)[0]
    if len(short_idx) > 0:
        plt.scatter(short_idx, scen.load_mw[short_idx], color='red', s=30, zorder=5, label='Shortage')
    plt.ylabel("MW")
    plt.title("System Status: Load vs Supply")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (5) Risk Index
    plt.figure(figsize=(10, 4))
    plt.plot(h, risk, 'tab:red', lw=1.5, label='Risk Index')
    plt.axhline(params.tau1, color='orange', ls='--', lw=1.5, label=f'Tau1={params.tau1:.2f}')
    plt.axhline(params.tau2, color='purple', ls='--', lw=1.5, label=f'Tau2={params.tau2:.2f}')
    plt.ylabel("Risk (0-1)")
    plt.title("Risk Dynamics & Thresholds")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (6) SOC
    plt.figure(figsize=(10, 4))
    plt.plot(h, bench.case_tou.soc[:-1], color='gray', ls=':', lw=1.5, label='SOC (TOU)')
    plt.plot(h, bench.case_risk_based.soc[:-1], color='tab:green', lw=2.0, label='SOC (Risk-Based)')
    plt.axhline(params.s1, color='orange', ls=':', alpha=0.5, label='S1 Target')
    plt.axhline(params.s2, color='purple', ls=':', alpha=0.5, label='S2 Target')
    plt.ylabel("SOC")
    plt.xlabel("Time (Hour)")
    plt.title("ESS Operation: SOC Response")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main Execution
# ============================================================

if __name__ == "__main__":
    # 1. Config Setup (Split Optimization vs Validation)
    config_opt = SimulationConfig(
        time=TimeConfig(num_hours=168),  # Speed up optimization (1 week)
        ess=ESSConfig(energy_capacity_mwh=1200.0)
    )

    config_val = SimulationConfig(
        time=TimeConfig(num_hours=8760),  # Full validation (1 year)
        ess=ESSConfig(energy_capacity_mwh=1200.0)
    )

    print(f"=== Simulation Start: Resilience Assessment (HILP Scenario) ===")
    print(f"Config: 4-Hour ESS (1200MWh), Optimization=168h, Validation=8760h")
    print(f"Target LOLE: ~20h/year (Weak Grid Baseline)")

    # 2. Optimization
    best_params = run_optimization(config_opt)

    # 3. Validation
    print(">>> Running Full Validation (1 year)...")
    rng_val = np.random.default_rng(config_val.random_seed + 100)
    val_scen = generate_synthetic_scenario(config_val, rng_val)
    runner = SimulationRunner(config_val, val_scen)

    print("\n>>> Running Benchmarks...")
    res_no = runner.simulate(Strategy.NO_ESS)
    res_tou = runner.simulate(Strategy.TOU)
    res_ren = runner.simulate(Strategy.RENEWABLE)
    res_risk = runner.simulate(Strategy.RISK_BASED, best_params)

    bench = BenchmarkResults(res_no, res_tou, res_ren, res_risk)

    # 4. Final Summary Table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<12} | {'LOLE (h)':<10} | {'EENS (MWh)':<12} | {'Profit ($)':<12}")
    print("-" * 80)
    print(
        f"{'No ESS':<12} | {res_no.lole_hours_without_ess:<10.2f} | {res_no.eens_mwh_without_ess:<12.2f} | {res_no.profit:<12.0f}")
    print(
        f"{'TOU':<12} | {res_tou.lole_hours_with_ess:<10.2f} | {res_tou.eens_mwh_with_ess:<12.2f} | {res_tou.profit:<12.0f}")
    print(
        f"{'Renewable':<12} | {res_ren.lole_hours_with_ess:<10.2f} | {res_ren.eens_mwh_with_ess:<12.2f} | {res_ren.profit:<12.0f}")
    print(
        f"{'Risk-Based':<12} | {res_risk.lole_hours_with_ess:<10.2f} | {res_risk.eens_mwh_with_ess:<12.2f} | {res_risk.profit:<12.0f}")
    print("=" * 80)

    # 5. Report
    print(f"\n[Optimized Parameters]")
    print(f"  Tau1 : {best_params.tau1:.4f}")
    print(f"  Tau2 : {best_params.tau2:.4f}")
    print(f"  S1   : {best_params.s1:.4f}")
    print(f"  S2   : {best_params.s2:.4f}")

    print(f"\n[Risk Model Diagnostics]")
    if res_risk.tau1_metrics:
        m = res_risk.tau1_metrics
        print(
            f"  > Medium Risk (Tau1={m.tau:.2f}): AUC={m.auc:.3f}, Recall={m.recall:.2f}, Resolve Rate={m.emergency_resolve_rate:.1%}")
    if res_risk.tau2_metrics:
        m = res_risk.tau2_metrics
        print(
            f"  > High Risk   (Tau2={m.tau:.2f}): AUC={m.auc:.3f}, Recall={m.recall:.2f}, Resolve Rate={m.emergency_resolve_rate:.1%}")
    print("=" * 80)

    # 6. Save Excel & Show Plots
    export_results_to_excel(bench, val_scen, best_params)
    plot_results(bench, best_params, val_scen)