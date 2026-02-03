# -*- coding: utf-8 -*-
"""
Risk-based ESS Scheduling (8760h) - Final of Finals (SSOT/DRY/Optimized)
- SSOT: Unified Capacity Time-Series (Gen/Line/Firm) shared by Risk + Simulation
- Smart Risk: Real Deficit-based (Base deficit + Expected N-1 deficit) + Forward Accumulated Energy Risk
- MPC: Flexible SOC floor with Slack, Cost-synced with Simulation, Global reuse, MPC skip on low-risk horizon
- Phase 1 Sizing: Event-based P/E joint coverage sizing using vectorized deficit calc (no Runner/MPC)
- Evaluation: Time-based (AUC/Precision/Recall/Resolved) + Event-based (Event Recall/Lead Time)
- Export: Excel + Plot for last evaluated RiskBased scenario
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
except Exception as e:
    raise RuntimeError(f"cvxpy import failed: {e}")

try:
    from skopt import gp_minimize
    from skopt.space import Real
except Exception as e:
    raise RuntimeError(f"scikit-optimize import failed: {e}")


# =========================
# Utilities
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    if abs(d) <= 1e-12:
        return default
    return float(n / d)


@contextlib.contextmanager
def suppress_stdout_stderr() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = fnull, fnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def robust_minmax_scale(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = float(np.min(x)) if x.size else 0.0
    xmax = float(np.max(x)) if x.size else 0.0
    if abs(xmax - xmin) <= eps:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def percentile_clip(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return x.copy()
    return np.clip(x, lo, hi)


def print_progress(prefix: str, n: int, total: int, bar_width: int = 30) -> None:
    pct = min(100.0, 100.0 * n / max(1, total))
    filled = int(bar_width * n / max(1, total))
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r{prefix} |{bar}| {n}/{total} ({pct:5.1f}%)", end="", flush=True)


def forward_window_sum(x: np.ndarray, window: int) -> np.ndarray:
    """
    Forward-looking sum:
      out[t] = sum_{k=0..window-1} x[t+k]
    with padding by zeros beyond T.
    """
    x = np.asarray(x, dtype=float)
    T = x.size
    w = int(max(1, window))
    cs = np.cumsum(np.concatenate([x, np.zeros(w, dtype=float)]))
    out = cs[w:w + T] - cs[0:T]
    return out


def roc_auc_trapezoid(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC using trapezoid integration over FPR.
    Returns nan if only one class exists.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1).astype(float)
    fps = np.cumsum(y_sorted == 0).astype(float)

    tpr = tps / max(1.0, tps[-1])
    fpr = fps / max(1.0, fps[-1])

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def lam_from_for_mttr(forced_outage_rate: float, mttr_hours: float) -> float:
    """
    Match the discrete two-state Markov used in scenario generation:
      mu = 1/MTTR
      lam = (FOR*mu)/(1-FOR)
    Here lam is used as per-hour Up->Down transition probability (clamped).
    """
    for_val = clamp(forced_outage_rate, 0.0, 0.98)
    mu = 1.0 / max(mttr_hours, 1e-9)
    lam = (for_val * mu) / max(1e-9, (1.0 - for_val))
    return clamp(lam, 0.0, 1.0)


def prob_fail_within_h(lam_step: float, h: int) -> float:
    """
    Discrete-time probability of at least one failure within next h steps:
      p = 1 - (1-lam)^h
    """
    lam_step = clamp(lam_step, 0.0, 1.0)
    h = int(max(1, h))
    return float(1.0 - (1.0 - lam_step) ** h)


# =========================
# Data Models
# =========================

class Strategy(enum.Enum):
    NO_ESS = "NoESS"
    TOU = "TOU"
    RENEWABLE_BASED = "RenewableBased"
    RISK_BASED = "RiskBased"


@dataclass(frozen=True)
class OptimizationWeights:
    lole_weight: float
    profit_weight: float


@dataclass(frozen=True)
class RiskWeights:
    w1_alpha: float
    w2_beta: float
    w3_sigma: float
    w4_accum: float


@dataclass(frozen=True)
class GeneratorUnit:
    name: str
    capacity_mw: float
    forced_outage_rate: float
    mttr_hours: float


@dataclass(frozen=True)
class TransmissionLine:
    name: str
    capacity_mw: float
    forced_outage_rate: float
    mttr_hours: float


@dataclass(frozen=True)
class ESSConfig:
    energy_capacity_mwh: float
    power_capacity_mw: float
    soc_init_percent: float
    eta_charge: float
    eta_discharge: float
    degradation_cost_per_mwh_throughput: float
    ramp_limit_mw_per_h: float


@dataclass(frozen=True)
class SimulationConfig:
    horizon_hours: int
    dt_hours: float
    mpc_horizon_hours: int

    generators: List[GeneratorUnit]
    lines: List[TransmissionLine]

    base_load_mw: float
    daily_load_swing_mw: float
    load_noise_std_mw: float

    renewable_capacity_mw: float
    solar_share: float
    wind_share: float
    renewable_noise_std_mw: float
    renewable_forecast_base_std_mw: float
    renewable_forecast_std_factor: float

    price_offpeak: float
    price_mid: float
    price_peak: float
    tou_peak_hours: List[int]
    tou_mid_hours: List[int]
    tou_offpeak_hours: List[int]

    risk_weights: RiskWeights
    risk_clip_percentiles: Tuple[float, float]
    risk_forecast_mode: str
    risk_forecast_noise_std: float
    ar1_window: int

    ess: ESSConfig

    offline_n_calls: int
    offline_mc_scenarios: int
    optimization_horizon_hours: int
    random_seed: int
    optimization_weights: OptimizationWeights

    tau1_bounds: Tuple[float, float]
    tau2_bounds: Tuple[float, float]
    s1_bounds: Tuple[float, float]
    s2_bounds: Tuple[float, float]

    mpc_use_binary: bool
    mpc_big_m: float
    mpc_cycle_penalty_per_mwh: float
    mpc_slack_penalty_per_mwh: float

    tou_low_price_quantile: float
    tou_high_price_quantile: float
    renewable_high_quantile: float
    renewable_low_quantile: float

    mc_eval_scenarios: int
    mc_eval_seed: int
    mc_show_progress: bool


@dataclass(frozen=True)
class ScenarioData:
    load_mw: np.ndarray
    renewable_actual_mw: np.ndarray
    renewable_forecast_mw: np.ndarray
    renewable_forecast_std_mw: np.ndarray
    price_per_mwh: np.ndarray
    gen_online: np.ndarray
    line_online: np.ndarray


@dataclass(frozen=True)
class RiskBasedPolicyParams:
    tau1: float
    tau2: float
    s1: float
    s2: float


@dataclass(frozen=True)
class CapacitySeries:
    gen_cap_mw: np.ndarray
    line_cap_mw: np.ndarray
    firm_cap_mw: np.ndarray  # min(gen, line)


@dataclass
class SimulationResult:
    strategy: Strategy
    params: Optional[RiskBasedPolicyParams]

    # SSOT time-series
    load_mw: np.ndarray
    renewable_actual_mw: np.ndarray
    firm_cap_mw: np.ndarray
    firm_supply_mw: np.ndarray
    price_per_mwh: np.ndarray

    # Risk / deficits
    risk_series: np.ndarray
    deficit_pre_ess_mw: np.ndarray
    unserved_mw: np.ndarray

    # ESS operation
    soc_percent: np.ndarray
    p_charge_mw: np.ndarray
    p_discharge_mw: np.ndarray
    p_emergency_mw: np.ndarray

    # Aggregates
    lole_hours: float
    eens_mwh: float
    profit: float


# =========================
# Scenario Generation
# =========================

def _simulate_two_state_markov(T: int, lam: float, mu: float, rng: np.random.Generator, initial_up: bool = True) -> np.ndarray:
    """
    Discrete two-state Markov chain.
    lam, mu are per-step transition probabilities in dt=1h.
    """
    online = np.zeros(T, dtype=bool)
    state_up = bool(initial_up)

    p_fail = clamp(lam, 0.0, 1.0)
    p_repair = clamp(mu, 0.0, 1.0)

    for t in range(T):
        online[t] = state_up
        if state_up:
            if rng.random() < p_fail:
                state_up = False
        else:
            if rng.random() < p_repair:
                state_up = True
    return online


def generate_synthetic_scenario(cfg: SimulationConfig, rng: np.random.Generator, length: int) -> ScenarioData:
    T = int(length)
    dt = float(cfg.dt_hours)

    hours = np.arange(T, dtype=float)

    daily_phase = 2.0 * math.pi * (hours % 24.0) / 24.0
    base_load = cfg.base_load_mw + cfg.daily_load_swing_mw * (0.5 * (1.0 + np.sin(daily_phase - math.pi / 2.0)))
    load = np.clip(base_load + rng.normal(0.0, cfg.load_noise_std_mw, size=T), 0.0, None)

    # Solar shape (normalized)
    h24 = (hours % 24.0)
    solar_shape = np.zeros(T, dtype=float)
    mask = (h24 >= 6.0) & (h24 <= 18.0)
    solar_shape[mask] = np.exp(-2.0 * ((h24[mask] - 12.0) / 6.0) ** 2)
    solar = robust_minmax_scale(solar_shape)

    # Wind AR(1)-like (simple)
    wind = np.zeros(T, dtype=float)
    val = float(rng.random())
    for t in range(T):
        val = 0.85 * val + 0.15 * float(rng.random())
        wind[t] = val
    wind = robust_minmax_scale(wind)

    ren_raw = cfg.solar_share * solar + cfg.wind_share * wind
    ren_actual = np.clip(
        cfg.renewable_capacity_mw * robust_minmax_scale(ren_raw) + rng.normal(0.0, cfg.renewable_noise_std_mw, size=T),
        0.0,
        cfg.renewable_capacity_mw
    )

    # Forecast (smoothed - noise)
    kernel = np.ones(3, dtype=float) / 3.0
    ren_fc = np.convolve(ren_actual, kernel, mode="same")
    ren_fc_std = cfg.renewable_forecast_base_std_mw + cfg.renewable_forecast_std_factor * ren_fc
    ren_fc = np.clip(ren_fc - rng.normal(0.0, ren_fc_std, size=T), 0.0, cfg.renewable_capacity_mw)

    # TOU Price
    price = np.zeros(T, dtype=float)
    for t in range(T):
        hh = int(t % 24)
        if hh in cfg.tou_peak_hours:
            price[t] = cfg.price_peak
        elif hh in cfg.tou_mid_hours:
            price[t] = cfg.price_mid
        else:
            price[t] = cfg.price_offpeak

    # Markov outages
    gen_on = np.zeros((T, len(cfg.generators)), dtype=bool)
    for i, g in enumerate(cfg.generators):
        mu = clamp(1.0 / max(g.mttr_hours, 0.1), 0.0, 1.0)
        lam = lam_from_for_mttr(g.forced_outage_rate, g.mttr_hours)
        gen_on[:, i] = _simulate_two_state_markov(T, lam, mu, rng, initial_up=True)

    line_on = np.zeros((T, len(cfg.lines)), dtype=bool)
    for i, l in enumerate(cfg.lines):
        mu = clamp(1.0 / max(l.mttr_hours, 0.1), 0.0, 1.0)
        lam = lam_from_for_mttr(l.forced_outage_rate, l.mttr_hours)
        line_on[:, i] = _simulate_two_state_markov(T, lam, mu, rng, initial_up=True)

    return ScenarioData(
        load_mw=load,
        renewable_actual_mw=ren_actual,
        renewable_forecast_mw=ren_fc,
        renewable_forecast_std_mw=ren_fc_std,
        price_per_mwh=price,
        gen_online=gen_on,
        line_online=line_on
    )


# =========================
# SSOT: Capacity Series
# =========================

def compute_capacity_series(cfg: SimulationConfig, scen: ScenarioData) -> CapacitySeries:
    T = int(len(scen.load_mw))

    gen_caps = np.asarray([g.capacity_mw for g in cfg.generators], dtype=float)
    if gen_caps.size == 0:
        gen_ts = np.zeros(T, dtype=float)
    else:
        gen_ts = (scen.gen_online @ gen_caps).astype(float)

    if len(cfg.lines) == 0:
        line_ts = np.full(T, float("inf"), dtype=float)
    else:
        line_caps = np.asarray([l.capacity_mw for l in cfg.lines], dtype=float)
        line_ts = (scen.line_online @ line_caps).astype(float)

    firm_ts = np.minimum(gen_ts, line_ts)
    return CapacitySeries(gen_cap_mw=gen_ts, line_cap_mw=line_ts, firm_cap_mw=firm_ts)


# =========================
# Event Analysis (Sizing & Event Metrics)
# =========================

class DeficitEvent(tuple):
    def __new__(cls, start: int, end: int, duration_h: float, peak_mw: float, energy_out_mwh: float, energy_from_ess_mwh: float):
        return super(DeficitEvent, cls).__new__(cls, (start, end, duration_h, peak_mw, energy_out_mwh, energy_from_ess_mwh))

    @property
    def start(self) -> int:
        return int(self[0])

    @property
    def end(self) -> int:
        return int(self[1])

    @property
    def duration_h(self) -> float:
        return float(self[2])

    @property
    def peak_mw(self) -> float:
        return float(self[3])

    @property
    def energy_out_mwh(self) -> float:
        return float(self[4])

    @property
    def energy_from_ess_mwh(self) -> float:
        return float(self[5])


def extract_deficit_events(deficit_mw: np.ndarray, dt: float, eta_dis: float, eps: float = 1e-9) -> List[DeficitEvent]:
    d = np.asarray(deficit_mw, dtype=float)
    T = int(d.size)
    events: List[DeficitEvent] = []

    in_evt = False
    s = 0

    for t in range(T):
        if (d[t] > eps) and (not in_evt):
            in_evt = True
            s = t

        if in_evt and ((d[t] <= eps) or (t == T - 1)):
            e = (t - 1) if (d[t] <= eps) else t
            seg = d[s:e + 1]
            peak = float(np.max(seg)) if seg.size else 0.0
            energy_out = float(np.sum(seg) * dt)
            energy_from = float(energy_out / max(eta_dis, 1e-9))
            events.append(DeficitEvent(s, e, float((e - s + 1) * dt), peak, energy_out, energy_from))
            in_evt = False

    return events


def analyze_deficit_events(all_deficits: List[np.ndarray], dt: float, eta_dis: float, q_target: float = 0.98) -> Dict[str, Any]:
    all_events: List[DeficitEvent] = []
    for deficit in all_deficits:
        all_events.extend(extract_deficit_events(deficit, dt, eta_dis))

    if not all_events:
        return {"n_events": 0, "p_req_q": 0.0, "e_req_q": 0.0, "joint_coverage": float("nan")}

    p_reqs = np.asarray([ev.peak_mw for ev in all_events], dtype=float)
    e_reqs = np.asarray([ev.energy_from_ess_mwh for ev in all_events], dtype=float)

    p_q = 0.0
    e_q = 0.0
    joint = 0.0

    for q_candidate in np.linspace(q_target, 0.999, 25):
        p_val = float(np.quantile(p_reqs, q_candidate))
        e_val = float(np.quantile(e_reqs, q_candidate))
        joint_val = float(np.mean((p_reqs <= p_val) & (e_reqs <= e_val)))
        if joint_val >= q_target:
            p_q, e_q, joint = p_val, e_val, joint_val
            break

    if p_q <= 0.0:
        p_q = float(np.quantile(p_reqs, 0.99))
        e_q = float(np.quantile(e_reqs, 0.99))
        joint = float(np.mean((p_reqs <= p_q) & (e_reqs <= e_q)))

    return {"n_events": int(len(all_events)), "p_req_q": p_q, "e_req_q": e_q, "joint_coverage": joint}


@dataclass(frozen=True)
class EventMetrics:
    threshold: float
    event_recall: float
    mean_lead_time_h: float
    n_events: int
    n_alarms: int


def evaluate_events(risk: np.ndarray, deficit: np.ndarray, threshold: float, lookback_h: int, dt: float, eps: float = 1e-9) -> EventMetrics:
    risk = np.asarray(risk, dtype=float)
    deficit = np.asarray(deficit, dtype=float)

    events = extract_deficit_events(deficit, dt, eta_dis=1.0, eps=eps)
    alarms = np.where(risk >= threshold)[0]

    detected = 0
    lead_times: List[float] = []

    for ev in events:
        start = ev.start
        w0 = max(0, start - int(lookback_h))
        cand = alarms[(alarms >= w0) & (alarms <= start)]
        if cand.size > 0:
            detected += 1
            lead_times.append(float((start - int(np.min(cand))) * dt))

    return EventMetrics(
        threshold=float(threshold),
        event_recall=safe_div(detected, len(events), 0.0),
        mean_lead_time_h=float(np.mean(lead_times)) if lead_times else 0.0,
        n_events=int(len(events)),
        n_alarms=int(alarms.size)
    )


# =========================
# Time-based Metrics
# =========================

def compute_time_metrics(risk: np.ndarray, deficit_pre: np.ndarray, unserved: np.ndarray, tau: float, eps: float = 1e-9) -> Dict[str, float]:
    risk = np.asarray(risk, dtype=float)
    deficit_pre = np.asarray(deficit_pre, dtype=float)
    unserved = np.asarray(unserved, dtype=float)

    y_true = (deficit_pre > eps).astype(int)
    y_pred = (risk >= float(tau)).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall = safe_div(tp, tp + fn, 0.0)
    precision = safe_div(tp, tp + fp, 0.0)

    called_true = (y_true == 1) & (y_pred == 1)
    denom = int(np.sum(called_true))
    resolved = int(np.sum(called_true & (unserved <= eps)))
    resolved_rate = safe_div(resolved, denom, 0.0)

    auc = roc_auc_trapezoid(y_true, risk)

    return {
        "AUC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "Resolved": float(resolved_rate),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "TN": float(tn),
    }


# =========================
# Risk Forecaster (oracle / ar1)
# =========================

class RiskForecaster:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def forecast(self, risk_series: np.ndarray, t: int, horizon: int, rng: Optional[np.random.Generator]) -> np.ndarray:
        risk_series = np.asarray(risk_series, dtype=float)
        T = int(risk_series.size)
        H = int(max(1, horizon))
        mode = str(self.cfg.risk_forecast_mode).lower()
        noise_std = float(self.cfg.risk_forecast_noise_std)

        if mode == "oracle":
            out = np.array([float(risk_series[min(T - 1, t + k)]) for k in range(H)], dtype=float)
        elif mode == "ar1":
            win = int(max(2, self.cfg.ar1_window))
            if t < 2:
                out = np.full(H, float(risk_series[min(T - 1, t)]), dtype=float)
            else:
                lo = max(1, t - win + 1)
                xs = risk_series[lo - 1:t]
                ys = risk_series[lo:t + 1]
                mu_x = float(np.mean(xs))
                mu_y = float(np.mean(ys))
                num = float(np.sum((xs - mu_x) * (ys - mu_y)))
                den = float(np.sum((xs - mu_x) ** 2)) + 1e-12
                phi = clamp(num / den, -0.99, 0.99)
                mu = float(np.mean(ys))

                last = float(risk_series[t])
                out = np.zeros(H, dtype=float)
                for k in range(H):
                    last = mu + phi * (last - mu)
                    out[k] = last
        else:
            raise ValueError(f"Unknown risk_forecast_mode: {mode}")

        if noise_std > 0.0 and rng is not None:
            out = out + rng.normal(0.0, noise_std, size=H)

        return np.clip(out, 0.0, 1.0)


# =========================
# Smart Risk (SSOT-compliant, no proxy)
# =========================

def compute_risk_series(cfg: SimulationConfig, scen: ScenarioData, cap: CapacitySeries) -> np.ndarray:
    """
    Risk components:
      alpha: supply pressure = net_load_fc / firm_cap
      beta: expected severity of worst N-1 contingency within horizon H
      sigma: forecast uncertainty (variance-like)
      accum: forward accumulated expected deficit-energy fraction within window W

    No proxy: N-1 effect uses firm_after_best computed consistently per time step.
    """
    T = int(len(scen.load_mw))
    dt = float(cfg.dt_hours)

    firm_cap = np.asarray(cap.firm_cap_mw, dtype=float)
    gen_cap = np.asarray(cap.gen_cap_mw, dtype=float)
    line_cap = np.asarray(cap.line_cap_mw, dtype=float)

    load = np.asarray(scen.load_mw, dtype=float)
    ren_fc = np.asarray(scen.renewable_forecast_mw, dtype=float)
    net_load_fc = np.maximum(0.0, load - ren_fc)

    # alpha raw (pressure)
    alpha_raw = net_load_fc / np.maximum(1e-6, firm_cap)

    # sigma raw (variance-like)
    sigma_raw = np.asarray(scen.renewable_forecast_std_mw, dtype=float) ** 2

    # beta raw + store firm_after_best and p_fail_best for accum
    H = int(max(1, cfg.mpc_horizon_hours))
    gen_total = float(np.sum([g.capacity_mw for g in cfg.generators])) if cfg.generators else 1.0

    beta_raw = np.zeros(T, dtype=float)
    firm_after_best = firm_cap.copy()
    p_fail_best = np.zeros(T, dtype=float)

    for t in range(T):
        cap_now = float(firm_cap[t])
        gen_now = float(gen_cap[t])
        line_now = float(line_cap[t])
        net = float(net_load_fc[t])

        best_beta = 0.0
        best_after = cap_now
        best_p = 0.0
        best_delta = 0.0

        # Generator contingencies
        for i, g in enumerate(cfg.generators):
            if scen.gen_online[t, i]:
                gen_after = gen_now - float(g.capacity_mw)
                line_after = line_now
                firm_after = float(min(gen_after, line_after))
                delta = float(max(0.0, cap_now - firm_after))
                if delta <= 0.0:
                    continue

                lam_step = lam_from_for_mttr(g.forced_outage_rate, g.mttr_hours)
                p_fail = prob_fail_within_h(lam_step, H)

                margin = firm_after - net
                denom = max(1e-6, max(delta, 0.25 * gen_total))
                severity = clamp(-margin / denom, 0.0, 1.0)

                beta_cand = severity * p_fail
                if beta_cand > best_beta:
                    best_beta = beta_cand
                    best_after = firm_after
                    best_p = p_fail
                    best_delta = delta

        # Line contingencies
        for j, l in enumerate(cfg.lines):
            if scen.line_online[t, j]:
                gen_after = gen_now
                line_after = line_now - float(l.capacity_mw)
                firm_after = float(min(gen_after, line_after))
                delta = float(max(0.0, cap_now - firm_after))
                if delta <= 0.0:
                    continue

                lam_step = lam_from_for_mttr(l.forced_outage_rate, l.mttr_hours)
                p_fail = prob_fail_within_h(lam_step, H)

                margin = firm_after - net
                denom = max(1e-6, max(delta, 0.25 * gen_total))
                severity = clamp(-margin / denom, 0.0, 1.0)

                beta_cand = severity * p_fail
                if beta_cand > best_beta:
                    best_beta = beta_cand
                    best_after = firm_after
                    best_p = p_fail
                    best_delta = delta

        beta_raw[t] = best_beta
        firm_after_best[t] = best_after
        p_fail_best[t] = best_p

    # Expected deficit components (forecast-based)
    d_base = np.maximum(0.0, net_load_fc - firm_cap)
    d_n1 = p_fail_best * np.maximum(0.0, net_load_fc - firm_after_best)
    d_exp = d_base + d_n1

    # Accumulated energy risk (forward-looking within window W)
    W = int(max(1, min(cfg.mpc_horizon_hours, cfg.mpc_horizon_hours)))  # keep explicit; can tune
    energy_out = forward_window_sum(d_exp, W) * dt  # MWh
    energy_from_storage = energy_out / max(1e-9, float(cfg.ess.eta_discharge))
    accum_raw = energy_from_storage / max(1e-9, float(cfg.ess.energy_capacity_mwh))  # fraction

    # Normalize with clipping
    p_lo, p_hi = cfg.risk_clip_percentiles
    alpha_n = robust_minmax_scale(percentile_clip(alpha_raw, p_lo, p_hi))
    beta_n = robust_minmax_scale(percentile_clip(beta_raw, p_lo, p_hi))
    sigma_n = robust_minmax_scale(percentile_clip(sigma_raw, p_lo, p_hi))
    accum_n = robust_minmax_scale(percentile_clip(accum_raw, p_lo, p_hi))

    w = cfg.risk_weights
    risk = (
        float(w.w1_alpha) * alpha_n
        + float(w.w2_beta) * beta_n
        + float(w.w3_sigma) * sigma_n
        + float(w.w4_accum) * accum_n
    )

    return np.clip(risk, 0.0, 1.0)


# =========================
# MPC (Global Reuse, Cost-synced, Slack)
# =========================

class RiskBasedMPC:
    def __init__(self, cfg: SimulationConfig):
        self.H = int(cfg.mpc_horizon_hours)
        self.dt = float(cfg.dt_hours)

        self.E_max = float(cfg.ess.energy_capacity_mwh)
        self.P_max = float(cfg.ess.power_capacity_mw)
        self.eta_c = float(cfg.ess.eta_charge)
        self.eta_d = float(cfg.ess.eta_discharge)

        # Cost sync with simulation:
        self.deg_cost = float(cfg.ess.degradation_cost_per_mwh_throughput)
        self.slack_pen = float(cfg.mpc_slack_penalty_per_mwh)

        # Parameters
        self.p_price = cp.Parameter(self.H, name="price")
        self.p_E_init = cp.Parameter(name="E_init")
        self.p_mask = cp.Parameter(self.H, nonneg=True, name="mask")
        self.p_E_req = cp.Parameter(name="E_req")

        # Variables
        self.E = cp.Variable(self.H + 1, name="E")
        self.pc = cp.Variable(self.H, nonneg=True, name="pc")
        self.pd = cp.Variable(self.H, nonneg=True, name="pd")
        self.slack = cp.Variable(self.H, nonneg=True, name="slack")

        cons = []
        cons.append(self.E[0] == self.p_E_init)

        # Relaxed no-simultaneous via sum limit
        cons.append(self.pc + self.pd <= self.P_max)

        for k in range(self.H):
            cons.append(
                self.E[k + 1]
                == self.E[k]
                + self.eta_c * self.pc[k] * self.dt
                - (1.0 / self.eta_d) * self.pd[k] * self.dt
            )

        cons.append(self.E >= 0.0)
        cons.append(self.E <= self.E_max)

        # SOC floor with slack (activated by mask)
        cons.append(self.E[1:] + (1.0 - self.p_mask) * self.E_max + self.slack >= self.p_E_req)

        revenue = cp.sum(cp.multiply(self.p_price, self.pd)) * self.dt
        cost = cp.sum(cp.multiply(self.p_price, self.pc)) * self.dt
        degradation = self.deg_cost * cp.sum(self.pc + self.pd) * self.dt
        slack_cost = self.slack_pen * cp.sum(self.slack)

        obj = cp.Maximize(revenue - cost - degradation - slack_cost)
        self.prob = cp.Problem(obj, cons)

        self._solvers = []
        installed = set(cp.installed_solvers())
        for s in ["GUROBI", "CLARABEL", "OSQP", "ECOS"]:
            if s in installed:
                self._solvers.append(getattr(cp, s))
        if not self._solvers:
            self._solvers = [cp.ECOS]

    def solve(self, price_fc: np.ndarray, risk_fc: np.ndarray, E_now: float, params: RiskBasedPolicyParams) -> Tuple[float, float]:
        self.p_price.value = np.asarray(price_fc, dtype=float)
        self.p_E_init.value = float(E_now)

        # Deadline logic: find earliest tau2, else earliest tau1
        tau1 = float(params.tau1)
        tau2 = float(params.tau2)

        t2 = next((k for k, r in enumerate(risk_fc) if float(r) >= tau2), None)
        t1 = None
        if t2 is None:
            t1 = next((k for k, r in enumerate(risk_fc) if float(r) >= tau1), None)

        mask = np.zeros(self.H, dtype=float)
        req = 0.0

        if t2 is not None:
            req = self.E_max * float(params.s2) / 100.0
            mask[t2:] = 1.0
        elif t1 is not None:
            req = self.E_max * float(params.s1) / 100.0
            mask[t1:] = 1.0

        self.p_mask.value = mask
        self.p_E_req.value = float(req)

        for solver in self._solvers:
            try:
                with suppress_stdout_stderr():
                    self.prob.solve(solver=solver, warm_start=True, verbose=False)
                if self.prob.status in ["optimal", "optimal_inaccurate"]:
                    pc0 = float(self.pc.value[0]) if self.pc.value is not None else 0.0
                    pd0 = float(self.pd.value[0]) if self.pd.value is not None else 0.0
                    if abs(pc0) < 1e-8:
                        pc0 = 0.0
                    if abs(pd0) < 1e-8:
                        pd0 = 0.0
                    return pc0, pd0
            except Exception:
                continue

        return 0.0, 0.0


# =========================
# ESS Physical Model (Simulation)
# =========================

class ESSModel:
    def __init__(self, cfg: ESSConfig):
        self.cfg = cfg
        self.E_max = float(cfg.energy_capacity_mwh)
        self.P_max = float(cfg.power_capacity_mw)
        self.eta_c = float(cfg.eta_charge)
        self.eta_d = float(cfg.eta_discharge)
        self.deg_cost = float(cfg.degradation_cost_per_mwh_throughput)

        self.E = clamp(self.E_max * float(cfg.soc_init_percent) / 100.0, 0.0, self.E_max)

    def soc_percent(self) -> float:
        return 100.0 * safe_div(self.E, self.E_max, 0.0)

    def step(self, pc: float, pd: float, dt: float) -> Tuple[float, float]:
        pc = float(max(0.0, pc))
        pd = float(max(0.0, pd))

        # Enforce sum limit (consistent with MPC relaxation)
        if pc + pd > self.P_max:
            if pc >= pd:
                pc = self.P_max
                pd = 0.0
            else:
                pd = self.P_max
                pc = 0.0

        # Energy constraints
        max_in = safe_div(self.E_max - self.E, self.eta_c * dt, 0.0)
        max_out = safe_div(self.E * self.eta_d, dt, 0.0)

        pc_act = min(pc, self.P_max, max_in)
        pd_act = min(pd, self.P_max, max_out)

        self.E = clamp(self.E + self.eta_c * pc_act * dt - (pd_act / self.eta_d) * dt, 0.0, self.E_max)
        return float(pc_act), float(pd_act)

    def emergency_discharge(self, deficit: float, dt: float) -> float:
        pd_cmd = min(float(deficit), self.P_max)
        _, pd_act = self.step(0.0, pd_cmd, dt)
        return float(pd_act)


# =========================
# Simulation Runner
# =========================

class SimulationRunner:
    def __init__(self, cfg: SimulationConfig, scen: ScenarioData, mpc: Optional[RiskBasedMPC] = None):
        self.cfg = cfg
        self.scen = scen

        # SSOT capacity once
        self.cap = compute_capacity_series(cfg, scen)

        # Risk series (SSOT-compliant)
        self.risk = compute_risk_series(cfg, scen, self.cap)

        # Supply series (actual renewable)
        self.firm_supply = self.cap.firm_cap_mw + scen.renewable_actual_mw

        # Benchmarks (SSOT thresholds)
        self.tou_low = float(np.quantile(scen.price_per_mwh, cfg.tou_low_price_quantile))
        self.tou_high = float(np.quantile(scen.price_per_mwh, cfg.tou_high_price_quantile))
        self.re_low = float(np.quantile(scen.renewable_actual_mw, cfg.renewable_low_quantile))
        self.re_high = float(np.quantile(scen.renewable_actual_mw, cfg.renewable_high_quantile))
        self.price_mid = float(np.quantile(scen.price_per_mwh, 0.50))

        self.forecaster = RiskForecaster(cfg)

        # Global MPC reuse
        self.mpc = mpc if mpc is not None else RiskBasedMPC(cfg)

    def _tou_command(self, price: float) -> Tuple[float, float]:
        if price <= self.tou_low:
            return self.cfg.ess.power_capacity_mw, 0.0
        if price >= self.tou_high:
            return 0.0, self.cfg.ess.power_capacity_mw
        return 0.0, 0.0

    def _renewable_command(self, ren: float, price: float) -> Tuple[float, float]:
        if ren >= self.re_high:
            return self.cfg.ess.power_capacity_mw, 0.0
        if ren <= self.re_low and price >= self.price_mid:
            return 0.0, self.cfg.ess.power_capacity_mw
        return 0.0, 0.0

    def run(self, strategy: Strategy, params: Optional[RiskBasedPolicyParams], rng: Optional[np.random.Generator]) -> SimulationResult:
        T = int(len(self.scen.load_mw))
        dt = float(self.cfg.dt_hours)

        ess = ESSModel(self.cfg.ess) if strategy != Strategy.NO_ESS else None

        # arrays
        soc = np.zeros(T, dtype=float)
        pc_arr = np.zeros(T, dtype=float)
        pd_arr = np.zeros(T, dtype=float)
        emg_arr = np.zeros(T, dtype=float)

        deficit_pre = np.zeros(T, dtype=float)
        unserved = np.zeros(T, dtype=float)

        profit = 0.0

        for t in range(T):
            load = float(self.scen.load_mw[t])
            ren = float(self.scen.renewable_actual_mw[t])
            price = float(self.scen.price_per_mwh[t])

            if ess is not None:
                soc[t] = ess.soc_percent()

            # Deficit before ESS (SSOT supply)
            supply = float(self.firm_supply[t])
            d_pre = max(0.0, load - supply)
            deficit_pre[t] = d_pre

            d_left = d_pre

            # Emergency discharge always first
            if ess is not None and d_left > 0.0:
                inj = ess.emergency_discharge(d_left, dt)
                emg_arr[t] = inj

                # revenue - degradation
                profit += price * inj * dt
                profit -= ess.deg_cost * inj * dt

                d_left = max(0.0, d_left - inj)

            if d_left > 0.0:
                unserved[t] = d_left

            # Arbitrage only when NO deficit pre (system is adequate without ESS)
            if ess is not None and d_pre <= 1e-12:
                pc_cmd, pd_cmd = 0.0, 0.0

                if strategy == Strategy.TOU:
                    pc_cmd, pd_cmd = self._tou_command(price)

                elif strategy == Strategy.RENEWABLE_BASED:
                    pc_cmd, pd_cmd = self._renewable_command(ren, price)

                elif strategy == Strategy.RISK_BASED:
                    if params is None:
                        raise ValueError("RiskBased requires params")

                    H = int(self.cfg.mpc_horizon_hours)

                    # price forecast (oracle)
                    idx_end = min(T, t + H)
                    price_fc = self.scen.price_per_mwh[t:idx_end].astype(float)
                    if price_fc.size < H:
                        price_fc = np.pad(price_fc, (0, H - price_fc.size), mode="edge")

                    # risk forecast (configurable)
                    risk_fc = self.forecaster.forecast(self.risk, t, H, rng)

                    # Speed: if horizon max risk < tau1 => skip MPC and use TOU
                    if float(np.max(risk_fc)) < float(params.tau1):
                        pc_cmd, pd_cmd = self._tou_command(price)
                    else:
                        pc_cmd, pd_cmd = self.mpc.solve(price_fc, risk_fc, ess.E, params)

                # Apply ESS step
                pc_act, pd_act = ess.step(pc_cmd, pd_cmd, dt)
                pc_arr[t] = pc_act
                pd_arr[t] = pd_act

                # arbitrage profit - degradation
                profit += (pd_act - pc_act) * price * dt
                profit -= ess.deg_cost * (pc_act + pd_act) * dt

        lole = float(np.sum(unserved > 1e-9) * dt)
        eens = float(np.sum(unserved) * dt)

        return SimulationResult(
            strategy=strategy,
            params=params,
            load_mw=self.scen.load_mw.copy(),
            renewable_actual_mw=self.scen.renewable_actual_mw.copy(),
            firm_cap_mw=self.cap.firm_cap_mw.copy(),
            firm_supply_mw=self.firm_supply.copy(),
            price_per_mwh=self.scen.price_per_mwh.copy(),
            risk_series=self.risk.copy(),
            deficit_pre_ess_mw=deficit_pre,
            unserved_mw=unserved,
            soc_percent=soc,
            p_charge_mw=pc_arr,
            p_discharge_mw=pd_arr,
            p_emergency_mw=emg_arr,
            lole_hours=lole,
            eens_mwh=eens,
            profit=float(profit),
        )


# =========================
# Optimization
# =========================

def optimize_params(cfg: SimulationConfig, mpc: RiskBasedMPC) -> RiskBasedPolicyParams:
    total = int(cfg.offline_n_calls)
    counter = {"n": 0}

    def objective(x: List[float]) -> float:
        counter["n"] += 1
        print_progress("[Offline Opt]", counter["n"], total)

        tau1, tau2, s1, s2 = map(float, x)
        if tau1 >= tau2 or s1 >= s2:
            return 1e9

        # bounds hard-check (skopt respects, but keep safe)
        if not (cfg.tau1_bounds[0] <= tau1 <= cfg.tau1_bounds[1]):
            return 1e9
        if not (cfg.tau2_bounds[0] <= tau2 <= cfg.tau2_bounds[1]):
            return 1e9
        if not (cfg.s1_bounds[0] <= s1 <= cfg.s1_bounds[1]):
            return 1e9
        if not (cfg.s2_bounds[0] <= s2 <= cfg.s2_bounds[1]):
            return 1e9

        params = RiskBasedPolicyParams(tau1=tau1, tau2=tau2, s1=s1, s2=s2)

        rng = np.random.default_rng(cfg.random_seed)

        avg_lole = 0.0
        avg_profit = 0.0

        for _ in range(int(cfg.offline_mc_scenarios)):
            scen = generate_synthetic_scenario(cfg, rng, cfg.optimization_horizon_hours)
            sim_rng = np.random.default_rng(rng.integers(0, 2**32))
            runner = SimulationRunner(cfg, scen, mpc)
            res = runner.run(Strategy.RISK_BASED, params, sim_rng)

            avg_lole += res.lole_hours
            avg_profit += res.profit

        avg_lole /= float(cfg.offline_mc_scenarios)
        avg_profit /= float(cfg.offline_mc_scenarios)

        w = cfg.optimization_weights
        return float(w.lole_weight * avg_lole - w.profit_weight * avg_profit)

    space = [
        Real(*cfg.tau1_bounds),
        Real(*cfg.tau2_bounds),
        Real(*cfg.s1_bounds),
        Real(*cfg.s2_bounds),
    ]

    res = gp_minimize(objective, space, n_calls=int(cfg.offline_n_calls), random_state=int(cfg.random_seed))
    print()
    return RiskBasedPolicyParams(*map(float, res.x))


# =========================
# Main
# =========================

def main():
    cfg = SimulationConfig(
        horizon_hours=8760,
        dt_hours=1.0,
        mpc_horizon_hours=6,

        generators=[GeneratorUnit("G", 200.0, 0.04, 4.0)] * 4,
        lines=[TransmissionLine("L", 600.0, 0.005, 4.0)] * 2,

        base_load_mw=580.0,
        daily_load_swing_mw=120.0,
        load_noise_std_mw=15.0,

        renewable_capacity_mw=300.0,
        solar_share=0.6,
        wind_share=0.4,
        renewable_noise_std_mw=10.0,
        renewable_forecast_base_std_mw=8.0,
        renewable_forecast_std_factor=0.05,

        price_offpeak=40.0,
        price_mid=80.0,
        price_peak=160.0,
        tou_peak_hours=[16, 17, 18, 19, 20, 21],
        tou_mid_hours=[7, 8, 9, 10, 11, 12, 13, 14, 15, 22],
        tou_offpeak_hours=[0, 1, 2, 3, 4, 5, 6, 23],

        risk_weights=RiskWeights(w1_alpha=0.30, w2_beta=0.30, w3_sigma=0.20, w4_accum=0.20),
        risk_clip_percentiles=(1.0, 99.0),
        risk_forecast_mode="oracle",
        risk_forecast_noise_std=0.05,
        ar1_window=24,

        ess=ESSConfig(
            energy_capacity_mwh=400.0,   # placeholder, updated in Phase 1
            power_capacity_mw=100.0,     # placeholder, updated in Phase 1
            soc_init_percent=50.0,
            eta_charge=0.95,
            eta_discharge=0.95,
            degradation_cost_per_mwh_throughput=2.0,
            ramp_limit_mw_per_h=100.0
        ),

        offline_n_calls=50,
        offline_mc_scenarios=5,
        optimization_horizon_hours=720,
        random_seed=42,
        optimization_weights=OptimizationWeights(lole_weight=200.0, profit_weight=0.002),

        tau1_bounds=(0.10, 0.60),
        tau2_bounds=(0.61, 0.99),
        s1_bounds=(10.0, 90.0),
        s2_bounds=(10.0, 100.0),

        mpc_use_binary=False,
        mpc_big_m=1e3,
        mpc_cycle_penalty_per_mwh=0.0,
        mpc_slack_penalty_per_mwh=1000.0,

        tou_low_price_quantile=0.20,
        tou_high_price_quantile=0.80,
        renewable_high_quantile=0.75,
        renewable_low_quantile=0.25,

        mc_eval_scenarios=10,
        mc_eval_seed=20260123,
        mc_show_progress=True,
    )

    # -------------------------
    # Phase 1: Data-driven sizing (NoESS)
    # -------------------------
    print("=== Phase 1: Data-Driven Sizing (NoESS) ===")
    rng = np.random.default_rng(cfg.random_seed)

    all_deficits: List[np.ndarray] = []
    n_sizing = 50
    print("Collecting deficit statistics...")

    for i in range(n_sizing):
        if cfg.mc_show_progress:
            print_progress("[Sizing]", i + 1, n_sizing)

        scen = generate_synthetic_scenario(cfg, rng, cfg.horizon_hours)
        cap = compute_capacity_series(cfg, scen)
        firm_supply = cap.firm_cap_mw + scen.renewable_actual_mw
        deficit = np.maximum(0.0, scen.load_mw - firm_supply)
        all_deficits.append(deficit)

    print()
    sizing_info = analyze_deficit_events(all_deficits, cfg.dt_hours, cfg.ess.eta_discharge, q_target=0.99)

    rec_P = float(sizing_info["p_req_q"])
    rec_E = float(sizing_info["e_req_q"])
    joint = float(sizing_info["joint_coverage"])

    print(f"\n[Sizing Result] Joint Coverage: {joint:.2%}")
    print(f"Recommended P_max: {rec_P:.2f} MW, E_max: {rec_E:.2f} MWh")

    new_ess = dataclasses.replace(
        cfg.ess,
        power_capacity_mw=rec_P,
        energy_capacity_mwh=rec_E,
        ramp_limit_mw_per_h=rec_P
    )
    cfg = dataclasses.replace(cfg, ess=new_ess)

    # Global MPC (reused)
    global_mpc = RiskBasedMPC(cfg)

    # -------------------------
    # Phase 2: Optimize params on shortened horizon
    # -------------------------
    print("\n=== Phase 2: Offline Optimization (Bayesian) ===")
    print(f"Optimization Horizon: {cfg.optimization_horizon_hours} hours")
    best_p = optimize_params(cfg, global_mpc)
    print(f"Best Params: {best_p}")

    # -------------------------
    # Phase 3: Monte Carlo Evaluation (Full Year)
    # -------------------------
    print("\n=== Phase 3: Monte Carlo Evaluation (8760h) ===")
    rng_eval = np.random.default_rng(cfg.mc_eval_seed)

    results_map: Dict[str, List[SimulationResult]] = {s.value: [] for s in Strategy}
    time_metrics_list: List[Dict[str, float]] = []
    event_metrics_list: List[EventMetrics] = []

    last_riskbased: Optional[SimulationResult] = None

    for i in range(int(cfg.mc_eval_scenarios)):
        if cfg.mc_show_progress:
            print_progress("[MC Eval]", i + 1, int(cfg.mc_eval_scenarios))

        scen = generate_synthetic_scenario(cfg, rng_eval, cfg.horizon_hours)
        runner = SimulationRunner(cfg, scen, global_mpc)

        sim_seed = int(rng_eval.integers(0, 2**32))

        for strat in Strategy:
            sim_rng = np.random.default_rng(sim_seed)
            res = runner.run(strat, best_p if strat == Strategy.RISK_BASED else None, sim_rng)
            results_map[strat.value].append(res)

            if strat == Strategy.RISK_BASED:
                last_riskbased = res
                time_metrics_list.append(compute_time_metrics(res.risk_series, res.deficit_pre_ess_mw, res.unserved_mw, best_p.tau2))
                event_metrics_list.append(evaluate_events(res.risk_series, res.deficit_pre_ess_mw, best_p.tau2, lookback_h=6, dt=cfg.dt_hours))

    print()

    # Summary table
    print(f"{'Case':<16} | {'LOLE(h)':<10} | {'EENS(MWh)':<12} | {'Profit($)':<15}")
    print("-" * 65)
    for k in [s.value for s in Strategy]:
        lole_mean = float(np.mean([r.lole_hours for r in results_map[k]]))
        eens_mean = float(np.mean([r.eens_mwh for r in results_map[k]]))
        prof_mean = float(np.mean([r.profit for r in results_map[k]]))
        print(f"{k:<16} | {lole_mean:10.2f} | {eens_mean:12.2f} | {prof_mean:15,.0f}")

    # Metrics report
    avg_auc = float(np.nanmean([m["AUC"] for m in time_metrics_list])) if time_metrics_list else float("nan")
    avg_prec = float(np.nanmean([m["Precision"] for m in time_metrics_list])) if time_metrics_list else float("nan")
    avg_rec = float(np.nanmean([m["Recall"] for m in time_metrics_list])) if time_metrics_list else float("nan")
    avg_res = float(np.nanmean([m["Resolved"] for m in time_metrics_list])) if time_metrics_list else float("nan")

    avg_evt_rec = float(np.mean([m.event_recall for m in event_metrics_list])) if event_metrics_list else float("nan")
    avg_lead = float(np.mean([m.mean_lead_time_h for m in event_metrics_list])) if event_metrics_list else float("nan")

    print("\n[Tau2 Metrics]")
    print(f"Time-Based  -> AUC: {avg_auc:.3f}, Precision: {avg_prec:.3f}, Recall: {avg_rec:.3f}, Resolved: {avg_res:.2%}")
    print(f"Event-Based -> Event Recall: {avg_evt_rec:.2%}, Avg Lead Time: {avg_lead:.2f} h")

    # Export + Plot
    if last_riskbased is not None:
        # 1. 전체 데이터(8760시간) DataFrame 생성
        df = pd.DataFrame({
            "Load_MW": last_riskbased.load_mw,
            "Renewable_Actual_MW": last_riskbased.renewable_actual_mw,
            "FirmCap_MW": last_riskbased.firm_cap_mw,
            "FirmSupply_MW": last_riskbased.firm_supply_mw,
            "Price_per_MWh": last_riskbased.price_per_mwh,
            "Risk": last_riskbased.risk_series,
            "Deficit_PreESS_MW": last_riskbased.deficit_pre_ess_mw,
            "Unserved_MW": last_riskbased.unserved_mw,
            "SOC_percent": last_riskbased.soc_percent,
            "P_charge_MW": last_riskbased.p_charge_mw,
            "P_discharge_MW": last_riskbased.p_discharge_mw,
            "P_emergency_MW": last_riskbased.p_emergency_mw,
        })

        # 2. 전체 데이터 저장 (기존)
        out_xlsx = "simulation_result_full.xlsx"
        df.to_excel(out_xlsx, index=False)
        print(f"\nSaved Full Year Data: '{out_xlsx}'")

        # 3. [추가됨] 1주일(168시간) 데이터만 잘라서 따로 저장
        # 데이터가 168시간보다 적으면 있는 만큼만 저장됩니다.
        n_week = min(168, len(df))
        df_week = df.iloc[:n_week]

        out_week_xlsx = "simulation_result_week1.xlsx"
        df_week.to_excel(out_week_xlsx, index=False)
        print(f"Saved First Week Data: '{out_week_xlsx}'")

        # 4. 그래프 그리기 (기존과 동일)
        plt.figure(figsize=(11, 7))
        plt.subplot(3, 1, 1)
        plt.plot(df["Deficit_PreESS_MW"].values[:n_week], label="Deficit_PreESS")
        plt.plot(df["P_emergency_MW"].values[:n_week], label="Emergency", linestyle="--")
        plt.legend()
        plt.title("Deficit & Emergency (First Week)")

        plt.subplot(3, 1, 2)
        plt.plot(df["SOC_percent"].values[:n_week], label="SOC(%)")
        plt.legend()
        plt.title("SOC (%)")

        plt.subplot(3, 1, 3)
        plt.plot(df["Risk"].values[:n_week], label="Risk")
        plt.legend()
        plt.title("Risk (First Week)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
