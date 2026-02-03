# -*- coding: utf-8 -*-
"""
ESS Sizing Quantile Sensitivity Wrapper
- Compare sizing quantile targets: 0.95 vs 0.98 vs 0.99
- Metrics: LOLE, Resolved (deficit-hour resolved), Cost (OpCost / CAPEX / Total)
- Fairness: same sizing scenarios + same evaluation scenarios across quantiles (fixed seeds)
- Based on: ess_final.py (Final of Finals code)

Usage:
  1) Save your final code as: ess_final.py
  2) Run:
        python sizing_sensitivity_wrapper.py
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# IMPORTANT: change this import if your module file name is different.
import ess_final as core


# =========================
# Experiment Controls
# =========================

SIZING_QUANTILES = [0.95, 0.98, 0.99]

N_SIZING_SCENARIOS = 50   # how many full-year scenarios for sizing event distribution
N_EVAL_SCENARIOS = 10     # MC evaluation scenarios (full year)
EXPORT_EXCEL = True
EXPORT_FILENAME = "sizing_quantile_sensitivity.xlsx"

# Capital cost model (OPTIONAL)
# If you don't want CAPEX, leave them as 0.
# Example (NOT a recommendation): CAPEX_PER_MW = 80_000 ($/MW), CAPEX_PER_MWH = 200_000 ($/MWh)
CAPEX_PER_MW = 0.0
CAPEX_PER_MWH = 0.0

EPS = 1e-9


# =========================
# Helpers
# =========================

def build_base_config() -> core.SimulationConfig:
    """
    Must match (or be compatible with) the config used in ess_final.py.
    You can adjust these defaults as needed.
    """
    cfg = core.SimulationConfig(
        horizon_hours=8760,
        dt_hours=1.0,
        mpc_horizon_hours=6,

        generators=[core.GeneratorUnit("G", 200.0, 0.04, 4.0)] * 4,
        lines=[core.TransmissionLine("L", 600.0, 0.005, 4.0)] * 2,

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

        risk_weights=core.RiskWeights(w1_alpha=0.30, w2_beta=0.30, w3_sigma=0.20, w4_accum=0.20),
        risk_clip_percentiles=(1.0, 99.0),
        risk_forecast_mode="oracle",
        risk_forecast_noise_std=0.05,
        ar1_window=24,

        # Placeholder ESS (will be overwritten by sizing)
        ess=core.ESSConfig(
            energy_capacity_mwh=400.0,
            power_capacity_mw=100.0,
            soc_init_percent=50.0,
            eta_charge=0.95,
            eta_discharge=0.95,
            degradation_cost_per_mwh_throughput=2.0,
            ramp_limit_mw_per_h=100.0,
        ),

        offline_n_calls=50,
        offline_mc_scenarios=5,
        optimization_horizon_hours=720,
        random_seed=42,
        optimization_weights=core.OptimizationWeights(lole_weight=200.0, profit_weight=0.002),

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

        mc_eval_scenarios=N_EVAL_SCENARIOS,
        mc_eval_seed=20260123,
        mc_show_progress=True,
    )
    return cfg


def summarize(values: List[float], z: float = 1.96) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
    half = float(z * std / np.sqrt(max(1, x.size)))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def resolved_hour_rate(deficit_pre: np.ndarray, unserved: np.ndarray, eps: float = EPS) -> float:
    deficit_pre = np.asarray(deficit_pre, dtype=float)
    unserved = np.asarray(unserved, dtype=float)
    mask_def = deficit_pre > eps
    denom = int(np.sum(mask_def))
    if denom == 0:
        return float("nan")
    resolved = int(np.sum(mask_def & (unserved <= eps)))
    return float(resolved / denom)


def resolved_energy_rate(deficit_pre: np.ndarray, unserved: np.ndarray, dt: float, eps: float = EPS) -> float:
    deficit_pre = np.asarray(deficit_pre, dtype=float)
    unserved = np.asarray(unserved, dtype=float)
    e_def = float(np.sum(deficit_pre) * dt)
    if e_def <= eps:
        return float("nan")
    e_uns = float(np.sum(unserved) * dt)
    return float(1.0 - (e_uns / e_def))


def compute_operating_cost(res: core.SimulationResult, cfg: core.SimulationConfig) -> Dict[str, float]:
    """
    Provide multiple cost views:
      - DegradationCost: deg_cost * throughput
      - NetEnergyCost: purchase - sale (emergency excluded)
      - TotalOpCost: NetEnergyCost + DegradationCost
      - NetCostFromProfit: -profit (as encoded by simulation objective/value)
    """
    dt = float(cfg.dt_hours)
    deg_cost = float(cfg.ess.degradation_cost_per_mwh_throughput)

    pc = np.asarray(res.p_charge_mw, dtype=float)
    pd = np.asarray(res.p_discharge_mw, dtype=float)
    emg = np.asarray(res.p_emergency_mw, dtype=float)
    price = np.asarray(res.price_per_mwh, dtype=float)

    throughput_mwh = float(np.sum((pc + pd + emg) * dt))
    degradation_cost = float(deg_cost * throughput_mwh)

    purchase = float(np.sum(pc * price * dt))
    sale = float(np.sum(pd * price * dt))  # emergency is reliability action, not market sale here
    net_energy_cost = float(purchase - sale)

    total_op_cost = float(net_energy_cost + degradation_cost)
    net_cost_from_profit = float(-res.profit)

    return {
        "Throughput_MWh": throughput_mwh,
        "DegradationCost_$": degradation_cost,
        "Purchase_$": purchase,
        "Sale_$": sale,
        "NetEnergyCost_$": net_energy_cost,
        "TotalOpCost_$": total_op_cost,
        "NetCostFromProfit_$": net_cost_from_profit,
    }


def capex_estimate(P_max: float, E_max: float) -> float:
    return float(CAPEX_PER_MW * P_max + CAPEX_PER_MWH * E_max)


def collect_sizing_deficits(cfg_base: core.SimulationConfig, sizing_seeds: List[int]) -> List[np.ndarray]:
    """
    Vectorized deficit calculation for sizing, independent of ESS.
    """
    all_def = []
    for seed in sizing_seeds:
        rng = np.random.default_rng(int(seed))
        scen = core.generate_synthetic_scenario(cfg_base, rng, cfg_base.horizon_hours)
        cap = core.compute_capacity_series(cfg_base, scen)
        firm_supply = cap.firm_cap_mw + scen.renewable_actual_mw
        deficit = np.maximum(0.0, scen.load_mw - firm_supply)
        all_def.append(deficit)
    return all_def


def collect_event_arrays(all_deficits: List[np.ndarray], dt: float, eta_dis: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract deficit events ONCE, then reuse for multiple quantile targets.
    """
    p_list: List[float] = []
    e_list: List[float] = []
    for d in all_deficits:
        events = core.extract_deficit_events(d, dt, eta_dis)
        for ev in events:
            p_list.append(float(ev.peak_mw))
            e_list.append(float(ev.energy_from_ess_mwh))
    return np.asarray(p_list, dtype=float), np.asarray(e_list, dtype=float)


def sizing_from_events(p_reqs: np.ndarray, e_reqs: np.ndarray, q_target: float) -> Dict[str, float]:
    """
    Same logic as analyze_deficit_events(), but without re-extracting events.
    Find minimal q_candidate >= q_target that achieves joint_coverage >= q_target.
    """
    if p_reqs.size == 0 or e_reqs.size == 0:
        return {"p_req_q": 0.0, "e_req_q": 0.0, "joint_coverage": float("nan"), "q_used": float("nan")}

    p_q = 0.0
    e_q = 0.0
    joint = 0.0
    q_used = float("nan")

    for q_candidate in np.linspace(q_target, 0.999, 25):
        p_val = float(np.quantile(p_reqs, q_candidate))
        e_val = float(np.quantile(e_reqs, q_candidate))
        joint_val = float(np.mean((p_reqs <= p_val) & (e_reqs <= e_val)))
        if joint_val >= q_target:
            p_q, e_q, joint = p_val, e_val, joint_val
            q_used = float(q_candidate)
            break

    if p_q <= 0.0:
        p_q = float(np.quantile(p_reqs, 0.99))
        e_q = float(np.quantile(e_reqs, 0.99))
        joint = float(np.mean((p_reqs <= p_q) & (e_reqs <= e_q)))
        q_used = 0.99

    return {"p_req_q": p_q, "e_req_q": e_q, "joint_coverage": joint, "q_used": q_used}


def pre_generate_eval_scenarios(cfg_base: core.SimulationConfig, eval_seeds: List[int]) -> List[core.ScenarioData]:
    scenarios = []
    for seed in eval_seeds:
        rng = np.random.default_rng(int(seed))
        scen = core.generate_synthetic_scenario(cfg_base, rng, cfg_base.horizon_hours)
        scenarios.append(scen)
    return scenarios


def compute_noess_baseline(eval_scenarios: List[core.ScenarioData], cfg_base: core.SimulationConfig) -> Dict[str, Dict[str, float]]:
    """
    Baseline metrics for NoESS using pure vectorized deficit calculation (no runner).
    """
    dt = float(cfg_base.dt_hours)
    lole_list = []
    eens_list = []
    deficit_hours_list = []

    for scen in eval_scenarios:
        cap = core.compute_capacity_series(cfg_base, scen)
        firm_supply = cap.firm_cap_mw + scen.renewable_actual_mw
        deficit = np.maximum(0.0, scen.load_mw - firm_supply)

        lole = float(np.sum(deficit > EPS) * dt)
        eens = float(np.sum(deficit) * dt)
        deficit_hours = float(np.sum(deficit > EPS) * dt)

        lole_list.append(lole)
        eens_list.append(eens)
        deficit_hours_list.append(deficit_hours)

    return {
        "LOLE": summarize(lole_list),
        "EENS": summarize(eens_list),
        "DeficitHours": summarize(deficit_hours_list),
    }


# =========================
# Main Experiment
# =========================

def main():
    cfg_base = build_base_config()

    # ---- fixed seeds (fair comparison across quantiles) ----
    base_seed = int(cfg_base.random_seed)
    sizing_seeds = [base_seed + 10_000 + i for i in range(N_SIZING_SCENARIOS)]
    eval_seeds = [int(cfg_base.mc_eval_seed) + 20_000 + i for i in range(N_EVAL_SCENARIOS)]
    # risk forecast noise seeds (kept identical across quantiles)
    noise_seeds = [int(cfg_base.mc_eval_seed) + 30_000 + i for i in range(N_EVAL_SCENARIOS)]

    # ---- Phase 0: Precompute sizing events once ----
    print("=== Precompute Sizing Deficit Events (once) ===")
    all_deficits = collect_sizing_deficits(cfg_base, sizing_seeds)
    p_reqs, e_reqs = collect_event_arrays(all_deficits, cfg_base.dt_hours, cfg_base.ess.eta_discharge)
    print(f"Total sizing events: {p_reqs.size}")

    # ---- Phase 0.5: Pre-generate evaluation scenarios once ----
    print("\n=== Pre-generate Evaluation Scenarios (once) ===")
    eval_scenarios = pre_generate_eval_scenarios(cfg_base, eval_seeds)

    # Baseline (NoESS) once
    print("\n=== NoESS Baseline (fixed across quantiles) ===")
    base_metrics = compute_noess_baseline(eval_scenarios, cfg_base)
    print(f"NoESS LOLE mean: {base_metrics['LOLE']['mean']:.2f} h (CI [{base_metrics['LOLE']['ci_low']:.2f}, {base_metrics['LOLE']['ci_high']:.2f}])")
    print(f"NoESS EENS mean: {base_metrics['EENS']['mean']:.2f} MWh")

    # ---- Quantile loop ----
    rows: List[Dict[str, float]] = []

    for q in SIZING_QUANTILES:
        print("\n" + "=" * 80)
        print(f"=== Quantile Target: {q:.2f} ===")

        # Phase 1: Sizing from events
        s = sizing_from_events(p_reqs, e_reqs, q_target=float(q))
        rec_P = float(s["p_req_q"])
        rec_E = float(s["e_req_q"])
        joint = float(s["joint_coverage"])
        q_used = float(s["q_used"])

        print(f"[Sizing] q_target={q:.2f}, q_used={q_used:.4f}, joint_coverage={joint:.2%}")
        print(f"[Sizing] P_max={rec_P:.2f} MW, E_max={rec_E:.2f} MWh")

        # Update config with sized ESS
        new_ess = dataclasses.replace(
            cfg_base.ess,
            power_capacity_mw=rec_P,
            energy_capacity_mwh=rec_E,
            ramp_limit_mw_per_h=rec_P,
        )
        cfg_q = dataclasses.replace(cfg_base, ess=new_ess)

        # CAPEX estimate (optional)
        capex = capex_estimate(rec_P, rec_E)

        # Phase 2: Optimize params (per-quantile)
        print("\n[Optimization] Start gp_minimize for this quantile...")
        global_mpc = core.RiskBasedMPC(cfg_q)
        best_p = core.optimize_params(cfg_q, global_mpc)
        print(f"[Optimization] Best Params: {best_p}")

        # Phase 3: Evaluate RiskBased on SAME scenarios
        lole_list = []
        eens_list = []
        resolved_h_list = []
        resolved_e_list = []
        profit_list = []
        op_cost_list = []
        capex_list = []
        total_cost_list = []
        net_cost_from_profit_list = []

        # Optional: tau2-alarm resolved (from core.compute_time_metrics)
        tau2_resolved_list = []

        for i, scen in enumerate(eval_scenarios):
            sim_rng = np.random.default_rng(int(noise_seeds[i]))  # keep identical across quantiles
            runner = core.SimulationRunner(cfg_q, scen, global_mpc)
            res = runner.run(core.Strategy.RISK_BASED, best_p, sim_rng)

            lole_list.append(float(res.lole_hours))
            eens_list.append(float(res.eens_mwh))

            rh = resolved_hour_rate(res.deficit_pre_ess_mw, res.unserved_mw, eps=EPS)
            re = resolved_energy_rate(res.deficit_pre_ess_mw, res.unserved_mw, dt=cfg_q.dt_hours, eps=EPS)
            resolved_h_list.append(float(rh) if not np.isnan(rh) else float("nan"))
            resolved_e_list.append(float(re) if not np.isnan(re) else float("nan"))

            profit_list.append(float(res.profit))

            cost_info = compute_operating_cost(res, cfg_q)
            op_cost = float(cost_info["TotalOpCost_$"])
            op_cost_list.append(op_cost)

            capex_list.append(capex)
            total_cost_list.append(float(op_cost + capex))

            net_cost_from_profit_list.append(float(cost_info["NetCostFromProfit_$"]))

            # Tau2 time-metric resolved (alarm-conditional, optional)
            tm = core.compute_time_metrics(res.risk_series, res.deficit_pre_ess_mw, res.unserved_mw, best_p.tau2)
            tau2_resolved_list.append(float(tm.get("Resolved", float("nan"))))

        lole_s = summarize(lole_list)
        eens_s = summarize(eens_list)
        rh_s = summarize([x for x in resolved_h_list if not np.isnan(x)])
        re_s = summarize([x for x in resolved_e_list if not np.isnan(x)])
        profit_s = summarize(profit_list)
        opcost_s = summarize(op_cost_list)
        totalcost_s = summarize(total_cost_list)
        netcost_from_profit_s = summarize(net_cost_from_profit_list)
        tau2_res_s = summarize([x for x in tau2_resolved_list if not np.isnan(x)])

        print("\n[Evaluation Summary] (RiskBased)")
        print(f"LOLE mean: {lole_s['mean']:.2f} h (CI [{lole_s['ci_low']:.2f}, {lole_s['ci_high']:.2f}])")
        print(f"Resolved(hour) mean: {rh_s['mean']:.2%} (CI [{rh_s['ci_low']:.2%}, {rh_s['ci_high']:.2%}])")
        print(f"Resolved(energy) mean: {re_s['mean']:.2%}")
        print(f"OpCost mean: {opcost_s['mean']:.0f} $, CAPEX: {capex:.0f} $, TotalCost mean: {totalcost_s['mean']:.0f} $")
        print(f"Profit mean: {profit_s['mean']:.0f} $, NetCostFromProfit mean: {netcost_from_profit_s['mean']:.0f} $")
        print(f"Tau2-Resolved(alarm-conditional) mean: {tau2_res_s['mean']:.2%}")

        row = {
            "SizingQuantileTarget": float(q),
            "SizingQuantileUsed": float(q_used),
            "JointCoverage": float(joint),

            "P_max_MW": float(rec_P),
            "E_max_MWh": float(rec_E),

            "tau1": float(best_p.tau1),
            "tau2": float(best_p.tau2),
            "s1": float(best_p.s1),
            "s2": float(best_p.s2),

            "LOLE_mean_h": float(lole_s["mean"]),
            "LOLE_ci_low_h": float(lole_s["ci_low"]),
            "LOLE_ci_high_h": float(lole_s["ci_high"]),

            "EENS_mean_MWh": float(eens_s["mean"]),

            "ResolvedHour_mean": float(rh_s["mean"]),
            "ResolvedHour_ci_low": float(rh_s["ci_low"]),
            "ResolvedHour_ci_high": float(rh_s["ci_high"]),

            "ResolvedEnergy_mean": float(re_s["mean"]),

            "OpCost_mean_$": float(opcost_s["mean"]),
            "CAPEX_$": float(capex),
            "TotalCost_mean_$": float(totalcost_s["mean"]),

            "Profit_mean_$": float(profit_s["mean"]),
            "NetCostFromProfit_mean_$": float(netcost_from_profit_s["mean"]),

            "Tau2ResolvedAlarm_mean": float(tau2_res_s["mean"]),

            # Baseline context (NoESS)
            "NoESS_LOLE_mean_h": float(base_metrics["LOLE"]["mean"]),
            "NoESS_EENS_mean_MWh": float(base_metrics["EENS"]["mean"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("=== Final Sensitivity Table ===")
    print(df.to_string(index=False))

    if EXPORT_EXCEL:
        with pd.ExcelWriter(EXPORT_FILENAME, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Summary")
        print(f"\nSaved: {EXPORT_FILENAME}")


if __name__ == "__main__":
    main()
