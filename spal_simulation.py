#!/usr/bin/env python3
"""
SPAL (Surface Pace and Athlete-Load) Monte Carlo Simulation
Based on the methodology described in the SPAL protocol paper.

Implements:
- Surface parameter distributions
- CPR_16, S_30, L, H, PCI calculations
- Ball liveness confounding
- Ranking reversal analysis
- Sensitivity analyses (pace & load modules)
- Summary statistics matching the paper's tables

Requirements: numpy, pandas
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Fixed seed for exact reproducibility
RNG = np.random.default_rng(2025)

# Surface class definitions (mean ± SD)
SURFACE_PARAMS = {
    "Hard_Fast": {
        "mu": (0.42, 0.04),
        "e": (0.85, 0.02),
        "bite": (0.10, 0.05),
        "SA": (0.9, 0.4),
        "ER": (91.0, 1.2),
        "SR": (65.5, 1.0),
    },
    "Hard_Medium": {
        "mu": (0.50, 0.04),
        "e": (0.83, 0.02),
        "bite": (0.22, 0.06),
        "SA": (1.1, 0.5),
        "ER": (89.5, 1.4),
        "SR": (66.6, 1.1),
    },
    "Hard_Slowed": {
        "mu": (0.65, 0.04),
        "e": (0.80, 0.02),
        "bite": (0.60, 0.08),
        "SA": (1.0, 0.4),
        "ER": (90.2, 1.3),
        "SR": (67.2, 1.0),
    },
    "Clay": {
        "mu": (0.78, 0.03),
        "e": (0.74, 0.02),
        "bite": (0.90, 0.05),
        "SA": (11.5, 2.0),
        "ER": (76.0, 2.5),
        "SR": (54.5, 1.5),
    },
    "Artificial_Grass": {
        "mu": (0.55, 0.04),
        "e": (0.79, 0.02),
        "bite": (0.35, 0.07),
        "SA": (8.5, 1.8),
        "ER": (81.0, 2.0),
        "SR": (61.2, 1.4),
    },
}

SURFACE_ORDER = list(SURFACE_PARAMS.keys())


def clip(x, low, high):
    return np.clip(x, low, high)


def sample_surface(surface: str, n: int) -> pd.DataFrame:
    """Sample n realizations of a given surface class."""
    p = SURFACE_PARAMS[surface]
    data = {
        "surface": surface,
        "mu": clip(RNG.normal(*p["mu"], n), 0.30, 0.90),
        "e": clip(RNG.normal(*p["e"], n), 0.65, 0.92),
        "bite": clip(RNG.normal(*p["bite"], n), 0.0, 1.0),
        "SA": clip(RNG.normal(*p["SA"], n), 0.1, 25.0),
        "ER": clip(RNG.normal(*p["ER"], n), 60.0, 100.0),
        "SR": clip(RNG.normal(*p["SR"], n), 40.0, 80.0),
    }
    return pd.DataFrame(data)


def compute_metrics(
    df: pd.DataFrame,
    ball_mod_strength: float = 0.22,
    bite_coeff: float = 0.14,
    humidity_weight: float = 1.4,
    sa_weight: float = 7.0,
) -> pd.DataFrame:
    """Compute all pace, load, and composite metrics."""
    df = df.copy()

    # CPR_16 (ITF formula)
    df["CPR_16"] = 100 * (1 - df["mu"]) + 150 * (0.81 - df["e"])

    # S_30 (rally-aligned horizontal speed ratio)
    tan30 = np.tan(np.deg2rad(30))
    raw_s30 = 1 - df["mu"] * (1 + df["e"]) * tan30 - bite_coeff * df["bite"] * df["mu"]
    df["S_30"] = clip(raw_s30, 0.36, 0.72)

    # Ball liveness B ∈ {48, 72, 90} with weights 0.25 / 0.50 / 0.25
    ball_choices = [48, 72, 90]
    ball_probs = [0.25, 0.50, 0.25]
    df["B"] = RNG.choice(ball_choices, size=len(df), p=ball_probs)

    # Observed (confounded) CPR
    df["CPR_obs"] = df["CPR_16"] * (0.78 + ball_mod_strength * df["B"] / 72)
    df["confounded"] = np.abs(df["CPR_obs"] - df["CPR_16"]) > 2.5

    # Load-friendliness score L
    sa_term = np.minimum(100, sa_weight * df["SA"])
    er_term = np.maximum(0, 100 - 1.8 * np.abs(df["ER"] - 78))
    sr_term = np.maximum(0, 100 - 2.2 * np.abs(df["SR"] - 60))
    df["L"] = clip((sa_term + er_term + sr_term) / 3, 0, 100)

    # Humidity penalty H
    humidity = RNG.uniform(45, 90, size=len(df))
    df["H"] = np.maximum(0, (humidity - 62) * humidity_weight)

    # PCI (equal weights)
    df["PCI"] = (
        0.2 * df["CPR_16"]
        + 0.2 * (100 * df["S_30"])
        + 0.2 * df["B"]
        + 0.2 * df["L"]
        + 0.2 * (100 - df["H"])
    )

    return df


def ranking_reversal_rate(n_iterations: int = 200) -> float:
    """Proportion of random multi-court subsets where CPR_16 and S_30 rankings diverge."""
    reversals = 0
    for _ in range(n_iterations):
        # One random court per surface class
        samples = []
        for surf in SURFACE_ORDER:
            s = sample_surface(surf, 1)
            s = compute_metrics(s)
            samples.append(s.iloc[0])
        subset = pd.DataFrame(samples)

        rank_cpr = subset["CPR_16"].rank(ascending=False)
        rank_s30 = subset["S_30"].rank(ascending=False)
        if not rank_cpr.equals(rank_s30):
            reversals += 1
    return reversals / n_iterations


def main_simulation(n_per_surface: int = 400) -> pd.DataFrame:
    """Run the main Monte Carlo simulation (N=2000)."""
    frames = []
    for surf in SURFACE_ORDER:
        df = sample_surface(surf, n_per_surface)
        df = compute_metrics(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    """Produce Table 1 style summary."""
    summary = (
        df.groupby("surface")
        .agg(
            CPR_16=("CPR_16", "mean"),
            CPR_16_std=("CPR_16", "std"),
            S_30=("S_30", "mean"),
            SA=("SA", "mean"),
            PCI=("PCI", "mean"),
            Confound_Rate=("confounded", "mean"),
        )
        .round(2)
    )
    summary["Confound_Rate"] = (summary["Confound_Rate"] * 100).round(0).astype(int)
    return summary.reindex(SURFACE_ORDER)


def ball_liveness_effect(surface: str = "Hard_Slowed", n: int = 400) -> pd.DataFrame:
    """Table 2: Ball liveness conditioning on a fixed surface class."""
    results = []
    for b_val, label in [(90, "Lively"), (72, "Normal"), (48, "Dead")]:
        df = sample_surface(surface, n)
        # Force ball score
        df = compute_metrics(df)
        df["B"] = b_val
        # Recompute dependent quantities
        df["CPR_obs"] = df["CPR_16"] * (0.78 + 0.22 * df["B"] / 72)
        df["PCI"] = (
            0.2 * df["CPR_16"]
            + 0.2 * (100 * df["S_30"])
            + 0.2 * df["B"]
            + 0.2 * df["L"]
            + 0.2 * (100 - df["H"])
        )
        results.append(
            {
                "Ball Vigor": label,
                "Ball Score (B)": b_val,
                "Mean Observed CPR_16": df["CPR_obs"].mean(),
                "Mean S_30": df["S_30"].mean(),
                "Mean PCI": df["PCI"].mean(),
            }
        )
    return pd.DataFrame(results).round(2)


def sensitivity_pace():
    """Sensitivity analyses for pace module (Tables 6-8)."""
    ball_strengths = [0.10, 0.16, 0.22, 0.28, 0.34]
    bite_coeffs = [0.06, 0.10, 0.14, 0.18, 0.22]
    humidity_weights = [0.6, 1.0, 1.4, 1.8, 2.2]

    def run_sensitivity(param_name, values, **kwargs):
        rows = []
        for val in values:
            kw = {param_name: val, **kwargs}
            # Smaller sample for sensitivity
            frames = []
            for surf in SURFACE_ORDER:
                df = sample_surface(surf, 150)
                df = compute_metrics(df, **kw)
                frames.append(df)
            full = pd.concat(frames)
            rev = ranking_reversal_rate_custom(full)  # approximate via full data ranking
            # Better: use dedicated ranking function with modified params
            conf = full["confounded"].mean()
            rows.append({param_name: val, "Reversal Rate": np.nan, "Confound Rate": conf})
        return pd.DataFrame(rows)

    # Proper ranking reversal under different parameters
    def ranking_reversal_with_params(n_iter=200, **compute_kwargs) -> float:
        reversals = 0
        for _ in range(n_iter):
            samples = []
            for surf in SURFACE_ORDER:
                s = sample_surface(surf, 1)
                s = compute_metrics(s, **compute_kwargs)
                samples.append(s.iloc[0])
            subset = pd.DataFrame(samples)
            if not subset["CPR_16"].rank(ascending=False).equals(subset["S_30"].rank(ascending=False)):
                reversals += 1
        return reversals / n_iter

    print("=== Sensitivity: Ball Modulation Strength ===")
    ball_rows = []
    for bs in ball_strengths:
        rev = ranking_reversal_with_params(ball_mod_strength=bs)
        # Confound rate
        frames = [compute_metrics(sample_surface(s, 150), ball_mod_strength=bs) for s in SURFACE_ORDER]
        conf = pd.concat(frames)["confounded"].mean()
        ball_rows.append({"Ball Strength": bs, "Reversal Rate": round(rev, 2), "Confound Rate": round(conf, 2)})
    print(pd.DataFrame(ball_rows).to_string(index=False))

    print("\n=== Sensitivity: Bite Coefficient ===")
    bite_rows = []
    for bc in bite_coeffs:
        rev = ranking_reversal_with_params(bite_coeff=bc)
        frames = [compute_metrics(sample_surface(s, 150), bite_coeff=bc) for s in SURFACE_ORDER]
        conf = pd.concat(frames)["confounded"].mean()
        bite_rows.append({"Bite Coefficient": bc, "Reversal Rate": round(rev, 2), "Confound Rate": round(conf, 2)})
    print(pd.DataFrame(bite_rows).to_string(index=False))

    print("\n=== Sensitivity: Humidity Weight ===")
    hum_rows = []
    for hw in humidity_weights:
        rev = ranking_reversal_with_params(humidity_weight=hw)
        frames = [compute_metrics(sample_surface(s, 150), humidity_weight=hw) for s in SURFACE_ORDER]
        conf = pd.concat(frames)["confounded"].mean()
        hum_rows.append({"Humidity Weight": hw, "Reversal Rate": round(rev, 2), "Confound Rate": round(conf, 2)})
    print(pd.DataFrame(hum_rows).to_string(index=False))


def sensitivity_load():
    """Sensitivity analyses for load module (Tables 9-11)."""
    print("\n=== Sensitivity: SA Perturbation (±1%) ===")
    base_surfaces = {
        "Hard": "Hard_Medium",
        "Clay": "Clay",
        "Grass": "Artificial_Grass",
    }
    sa_rows = []
    for name, surf in base_surfaces.items():
        df = sample_surface(surf, 400)
        df = compute_metrics(df)
        base_pci = df["PCI"].mean()

        for delta in [-0.01, 0.01]:
            df_pert = df.copy()
            df_pert["SA"] = df["SA"] * (1 + delta)
            # Recompute L and PCI
            sa_term = np.minimum(100, 7 * df_pert["SA"])
            er_term = np.maximum(0, 100 - 1.8 * np.abs(df_pert["ER"] - 78))
            sr_term = np.maximum(0, 100 - 2.2 * np.abs(df_pert["SR"] - 60))
            df_pert["L"] = clip((sa_term + er_term + sr_term) / 3, 0, 100)
            df_pert["PCI"] = (
                0.2 * df_pert["CPR_16"]
                + 0.2 * (100 * df_pert["S_30"])
                + 0.2 * df_pert["B"]
                + 0.2 * df_pert["L"]
                + 0.2 * (100 - df_pert["H"])
            )
            sa_rows.append({
                "Surface": name,
                "Perturbation": f"{delta*100:+.0f}%",
                "Mean PCI": round(df_pert["PCI"].mean(), 2),
                "ΔPCI": round(df_pert["PCI"].mean() - base_pci, 2),
            })
    print(pd.DataFrame(sa_rows).to_string(index=False))

    print("\n=== SA Weighting Coefficient Effect ===")
    weight_rows = []
    for w in [4, 7, 10]:
        row = {"SA Weight": w}
        for name, surf in base_surfaces.items():
            df = sample_surface(surf, 400)
            df = compute_metrics(df, sa_weight=w)
            row[name] = round(df["L"].mean(), 1)
        row["Hard-Clay Diff"] = round(row["Clay"] - row["Hard"], 1)
        weight_rows.append(row)
    print(pd.DataFrame(weight_rows).to_string(index=False))


def main():
    print("=" * 70)
    print("SPAL Monte Carlo Simulation (seed=2025)")
    print("=" * 70)

    # Main simulation
    print("\n[1] Running main Monte Carlo (N=400 per surface)...")
    df_main = main_simulation(400)
    summary = summarize_main(df_main)
    print("\n--- Table 1: Monte Carlo Summary ---")
    print(summary.to_string())

    overall_confound = df_main["confounded"].mean()
    print(f"\nOverall Confound Rate: {overall_confound:.1%}")

    # Ranking reversal
    print("\n[2] Computing ranking reversal rate (200 iterations)...")
    rev_rate = ranking_reversal_rate(200)
    print(f"CPR_16 vs S_30 Rank Reversal Rate: {rev_rate:.1%}")

    # Ball liveness effect
    print("\n[3] Ball liveness effect on Hard_Slowed...")
    ball_table = ball_liveness_effect()
    print("\n--- Table 2: Ball Liveness Conditioning ---")
    print(ball_table.to_string(index=False))

    # Sensitivity analyses
    print("\n[4] Sensitivity Analyses...")
    sensitivity_pace()
    sensitivity_load()

    # Save full results
    output_path = "/workspace/artifacts/spal_simulation_results.csv"
    df_main.to_csv(output_path, index=False)
    print(f"\nFull simulation data saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Simulation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
