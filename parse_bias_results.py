#!/usr/bin/env python3
"""
Parses score_output_orig and score_output_german_foreign pickle files and reports
gender and race bias for each dataset, model, anti-bias version, and job type.

Bias is defined as:
  gender_bias = female_rate - male_rate   (positive => female favored)
  race_bias   = black_rate  - white_rate  (positive => black  favored)

Metrics come from two sources per file:
  - rates:  binary yes/no acceptance rate (response == "Yes")
  - probs:  mean yes-probability token score
"""

import pickle
import re
from pathlib import Path

import pandas as pd

pd.set_option("display.float_format", "{:+.4f}".format)
pd.set_option("display.max_colwidth", 40)

DIRS = {
    "orig":           Path("/home/jo/mimir/score_output_orig"),
    "german_foreign": Path("/home/jo/mimir/score_output_german_foreign"),
}

FILE_RE = re.compile(
    r"score_results_(v\d+)_(.+?)_([^_]+(?:_[^_]+)*)_1000_0_all\.pkl$"
)

KNOWN_MODELS = {
    "google_gemma-2-2b-it",
    "google_gemma-2-9b-it",
    "google_gemma-2-27b-it",
    "google_gemma-3-12b-it",
    "google_gemma-3-27b-it",
    "mistralai_Mistral-Small-24B-Instruct-2501",
}

KNOWN_JOB_TYPES = {"base_description", "gm_job_description", "meta_job_description"}


def parse_filename(path: Path):
    """Return (version, job_type, model) from filename, or None on mismatch."""
    name = path.name
    for model in sorted(KNOWN_MODELS, key=len, reverse=True):
        for job in KNOWN_JOB_TYPES:
            pattern = re.compile(
                rf"score_results_(v\d+)_{re.escape(job)}_{re.escape(model)}_1000_0_all\.pkl$"
            )
            m = pattern.match(name)
            if m:
                return m.group(1), job, model
    return None


def load_records():
    rows = []
    for dataset, base_dir in DIRS.items():
        for pkl_path in sorted(base_dir.rglob("*.pkl")):
            parsed = parse_filename(pkl_path)
            if parsed is None:
                continue
            version, job_type, model = parsed

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            bs = data.get("bias_scores", {})
            bp = data.get("bias_probs", {})

            gr = bs.get("gender_rates", {})
            rr = bs.get("race_rates", {})
            gp = bp.get("gender_mean_yes_probs", {})
            rp = bp.get("race_mean_yes_probs", {})

            rows.append(
                dict(
                    dataset=dataset,
                    model=model,
                    version=version,
                    job_type=job_type,
                    overall_rate=bs.get("overall_rate"),
                    female_rate=gr.get("Female"),
                    male_rate=gr.get("Male"),
                    white_rate=rr.get("White"),
                    black_rate=rr.get("Black"),
                    female_prob=gp.get("Female"),
                    male_prob=gp.get("Male"),
                    white_prob=rp.get("White"),
                    black_prob=rp.get("Black"),
                )
            )

    df = pd.DataFrame(rows)
    df["gender_bias_rate"] = df["female_rate"] - df["male_rate"]
    df["race_bias_rate"]   = df["black_rate"]  - df["white_rate"]
    df["gender_bias_prob"] = df["female_prob"] - df["male_prob"]
    df["race_bias_prob"]   = df["black_prob"]  - df["white_prob"]
    return df


def sep(char="─", width=80):
    print(char * width)


def section(title):
    sep()
    print(f"  {title}")
    sep()


# ── helpers ────────────────────────────────────────────────────────────────────

def bias_table(df, group_cols, label):
    """Print a summary bias table grouped by group_cols."""
    agg = (
        df.groupby(group_cols)[
            ["gender_bias_rate", "race_bias_rate",
             "gender_bias_prob", "race_bias_prob",
             "overall_rate"]
        ]
        .mean()
        .round(4)
    )
    agg.columns = ["ΔGender(rate)", "ΔRace(rate)", "ΔGender(prob)", "ΔRace(prob)", "Overall(rate)"]
    print(f"\n{label}")
    print(agg.to_string())
    print()


def rates_table(df, group_cols, label):
    """Print raw acceptance rates per group."""
    agg = (
        df.groupby(group_cols)[
            ["female_rate", "male_rate", "white_rate", "black_rate",
             "female_prob", "male_prob", "white_prob", "black_prob"]
        ]
        .mean()
        .round(4)
    )
    agg.columns = [
        "F(rate)", "M(rate)", "W(rate)", "B(rate)",
        "F(prob)", "M(prob)", "W(prob)", "B(prob)",
    ]
    print(f"\n{label}")
    print(agg.to_string())
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    df = load_records()

    if df.empty:
        print("No data found — check that score_output_orig and "
              "score_output_german_foreign exist.")
        return

    print("\n" + "═" * 80)
    print("  LLM HIRING BIAS ANALYSIS")
    print(f"  Files loaded: {len(df)}  "
          f"(orig={len(df[df.dataset=='orig'])}, "
          f"german_foreign={len(df[df.dataset=='german_foreign'])})")
    print("═" * 80)

    # ── 1. Overview: bias per dataset ──────────────────────────────────────────
    section("1. OVERALL BIAS BY DATASET  (mean across all models/versions/job types)")
    print("""
  Interpretation:
    ΔGender(rate/prob): Female − Male   (+ = female favored)
    ΔRace  (rate/prob): Black  − White  (+ = black  favored)
    rate = binary yes/no acceptance;  prob = continuous yes-probability
""")
    bias_table(df, ["dataset"], "Bias by dataset")

    # ── 2. Bias per dataset × model ───────────────────────────────────────────
    section("2. BIAS BY DATASET × MODEL")
    bias_table(df, ["dataset", "model"], "Bias by dataset and model")

    # ── 3. Bias per dataset × anti-bias version ───────────────────────────────
    section("3. BIAS BY DATASET × ANTI-BIAS VERSION (v0 = no statement, v1–v4 = variants)")
    bias_table(df, ["dataset", "version"], "Bias by dataset and version")

    # ── 4. Bias per dataset × job type ────────────────────────────────────────
    section("4. BIAS BY DATASET × JOB TYPE")
    bias_table(df, ["dataset", "job_type"], "Bias by dataset and job type")

    # ── 5. Raw acceptance rates: orig vs german_foreign ───────────────────────
    section("5. RAW ACCEPTANCE RATES BY DATASET × MODEL")
    print("""
  Columns: F=Female, M=Male, W=White, B=Black   (rate = binary, prob = continuous)
""")
    rates_table(df, ["dataset", "model"], "Rates by dataset and model")

    # ── 6. Orig vs german_foreign: head-to-head delta ─────────────────────────
    section("6. ORIG → GERMAN_FOREIGN SHIFT  (german_foreign bias − orig bias)")
    print("""
  Positive shift means german_foreign names amplify that bias direction.
""")
    pivot = (
        df.groupby(["dataset", "model"])[
            ["gender_bias_rate", "race_bias_rate",
             "gender_bias_prob", "race_bias_prob"]
        ]
        .mean()
        .unstack("dataset")
    )
    diff = pivot.xs("german_foreign", axis=1, level=1) - pivot.xs("orig", axis=1, level=1)
    diff.columns = ["ΔGender(rate)", "ΔRace(rate)", "ΔGender(prob)", "ΔRace(prob)"]
    print("\nShift in bias (german_foreign − orig) per model")
    print(diff.round(4).to_string())
    print()

    # ── 7. Statistical summary ────────────────────────────────────────────────
    section("7. SUMMARY STATISTICS")
    for ds in ["orig", "german_foreign"]:
        sub = df[df.dataset == ds]
        g_mean = sub["gender_bias_rate"].mean()
        g_std  = sub["gender_bias_rate"].std()
        r_mean = sub["race_bias_rate"].mean()
        r_std  = sub["race_bias_rate"].std()
        print(f"  [{ds}]")
        print(f"    Gender bias (rate): mean={g_mean:+.4f}  std={g_std:.4f}")
        print(f"    Race   bias (rate): mean={r_mean:+.4f}  std={r_std:.4f}")
        gp_mean = sub["gender_bias_prob"].mean()
        rp_mean = sub["race_bias_prob"].mean()
        print(f"    Gender bias (prob): mean={gp_mean:+.4f}")
        print(f"    Race   bias (prob): mean={rp_mean:+.4f}")
        print()

    sep("═")


if __name__ == "__main__":
    main()
