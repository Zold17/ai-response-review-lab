"""Scoring and quality analytics for the AI Response Review Lab.

Everything here is plain pandas and numpy, so it can be tested without Streamlit.
The Streamlit app (app.py) only calls these functions and draws the results.
"""
from itertools import combinations

import numpy as np
import pandas as pd

CRITERIA = ["accuracy", "helpfulness", "adherence"]

CRITERIA_LABELS = {
    "accuracy": "Accuracy",
    "helpfulness": "Helpfulness",
    "adherence": "Guideline adherence",
}

CRITERIA_HELP = {
    "accuracy": "Is the content correct? 1 = wrong or invented, 5 = fully correct.",
    "helpfulness": "Does it help with what the user asked? 1 = useless, 5 = very useful.",
    "adherence": "Does it follow the prompt's instructions and the safety and tone rules? "
    "1 = ignores them, 5 = follows all of them.",
}

ERROR_TAGS = [
    "none",
    "factual_error",
    "hallucination",
    "instruction_not_followed",
    "off_topic",
    "tone_issue",
    "safety_issue",
    "sycophancy",
]

TAG_LABELS = {
    "none": "No problem",
    "factual_error": "Factual error",
    "hallucination": "Hallucination (invented content)",
    "instruction_not_followed": "Instruction not followed",
    "off_topic": "Off topic",
    "tone_issue": "Tone issue",
    "safety_issue": "Safety issue",
    "sycophancy": "Sycophancy (agrees with a wrong claim)",
}

# A review passes only when every criterion is at least this score.
PASS_MIN = 3


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def add_scores(df):
    """Add the overall score (mean of the criteria) and the pass/fail flag."""
    out = df.copy()
    out["overall"] = out[CRITERIA].mean(axis=1)
    out["passed"] = (out[CRITERIA] >= PASS_MIN).all(axis=1)
    return out


# --------------------------------------------------------------------------
# Practice mode: compare a person's review with the reference rating
# --------------------------------------------------------------------------
def build_record(sample_id, sample, scores, tag):
    """Combine one submitted review with the reference rating of the sample."""
    record = {
        "id": sample_id,
        "category": sample["category"],
        "error_tag": tag,
        "ref_error_tag": sample["ref_error_tag"],
    }
    for c in CRITERIA:
        record[c] = int(scores[c])
        record[f"ref_{c}"] = int(sample[f"ref_{c}"])
    return record


def compare_table(record):
    """Per-criterion table: your score, the reference score and the gap."""
    rows = []
    for c in CRITERIA:
        rows.append(
            {
                "Criterion": CRITERIA_LABELS[c],
                "You": record[c],
                "Reference": record[f"ref_{c}"],
                "Gap": record[c] - record[f"ref_{c}"],
            }
        )
    return pd.DataFrame(rows)


def record_check(record):
    """Summarise one review against its reference."""
    gaps = [abs(record[c] - record[f"ref_{c}"]) for c in CRITERIA]
    return {
        "all_within_one": all(g <= 1 for g in gaps),
        "tag_match": record["error_tag"] == record["ref_error_tag"],
        "mean_abs_gap": float(np.mean(gaps)),
    }


def session_summary(records):
    """Summary over every item reviewed in a practice session."""
    if not records:
        return {"reviewed": 0, "mean_abs_gap": 0.0, "within_one_pct": 0.0, "tag_match_pct": 0.0}
    df = pd.DataFrame(records)
    gaps = np.column_stack([(df[c] - df[f"ref_{c}"]).abs().to_numpy() for c in CRITERIA])
    return {
        "reviewed": int(len(df)),
        "mean_abs_gap": float(gaps.mean()),
        "within_one_pct": float((gaps <= 1).mean() * 100),
        "tag_match_pct": float((df["error_tag"] == df["ref_error_tag"]).mean() * 100),
    }


# --------------------------------------------------------------------------
# Analytics on the review log
# --------------------------------------------------------------------------
def kpis(df):
    """Headline numbers for a (possibly filtered) review log."""
    if df.empty:
        return {"items": 0, "reviews": 0, "avg_overall": float("nan"),
                "pass_rate": float("nan"), "avg_seconds": float("nan")}
    return {
        "items": int(df["item_id"].nunique()),
        "reviews": int(len(df)),
        "avg_overall": float(df["overall"].mean()),
        "pass_rate": float(df["passed"].mean() * 100),
        "avg_seconds": float(df["review_seconds"].mean()),
    }


def by_category(df):
    """Pass rate and average score per category, weakest category first."""
    g = df.groupby("category")
    out = pd.DataFrame(
        {
            "items": g["item_id"].nunique(),
            "avg_overall": g["overall"].mean(),
            "pass_rate": g["passed"].mean() * 100,
        }
    ).reset_index()
    return out.sort_values("pass_rate").reset_index(drop=True)


def error_distribution(df):
    """How often each problem type was flagged (reviews with no problem excluded)."""
    flagged = df[df["error_tag"] != "none"]
    out = flagged.groupby("error_tag").size().rename("count").reset_index()
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def reviewer_summary(df):
    """Per reviewer: volume, average score, pass rate, speed, and strictness vs the team."""
    team_mean = df["overall"].mean()
    g = df.groupby("reviewer")
    out = pd.DataFrame(
        {
            "reviews": g.size(),
            "avg_overall": g["overall"].mean(),
            "pass_rate": g["passed"].mean() * 100,
            "avg_seconds": g["review_seconds"].mean(),
        }
    ).reset_index()
    out["vs_team"] = out["avg_overall"] - team_mean
    return out


def weekly_trend(df):
    """Average score and pass rate per week (weeks start on Monday)."""
    d = df.copy()
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    g = d.groupby("week")
    out = pd.DataFrame(
        {"avg_overall": g["overall"].mean(), "pass_rate": g["passed"].mean() * 100}
    ).reset_index()
    return out


def cohen_kappa(a, b):
    """Cohen's kappa: agreement between two raters, corrected for chance."""
    a = np.asarray(a)
    b = np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    observed = float(np.mean(a == b))
    expected = float(sum(np.mean(a == lab) * np.mean(b == lab) for lab in labels))
    if expected == 1:
        return 1.0 if observed == 1 else float("nan")
    return (observed - expected) / (1 - expected)


def kappa_label(k):
    """Plain-language reading of kappa (Landis and Koch bands)."""
    if k != k:  # NaN
        return "not defined"
    if k < 0.2:
        return "slight agreement"
    if k < 0.4:
        return "fair agreement"
    if k < 0.6:
        return "moderate agreement"
    if k < 0.8:
        return "substantial agreement"
    return "almost perfect agreement"


def pairwise_agreement(df):
    """Pass/fail agreement for every pair of reviewers who rated the same items."""
    wide = df.assign(passed_int=df["passed"].astype(int)).pivot_table(
        index="item_id", columns="reviewer", values="passed_int", aggfunc="first"
    )
    rows = []
    for r1, r2 in combinations(sorted(wide.columns), 2):
        pair = wide[[r1, r2]].dropna()
        if pair.empty:
            continue
        a = pair[r1].astype(int).to_numpy()
        b = pair[r2].astype(int).to_numpy()
        rows.append(
            {
                "pair": f"{r1} vs {r2}",
                "items": int(len(pair)),
                "agreement": float(np.mean(a == b) * 100),
                "kappa": float(cohen_kappa(a, b)),
            }
        )
    return pd.DataFrame(rows, columns=["pair", "items", "agreement", "kappa"])


def calibration_items(df, top_n=10):
    """Items the reviewers disagree on most. Split pass/fail decisions come first."""
    g = df.groupby(["item_id", "category"])
    out = pd.DataFrame(
        {
            "min_score": g["overall"].min(),
            "max_score": g["overall"].max(),
            "pass_votes": g["passed"].sum(),
            "reviewers": g["reviewer"].nunique(),
        }
    ).reset_index()
    out["spread"] = out["max_score"] - out["min_score"]
    out["split_decision"] = (out["pass_votes"] > 0) & (out["pass_votes"] < out["reviewers"])
    out = out.sort_values(["split_decision", "spread"], ascending=[False, False])
    return out.head(top_n).reset_index(drop=True)


def findings(df):
    """Short plain-language findings, computed from the data (nothing hard-coded)."""
    notes = []
    if df.empty:
        return notes

    cat = by_category(df)
    if len(cat) >= 2:
        low, high = cat.iloc[0], cat.iloc[-1]
        notes.append(
            f"Weakest category: **{low['category']}** ({low['pass_rate']:.0f}% of reviews pass). "
            f"Strongest: **{high['category']}** ({high['pass_rate']:.0f}%)."
        )

    err = error_distribution(df)
    if not err.empty:
        top = err.iloc[0]
        share = top["count"] / len(df) * 100
        notes.append(
            f"The most common problem is **{TAG_LABELS[top['error_tag']]}**, "
            f"flagged in {share:.0f}% of reviews."
        )

    rv = reviewer_summary(df).sort_values("avg_overall")
    if len(rv) >= 2:
        strict, lenient = rv.iloc[0], rv.iloc[-1]
        notes.append(
            f"{strict['reviewer']} scores the strictest ({strict['vs_team']:+.2f} vs the team average) "
            f"and {lenient['reviewer']} the most lenient ({lenient['vs_team']:+.2f}). "
            "A gap like this is a reason to recalibrate with shared examples."
        )

    pw = pairwise_agreement(df)
    if not pw.empty:
        k = float(pw["kappa"].mean())
        notes.append(
            f"Average pass/fail agreement between reviewers is {pw['agreement'].mean():.0f}% "
            f"(Cohen's kappa {k:.2f}, {kappa_label(k)})."
        )
        n_split = int(calibration_items(df, top_n=len(df))["split_decision"].sum())
        notes.append(
            f"{n_split} items received a split pass/fail decision. "
            "They are the best candidates to discuss when recalibrating."
        )
    return notes
