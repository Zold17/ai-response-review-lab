"""Generate the sample review log (review_log.csv).

The data is synthetic. It is generated with a fixed random seed so the file can be
recreated exactly:  python generate_data.py

How it works
- 80 items across 6 categories. Each item has a hidden "true" problem (or none).
- Three reviewers score every item on three criteria (1-5). Each reviewer has a
  built-in bias (strict or lenient) and some random noise, so they disagree
  sometimes, like real reviewers do.
- Reviewers notice the true problem most of the time, miss it sometimes, and
  occasionally flag a problem that is not there.
"""
import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2026-09-14")  # a Monday
DAYS = 28

# Realism knobs. Higher reviewer noise and milder problems create borderline items,
# so reviewers disagree on some of them, as real review teams do.
ITEM_SD = 0.45         # how much true quality varies from item to item
REVIEWER_SD = 0.70     # random noise in each reviewer's score
MIN_SEVERITY = 0.55    # a problem lowers scores by 55% to 100% of its full effect

# reviewer -> (score bias, average seconds per review)
REVIEWERS = {
    "Reviewer A": (0.00, 70),
    "Reviewer B": (0.20, 55),
    "Reviewer C": (-0.45, 85),
}

# category -> number of items, base quality, chance of each hidden problem
CATEGORIES = {
    "Factual Q&A": dict(n=18, base=4.3, errors={"factual_error": 0.18, "hallucination": 0.10}),
    "Math": dict(n=12, base=4.2, errors={"factual_error": 0.22}),
    "Instruction following": dict(n=16, base=4.2, errors={"instruction_not_followed": 0.30, "off_topic": 0.06}),
    "Writing": dict(n=14, base=4.4, errors={"tone_issue": 0.15, "instruction_not_followed": 0.10}),
    "Safety": dict(n=12, base=4.1, errors={"safety_issue": 0.25, "sycophancy": 0.10}),
    "Translation": dict(n=8, base=4.5, errors={"factual_error": 0.12, "instruction_not_followed": 0.08}),
}

# how much each hidden problem lowers each criterion
EFFECT = {
    "factual_error": {"accuracy": -2.6, "helpfulness": -1.2},
    "hallucination": {"accuracy": -3.0, "helpfulness": -1.5},
    "instruction_not_followed": {"adherence": -2.6, "helpfulness": -0.8},
    "off_topic": {"helpfulness": -3.0, "adherence": -2.2},
    "tone_issue": {"helpfulness": -1.6, "adherence": -1.0},
    "safety_issue": {"adherence": -3.0, "accuracy": -0.8},
    "sycophancy": {"accuracy": -2.6, "helpfulness": -1.4},
}

CRITERIA = ["accuracy", "helpfulness", "adherence"]
ALL_TAGS = list(EFFECT)


def draw_problem(rng, probabilities):
    """Pick the hidden problem for an item (or 'none')."""
    u = rng.random()
    total = 0.0
    for tag, p in probabilities.items():
        total += p
        if u < total:
            return tag
    return "none"


def build_log(seed=SEED):
    rng = np.random.default_rng(seed)

    # 1) items
    items = []
    for category, spec in CATEGORIES.items():
        for _ in range(spec["n"]):
            tag = draw_problem(rng, spec["errors"])
            latent = {c: spec["base"] + rng.normal(0, ITEM_SD) for c in CRITERIA}
            for c, delta in EFFECT.get(tag, {}).items():
                latent[c] += delta * rng.uniform(MIN_SEVERITY, 1.0)
            items.append(
                {
                    "category": category,
                    "true_tag": tag,
                    "latent": latent,
                    "date": START + pd.Timedelta(days=int(rng.integers(0, DAYS))),
                }
            )
    items.sort(key=lambda it: it["date"])

    # 2) reviews
    rows = []
    for number, item in enumerate(items, start=1):
        item_id = f"Q{number:03d}"
        for reviewer, (bias, mean_seconds) in REVIEWERS.items():
            scores = {}
            for c in CRITERIA:
                raw = item["latent"][c] + bias + rng.normal(0, REVIEWER_SD)
                scores[c] = int(np.clip(np.rint(raw), 1, 5))

            tag = item["true_tag"]
            if tag != "none":
                if rng.random() > 0.85:  # the reviewer misses the problem
                    tag = "none"
            elif rng.random() < 0.05:  # the reviewer flags a problem that is not there
                tag = str(rng.choice(ALL_TAGS))

            seconds = rng.lognormal(mean=np.log(mean_seconds), sigma=0.35)
            if item["true_tag"] != "none":
                seconds += 25  # problems take longer to document
            rows.append(
                {
                    "item_id": item_id,
                    "date": (item["date"] + pd.Timedelta(days=int(rng.integers(0, 3)))).date().isoformat(),
                    "category": item["category"],
                    "reviewer": reviewer,
                    **scores,
                    "error_tag": tag,
                    "review_seconds": int(max(15, round(seconds))),
                }
            )

    log = pd.DataFrame(rows)
    return log.sort_values(["date", "item_id", "reviewer"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build_log()
    df.to_csv("review_log.csv", index=False)
    print(f"Wrote review_log.csv: {len(df)} reviews, {df['item_id'].nunique()} items")
