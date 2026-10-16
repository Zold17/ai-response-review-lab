# AI Response Review Lab

A small Streamlit project that combines two skills used in AI training and data work:

- **Rating AI answers with a rubric** (practice mode with reference ratings and reasons).
- **Measuring how good and how consistent a team of reviewers is** (analytics on a review log).

**Live demo:** _add your Streamlit Cloud link here_

> All data in this project is sample data made for learning. It does not come from any company or client project.

## What it does

### 1. Review practice
- Shows 20 prompt and response pairs (in Indonesian) with deliberate flaws: wrong facts, invented content, ignored instructions, off-topic answers, bad tone, unsafe advice and agreeing with a wrong claim.
- You score each response on three criteria (1 to 5) and pick the main problem.
- After you submit, the app shows your scores next to the reference scores and explains the reference rating.
- Session summary: average gap from the reference, share of scores within 1 point, and CSV download of your reviews.

### 2. Quality analytics
Works on a log of 240 reviews (80 items, 3 reviewers):

- KPIs: items reviewed, average score, pass rate, average review time.
- **Reviewer agreement** on pass/fail with Cohen's kappa (calculated in `analytics.py`, no extra library).
- Pass rate by category, most common problems, weekly pass rate, reviewer strictness.
- **Calibration items:** the items reviewers disagree on most, listed first when they split on pass/fail.
- A "What the data says" section whose sentences are written from the numbers, not typed by hand.

## Rubric

| Criterion | Question | 1 | 5 |
|---|---|---|---|
| Accuracy | Is the content correct? | Wrong or invented | Fully correct |
| Helpfulness | Does it help with what was asked? | Useless | Very useful |
| Guideline adherence | Does it follow instructions, safety and tone rules? | Ignores them | Follows all |

**Pass rule:** a review passes only if all three scores are 3 or higher.

**Problem tags:** factual error, hallucination, instruction not followed, off topic, tone issue, safety issue, sycophancy, none.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (three tabs) |
| `analytics.py` | All scoring and analysis logic (Pandas and NumPy only) |
| `generate_data.py` | Creates `review_log.csv` with a fixed random seed |
| `samples.csv` | 20 practice items with reference ratings and reasons |
| `review_log.csv` | Synthetic review log |
| `requirements.txt` | Python dependencies |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To recreate the review log: `python generate_data.py`

## Deploy

Push these files to a public GitHub repository, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) with `app.py` as the main file.

## How the synthetic data is made

`generate_data.py` creates 80 items in 6 categories. Each item has a hidden problem (or none). Three reviewers score every item with their own strictness and random noise, so they sometimes disagree. The numbers in the dashboard therefore show how the analysis works, not real-world quality rates.

## Limitations

- The reference ratings are one person's judgement, not an official standard.
- 20 practice items and 80 log items are too few for real conclusions.
- Kappa on 80 items has a wide margin of error.

## Ideas for next steps

- Let users upload their own review log (CSV) and run the same analytics.
- Add a confidence interval for kappa.
- Add more practice items in English and a second rubric.
- Track how a reviewer's gap from the reference changes over several sessions.
