"""AI Response Review Lab - a Streamlit app.

Tab 1  Review practice: rate AI answers with a rubric and compare with a reference rating.
Tab 2  Quality analytics: how consistent and how good are the reviews in a review log?
Tab 3  About: rubric, pass rule and notes on the data.

Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import analytics as an

BASE = Path(__file__).parent

st.set_page_config(page_title="AI Response Review Lab", page_icon="🧪", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_samples():
    return pd.read_csv(BASE / "samples.csv")


@st.cache_data
def load_log():
    df = pd.read_csv(BASE / "review_log.csv", parse_dates=["date"])
    return an.add_scores(df)


samples = load_samples()
sample_by_id = samples.set_index("id")
N = len(samples)

# ------------------------------------------------------------------ state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "results" not in st.session_state:
    st.session_state.results = {}

WIDGET_PREFIXES = tuple(f"{c}_" for c in an.CRITERIA) + ("tag_",)


def submit(sample_id):
    """Save the current review together with the reference rating."""
    scores = {c: st.session_state[f"{c}_{sample_id}"] for c in an.CRITERIA}
    tag = st.session_state[f"tag_{sample_id}"]
    st.session_state.results[sample_id] = an.build_record(
        sample_id, sample_by_id.loc[sample_id], scores, tag
    )


def go(step):
    st.session_state.idx = max(0, min(N - 1, st.session_state.idx + step))


def restart():
    st.session_state.idx = 0
    st.session_state.results = {}
    for key in list(st.session_state.keys()):
        if key.startswith(WIDGET_PREFIXES):
            del st.session_state[key]


# ------------------------------------------------------------------ header
st.title("AI Response Review Lab")
st.caption(
    "Practice rating AI answers with a rubric, then see how consistent a team of reviewers is. "
    "All data in this app is sample data made for learning."
)

tab_review, tab_analytics, tab_about = st.tabs(["Review practice", "Quality analytics", "About"])

# ------------------------------------------------------------------ tab 1
with tab_review:
    results = st.session_state.results
    idx = st.session_state.idx
    row = samples.iloc[idx]
    sample_id = row["id"]
    done = sample_id in results

    st.progress(len(results) / N, text=f"{len(results)} of {N} items reviewed")

    left, right = st.columns([3, 2])

    with left:
        st.caption(f"Item {idx + 1} of {N}  ·  {row['category']}")
        with st.container(border=True):
            st.markdown("**Prompt**")
            st.write(row["prompt"])
            st.markdown("**AI response**")
            st.write(row["response"])
        st.caption("Prompts and responses are in Indonesian. The reference notes are in English.")

    with right:
        if not done:
            st.markdown("**Your review**")
            for c in an.CRITERIA:
                st.slider(
                    an.CRITERIA_LABELS[c], 1, 5, 3,
                    key=f"{c}_{sample_id}", help=an.CRITERIA_HELP[c],
                )
            st.selectbox(
                "Main problem (if any)", an.ERROR_TAGS,
                key=f"tag_{sample_id}", format_func=lambda t: an.TAG_LABELS[t],
            )
            st.button("Submit review", type="primary", on_click=submit, args=(sample_id,))
        else:
            rec = results[sample_id]
            check = an.record_check(rec)
            st.markdown("**Your review vs the reference**")
            st.dataframe(an.compare_table(rec), hide_index=True)
            st.write(
                f"Your main problem: **{an.TAG_LABELS[rec['error_tag']]}**  \n"
                f"Reference: **{an.TAG_LABELS[rec['ref_error_tag']]}**"
            )
            if check["all_within_one"] and check["tag_match"]:
                st.success("Within 1 point on every criterion, and the same main problem.")
            elif check["all_within_one"]:
                st.info("Scores are close, but the main problem differs from the reference.")
            else:
                st.warning("At least one score is more than 1 point from the reference.")
            st.markdown("**Why the reference scored it this way**")
            st.write(row["rationale"])

    nav_prev, nav_next, _ = st.columns([1, 1, 4])
    nav_prev.button("← Previous", on_click=go, args=(-1,), disabled=idx == 0)
    nav_next.button("Next →", on_click=go, args=(1,), disabled=idx == N - 1)

    if results:
        st.divider()
        st.subheader("Your session")
        summary = an.session_summary(list(results.values()))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items reviewed", summary["reviewed"])
        m2.metric("Average gap from reference", f"{summary['mean_abs_gap']:.2f} pts")
        m3.metric("Scores within 1 point", f"{summary['within_one_pct']:.0f}%")
        m4.metric("Same main problem", f"{summary['tag_match_pct']:.0f}%")
        csv = pd.DataFrame(list(results.values())).to_csv(index=False)
        b1, b2, _ = st.columns([1, 1, 4])
        b1.download_button("Download my reviews (CSV)", data=csv, file_name="my_reviews.csv", mime="text/csv")
        b2.button("Start over", on_click=restart)

# ------------------------------------------------------------------ tab 2
with tab_analytics:
    log = load_log()
    all_cats = sorted(log["category"].unique())
    all_revs = sorted(log["reviewer"].unique())

    f1, f2 = st.columns(2)
    cats = f1.multiselect("Categories", all_cats, default=all_cats)
    revs = f2.multiselect("Reviewers", all_revs, default=all_revs)
    data = log[log["category"].isin(cats) & log["reviewer"].isin(revs)]

    if data.empty:
        st.info("Select at least one category and one reviewer.")
    else:
        k = an.kpis(data)
        pw = an.pairwise_agreement(data)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Items reviewed", k["items"])
        c2.metric("Average score", f"{k['avg_overall']:.2f} / 5")
        c3.metric("Pass rate", f"{k['pass_rate']:.0f}%")
        c4.metric("Avg review time", f"{k['avg_seconds']:.0f} s")
        c5.metric("Reviewer agreement (kappa)", f"{pw['kappa'].mean():.2f}" if not pw.empty else "n/a")

        st.subheader("What the data says")
        for line in an.findings(data):
            st.markdown(f"- {line}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Pass rate by category (%)")
            cat_df = an.by_category(data)
            st.bar_chart(cat_df.set_index("category")["pass_rate"])
        with col_b:
            st.subheader("Most common problems (reviews flagged)")
            err_df = an.error_distribution(data)
            if err_df.empty:
                st.info("No problems were flagged in this selection.")
            else:
                err_df["error_tag"] = err_df["error_tag"].map(an.TAG_LABELS)
                st.bar_chart(err_df.set_index("error_tag")["count"])

        st.subheader("Pass rate per week (%)")
        trend = an.weekly_trend(data)
        st.line_chart(trend.set_index("week")["pass_rate"])

        st.subheader("Reviewers")
        rv = an.reviewer_summary(data).round(2)
        rv.columns = ["Reviewer", "Reviews", "Avg score", "Pass rate (%)", "Avg seconds", "vs team"]
        st.dataframe(rv, hide_index=True)

        st.subheader("Agreement between reviewers")
        if pw.empty:
            st.info("Select at least two reviewers to see agreement.")
        else:
            pw_view = pw.round(2)
            pw_view.columns = ["Pair", "Shared items", "Agreement (%)", "Cohen's kappa"]
            st.dataframe(pw_view, hide_index=True)
            st.caption(
                "Cohen's kappa corrects the raw agreement for chance. "
                "Around 0.6 and above is usually read as substantial agreement."
            )

            st.subheader("Calibration items")
            cal = an.calibration_items(data, top_n=10).round(2)
            cal = cal[["item_id", "category", "min_score", "max_score", "spread", "split_decision"]]
            cal.columns = ["Item", "Category", "Lowest score", "Highest score", "Spread", "Split pass/fail"]
            st.dataframe(cal, hide_index=True)
            st.caption("Items where reviewers disagree most. Discuss these first when recalibrating.")

        with st.expander("Raw review log"):
            st.dataframe(data.drop(columns=["passed"]).assign(overall=data["overall"].round(2)), hide_index=True)

# ------------------------------------------------------------------ tab 3
with tab_about:
    st.markdown(
        """
### What this project shows
- **AI trainer skills:** rubric-based scoring, a problem taxonomy, and writing the reason behind a rating.
- **Data analyst skills:** cleaning and aggregating a review log with Pandas, KPIs, trends and an
  agreement metric (Cohen's kappa), with the findings written from the data.

### Rubric
Each response gets three scores from 1 to 5.

| Criterion | Question | 1 | 5 |
|---|---|---|---|
| Accuracy | Is the content correct? | Wrong or invented | Fully correct |
| Helpfulness | Does it help with what was asked? | Useless | Very useful |
| Guideline adherence | Does it follow the instructions, safety and tone rules? | Ignores them | Follows all |

**Pass rule:** a review passes only if all three scores are 3 or higher.

**Problem tags:** factual error, hallucination, instruction not followed, off topic, tone issue,
safety issue, sycophancy, or none.

### About the data
- `samples.csv`: 20 prompt and response pairs written for practice, many with deliberate flaws.
  The reference ratings are the author's own judgement, not an official standard.
- `review_log.csv`: a **synthetic** log of 240 reviews (80 items, 3 reviewers) made by
  `generate_data.py` with a fixed random seed. It does not come from any real company or project.
        """
    )
