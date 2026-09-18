from __future__ import annotations
import os
import sys
import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.rag.pipeline import RAGPipeline  

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "src", "models", "artifacts")
POLICIES_DIR = os.path.join(os.path.dirname(__file__), "policies")

RISK_GRADE_BANDS = [
    (0.05, "A", "Minimal risk"),
    (0.10, "B", "Low risk"),
    (0.15, "C", "Low-moderate risk"),
    (0.22, "D", "Moderate risk"),
    (0.30, "E", "Elevated risk"),
    (0.35, "F", "High risk"),
    (1.01, "G", "Very high risk"),
]

AUTO_APPROVE_MAX_PD = 0.15
AUTO_APPROVE_GRADES = {"A", "B", "C"}


def risk_grade(pd_score: float) -> tuple[str, str]:
    for threshold, grade, label in RISK_GRADE_BANDS:
        if pd_score < threshold:
            return grade, label
    return "G", "Very high risk"

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ARTIFACTS_DIR, "pd_model_xgb.joblib"))


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_test_data():
    return pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))


@st.cache_resource
def load_rag_pipeline():
    pipeline = RAGPipeline()
    index_dir = os.path.join(ARTIFACTS_DIR, "..", "..", "rag", "index")
    if os.path.isdir(index_dir):
        pipeline.load_index(index_dir)
    else:
        pipeline.build_index(POLICIES_DIR)
    return pipeline


def top_shap_drivers(shap_values, feature_names, n=5) -> dict[str, float]:
    contrib = pd.Series(shap_values, index=feature_names).sort_values(key=abs, ascending=False)
    return contrib.head(n).to_dict()


def main():
    st.set_page_config(page_title="Credit Risk Decision & Policy Copilot", layout="wide")
    st.title("Credit Risk Decision & Policy Copilot")
    st.caption(
        "Predicted default risk, the model's own evidence for that prediction, and the bank "
        "policy that governs the decision — in one place."
    )

    try:
        model = load_model()
        test_df = load_test_data()
    except FileNotFoundError as e:
        st.error(
            f"Missing artifact: {e}\n\n"
            "Run notebooks 03 and 04 first to produce the processed data and trained model."
        )
        st.stop()

    X_test = test_df.drop(columns=["target"])
    explainer = load_explainer(model)

    st.sidebar.header("Applicant")
    applicant_idx = st.sidebar.number_input(
        "Test-set row index", min_value=0, max_value=len(X_test) - 1, value=0, step=1,
    )
    applicant = X_test.iloc[[applicant_idx]]
    actual_outcome = "Charged Off" if test_df.iloc[applicant_idx]["target"] == 1 else "Fully Paid"
    st.sidebar.caption(f"Actual historical outcome: **{actual_outcome}**")

    pd_score = float(model.predict_proba(applicant)[0, 1])
    grade, grade_label = risk_grade(pd_score)

    shap_values = explainer.shap_values(applicant)[0]
    drivers = top_shap_drivers(shap_values, applicant.columns.tolist(), n=6)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted probability of default", f"{pd_score:.1%}")
    col2.metric("Risk Grade", f"{grade} — {grade_label}")

    auto_approve_eligible = pd_score <= AUTO_APPROVE_MAX_PD and grade in AUTO_APPROVE_GRADES
    col3.metric("Auto-approval eligible?", "Yes" if auto_approve_eligible else "No — review required")

    st.subheader("Top risk drivers (SHAP)")
    driver_df = pd.DataFrame(
        [{"feature": k, "shap_value": v} for k, v in drivers.items()]
    ).sort_values("shap_value", key=abs, ascending=True)
    st.bar_chart(driver_df.set_index("feature"))
    st.caption(
        "Positive values push the prediction toward higher default risk; negative values push toward lower risk."
    )

    st.divider()
    st.subheader("Ask the Copilot")
    st.caption(
        "Answers combine this applicant's model evidence above with retrieved bank policy — "
        "never invented rules."
    )

    default_question = "Why can't this applicant be automatically approved?" if not auto_approve_eligible \
        else "What policy allows this applicant to be automatically approved?"
    question = st.text_input("Question", value=default_question)

    if st.button("Ask", type="primary"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning(
                "ANTHROPIC_API_KEY is not set in this environment, so the Copilot can't call "
                "the LLM to generate an answer. Retrieval-only results are shown below instead."
            )
            rag = load_rag_pipeline()
            retrieved = rag.retrieve(question, k=4)
            for chunk, score in retrieved:
                with st.expander(f"{chunk.source} (similarity={score:.3f})"):
                    st.write(chunk.text)
        else:
            rag = load_rag_pipeline()
            with st.spinner("Retrieving policy and generating explanation..."):
                result = rag.answer(question, pd_score=pd_score, risk_drivers=drivers, k=4)
            st.write(result["answer"])
            st.caption("Sources:")
            for s in result["retrieved_sources"]:
                st.caption(f"- {s['source']} (similarity={s['score']:.3f})")


if __name__ == "__main__":
    main()