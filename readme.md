# Credit Risk Decision & Policy Copilot

A banking-focused AI system that combines a Probability of Default (PD) credit
risk model with a Retrieval-Augmented Generation (RAG) policy assistant. Given
an applicant, it predicts their default risk, explains *why* using SHAP, then
retrieves the relevant bank policy and generates an explanation grounded in
both the model's evidence and the actual policy text — never invented rules.

## Architecture

```
Applicant data ──> PD Model (XGBoost) ──> Predicted PD + SHAP risk drivers
                                                    │
                                                    ▼
                                          Credit Risk Copilot (app/app.py)
                                                    ▲
                                                    │
Policy PDFs ──> RAG pipeline (ingest → chunk → embed → retrieve) ──> Grounded explanation
```

The PD model and the RAG pipeline are trained/built independently, then
combined only at the app layer — the LLM is never allowed to reason about
credit risk on its own; it only explains what the model and the policies
already say.

## Project structure

```
credit-risk-policy-copilot/
├── data/
│   ├── raw/                     # place the full Lending Club CSV here
│   └── processed/                # notebook outputs (train/test features, etc.)
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_credit_risk_model.ipynb
│   ├── 05_model_validation.ipynb
│   └── 06_rag_evaluation.ipynb
├── src/
│   ├── rag/                      # ingestion, chunking, embeddings, vector store,
│   │                              # reranking, generation, pipeline orchestration
│   └── models/artifacts/         # saved model, preprocessor, threshold (generated)
├── policies/                     # synthetic bank policy PDFs (credit, lending,
│                                  # approval, risk, collateral, exception)
├── app/
│   └── app.py                    # Streamlit Copilot interface
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
and place the accepted-loans CSV at `data/raw/lending_club.csv`.

## Running the pipeline

Run the notebooks in order — each one reads the previous notebook's output:

1. `01_data_understanding.ipynb` — target definition, leakage-column removal
2. `02_eda.ipynb` — distributions, class imbalance strategy
3. `03_feature_engineering.ipynb` — encoding, scaling, derived ratios
4. `04_credit_risk_model.ipynb` — trains Logistic Regression + XGBoost, threshold analysis
5. `05_model_validation.ipynb` — cross-validation, SHAP, PSI drift check
6. `06_rag_evaluation.ipynb` — builds the policy index, measures retrieval quality

## Running the app

```bash
cd app
streamlit run app.py
```

To enable LLM-generated (rather than retrieval-only) explanations, set:

```bash
export ANTHROPIC_API_KEY=your-key-here   # Windows: set ANTHROPIC_API_KEY=your-key-here
```

Without a key, the app falls back to showing the raw retrieved policy excerpts.

## Deploying for others to see (e.g. a portfolio link)

The GitHub repo can stay private while the deployed app is public:

1. Push this repo to GitHub (private is fine)
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. "New app" → point it at this repo, branch, and `app/app.py`
4. Under the app's **Settings → Secrets**, add `ANTHROPIC_API_KEY` if you want live LLM answers
5. Share the resulting `*.streamlit.app` link — no GitHub access needed to view it

## Key results (full Lending Club dataset)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.719 | 0.729 |
| PR-AUC | 0.385 | 0.403 |

- 1.35M resolved loans (Fully Paid / Charged Off), 19.96% default rate
- 5-fold CV stability: ROC-AUC std = 0.0011
- RAG retrieval: Hit Rate@3 = 100%, MRR = 1.0 on the evaluation query set

## Known limitations

- PSI drift check currently compares a random train/test split of the same
  time period, which will always look stable — a temporal split (older
  vintages vs. newer) is needed to actually validate drift monitoring.
- The F1-optimal decision threshold (0.55) trades off precision/recall for a
  binary flag; a production system should tie the threshold to the actual
  cost of a false decline vs. a false approval rather than defaulting to F1.