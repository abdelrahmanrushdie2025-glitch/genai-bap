# GenAI Business Analytics Platform: Prototype

**Big Data Challenge: final project prototype**
Team: Abdelrahman Rushdie (Project Lead), Deepak Balaji (Research & Data), Rahul Karanam (Cloud Tech),
Durga Hima Sagar (Governance & AI), Purnay Kanchi (Presentation)

This cloud-native analytics platform is built for the fictional enterprise *Aurora Meridian Group*. It brings together
structured ledger data, unstructured documents and public SEC filings. On top of that data it provides:

| Module | What it does | Technique |
|---|---|---|
| Executive Overview | KPIs, trends, mix by business unit and category | Pandas, Altair |
| Ask the Data | Plain-English question → safe SQL → executive summary | Gemini (NL→SQL) + read-only SQLite |
| Document Insights | Grounded answers over reports, memos and regulation, with sources | RAG: TF-IDF retrieval + Gemini |
| Anomaly Detection | Screens every expense; explains each alert | Audit rules + Isolation Forest |
| Forecasting | Revenue and cost forecasts with 90% bands and a backtest | Trend + seasonality least squares |
| Green Cloud | Live token, energy and CO2 meter; what-if scenario | Caching, right-sized model, offline-first |

The app **works without an API key**. It runs in an offline demo mode that uses deterministic SQL templates and
extractive answers. Adding a free Gemini key unlocks free-form GenAI answers.

## Data
- `data/ledger.csv.gz`: 60,250 synthetic general-ledger transactions (Jan 2023 to Sep 2026). It includes 250 labelled,
  injected anomalies, which are used only to measure detection quality. Real ledgers are confidential, so the
  approach follows the PaySim dataset on Kaggle. Regenerate it with `python scripts/generate_data.py`.
- `docs/`: 8 fictional internal documents (earnings summary, board memo, audit report, regulation briefing and others).
- SEC EDGAR XBRL API: real quarterly revenue and net income for 9 US financial peers. Use
  **Overview → Data sources → Refresh peer data from SEC EDGAR**. Until you refresh, an anonymised synthetic demo
  sample is shown.

## Results (from `tests/`)
- Anomaly detection: **99.6% recall** and **74% precision**. Only about 1% of transactions are sent to human review.
- Revenue forecast backtest: about **6.5% MAPE** on a 6-month holdout.
- SQL guardrails block every non-SELECT statement, and the evaluation labels are hidden from the AI.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Gemini key
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a **public** GitHub repository and upload all files in this folder.
   Do not upload `.streamlit/secrets.toml`.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
   Pick the repository, branch `main`, and main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "AIza..."
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   SEC_USER_AGENT = "GenAI-BAP University Project your.name@university.edu"
   ```
4. Click **Deploy**. Open the app a few minutes before presenting, because free apps sleep when unused.

If Gemini reports that a model is not found, change `GEMINI_MODEL` to a model listed in Google AI Studio.
The client also tries several fallback models automatically.

## Tests
```bash
python tests/test_core.py   # 8 core tests (offline; mock Gemini + mock SEC payload)
python tests/ui_smoke.py    # runs every page against a Streamlit stub
```

## Project structure
```
app.py                    Executive Overview (home)
pages/                    Ask the Data, Document Insights, Anomaly Detection, Forecasting, Green Cloud
bap/                      Core library (UI-independent, reusable behind an API)
  data.py  nl2sql.py  rag.py  anomaly.py  forecast.py  sec.py  llm.py  green.py  ui.py
data/  docs/  scripts/  tests/
```

## Production architecture (proposed, Azure)
Data Factory or Functions handle ingestion, into ADLS Gen2 (Parquet) storage. Databricks or Synapse Spark run the big-data
batch jobs, and a SQL warehouse serves the results. Azure OpenAI or Gemini sits behind a private endpoint with RAG.
The app runs on Container Apps, which scales to zero when idle. Monitoring and FinOps dashboards cover cost and carbon.
Batch workloads are scheduled in low-carbon regions such as Sweden Central.

## Responsible AI and governance
The AI only ever has read-only access, and every number can be traced: each answer shows its SQL or source passages.
Sensitive columns and evaluation labels are hidden from the model, and humans review alerts and make the decisions.
Prompts, models and token usage are logged.
The design also follows the EU AI Act, DORA and GDPR. The company, ledger and documents are fictional, and this
prototype is not financial advice.
