"""Core test-suite (WP4 Testing & Compliance). Run:  python -m pytest -q   or   python tests/test_core.py

Uses a local mock of the Gemini REST API and a mock SEC XBRL payload, so it
runs offline and costs no tokens.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bap import anomaly, data, forecast, green, llm, nl2sql, rag, sec  # noqa: E402


# ------------------------------------------------------------------ fixtures
class _MockGemini(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _MockGemini.calls += 1
        sys_txt = body.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        if "SQLite SELECT" in sys_txt:
            text = ("```sql\nSELECT business_unit, ROUND(SUM(amount_eur),2) AS revenue_eur FROM ledger "
                    "WHERE account_type='Revenue' GROUP BY 1 ORDER BY 2 DESC\n```")
        else:
            text = "Retail Banking leads revenue. [mock]"
        out = {"candidates": [{"content": {"parts": [{"text": text}]}}],
               "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30, "totalTokenCount": 150}}
        raw = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def log_message(self, *a):
        pass


def _start_mock():
    srv = HTTPServer(("127.0.0.1", 0), _MockGemini)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ------------------------------------------------------------------ tests
def test_ledger_shape():
    df = data.load_ledger()
    assert len(df) > 50_000 and df["amount_eur"].min() > 0
    assert set(df["account_type"]) == {"Revenue", "Expense"}


def test_sql_guardrails():
    con = data.sqlite_connection(data.load_ledger())
    for bad in ["DELETE FROM ledger", "SELECT 1; DROP TABLE ledger", "PRAGMA table_info(ledger)",
                "WITH x AS (SELECT 1) INSERT INTO ledger VALUES (1)"]:
        try:
            nl2sql.run_sql(con, bad)
            raise AssertionError(f"not blocked: {bad}")
        except ValueError:
            pass
    # evaluation labels are hidden from the query engine
    cols = nl2sql.run_sql(con, "SELECT * FROM ledger LIMIT 1").columns
    assert "is_injected_anomaly" not in cols and "anomaly_type" not in cols


def test_offline_nl2sql():
    led = data.load_ledger(); s, _ = data.load_sec()
    con = data.sqlite_connection(led, s)
    for q in nl2sql.SUGGESTED_QUESTIONS:
        r = nl2sql.ask(q, con, use_llm=False)
        assert r.mode == "offline" and not r.df.empty, q


def test_anomaly_quality():
    ev = anomaly.evaluate(anomaly.detect(data.load_ledger()))
    assert ev["recall"] >= 0.95 and ev["precision"] >= 0.6, ev


def test_forecast_accuracy():
    s = forecast.ledger_monthly(data.load_ledger(), "Revenue")
    assert forecast.backtest_mape(s, 6) < 12
    fc = forecast.fit_forecast(s, 6)
    assert fc["forecast"].notna().sum() == 6 and (fc["upper"].dropna() > fc["lower"].dropna()).all()


def test_rag_retrieval():
    r = rag.Retriever(data.load_documents())
    top = r.search("internal audit split invoices approval threshold", k=1)[0][0]
    assert "Audit" in top.title
    top = r.search("DORA operational resilience", k=1)[0][0]
    assert "Regulatory" in top.title


def test_sec_normaliser():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"val": 100, "frame": "CY2024Q1"}, {"val": 110, "frame": "CY2024Q2"},
            {"val": 120, "frame": "CY2024Q3"}, {"val": 460, "frame": "CY2024"},
            {"val": 999, "frame": "CY2024Q2I"}]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"val": 10, "frame": "CY2024Q1"}, {"val": 11, "frame": "CY2024Q2"},
            {"val": 12, "frame": "CY2024Q3"}, {"val": 46, "frame": "CY2024"}]}}}}}
    df = sec.normalise("TST", "Test Corp", facts)
    assert list(df["fiscal_quarter"]) == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    assert df.loc[df.fiscal_quarter == "2024Q4", "revenue_usd"].item() == 130  # derived Q4
    assert df.loc[df.fiscal_quarter == "2024Q4", "net_income_usd"].item() == 13


def test_llm_client_cache_and_green_metrics():
    srv = _start_mock()
    old_api = llm.API
    llm.API = f"http://127.0.0.1:{srv.server_port}/v1beta/models/{{model}}:generateContent"
    os.environ["GEMINI_API_KEY"] = "test-key"
    llm.clear_cache(); green.reset()
    try:
        led = data.load_ledger(); con = data.sqlite_connection(led)
        r1 = nl2sql.ask("Revenue per business unit please", con)
        assert r1.mode == "gemini" and r1.df.iloc[0]["business_unit"] == "Retail Banking"
        calls = _MockGemini.calls
        r2 = nl2sql.ask("Revenue per business unit please", con)
        assert r2.mode == "gemini (cached)" and _MockGemini.calls == calls  # zero new API calls
        m = green.load()
        assert m.cache_hits == 2 and m.tokens_saved_by_cache == 300 and m.llm_calls == 2
    finally:
        llm.API = old_api; os.environ.pop("GEMINI_API_KEY", None)
        llm.clear_cache(); green.reset(); srv.shutdown()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
