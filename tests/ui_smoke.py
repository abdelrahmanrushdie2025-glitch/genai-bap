"""UI smoke test: executes every Streamlit page against a stub `streamlit`/`altair`
module so code paths (including button branches) run without a browser.
Run:  python tests/ui_smoke.py
(With real Streamlit installed you can also use streamlit.testing.v1.AppTest.)
"""
from __future__ import annotations

import pathlib
import runpy
import sys
import traceback
import types
from unittest.mock import MagicMock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Rerun(Exception):
    pass


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def make_streamlit(press_buttons: bool, radio_index: int = 0):
    st = MagicMock(name="streamlit")
    st.session_state = SessionState()
    st.secrets = {}
    calls = []

    def passthrough(*a, **k):
        if a and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

    st.cache_data = passthrough
    st.cache_resource = passthrough

    def columns(spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [st for _ in range(n)]

    st.columns = columns
    st.tabs = lambda labels: [st for _ in labels]
    st.button = lambda label, *a, **k: (calls.append(("button", label)) or press_buttons) and "Clear" not in label and "Reset" not in label
    st.download_button = lambda *a, **k: False
    def selectbox(label, options, index=0, **k):
        opts = list(options)
        if not opts:
            return None
        return opts[1] if str(opts[0]).startswith("- ") and press_buttons and len(opts) > 1 else opts[index]

    st.selectbox = selectbox
    st.radio = lambda label, options, **k: list(options)[min(radio_index, len(list(options)) - 1)]
    st.multiselect = lambda label, options, default=None, **k: default or []
    st.slider = lambda label, lo=None, hi=None, value=None, step=None, **k: value
    st.text_input = lambda label, value="", **k: value
    st.chat_input = lambda *a, **k: None
    st.toggle = lambda label, value=False, key=None, **k: st.session_state.get(key, value)

    def rerun():
        raise _Rerun()

    st.rerun = rerun
    for name in ("spinner", "expander", "chat_message", "container", "sidebar", "form"):
        cm = MagicMock()
        cm.__enter__ = lambda s, *a: st
        cm.__exit__ = lambda s, *a: False
        setattr(st, name, (lambda c: (lambda *a, **k: c))(cm))
    st.sidebar = st  # `with st.sidebar:` and st.sidebar.xxx
    st.__enter__ = lambda s, *a: st
    st.__exit__ = lambda s, *a: False

    def chart(obj, *a, **k):
        calls.append(("chart", obj))

    st.altair_chart = chart
    st.dataframe = lambda data, *a, **k: calls.append(("dataframe", getattr(data, "shape", None)))
    st.metric = lambda label, value, *a, **k: calls.append(("metric", label, value))
    st._calls = calls
    return st


class Chainable:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: self

    def __add__(self, other):
        return self


def make_altair():
    alt = types.ModuleType("altair")
    for n in ("Chart", "X", "Y", "Color", "Scale", "Axis", "Tooltip"):
        setattr(alt, n, Chainable)
    return alt


PAGES = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))


def run(press: bool, radio: int = 0) -> int:
    fails = 0
    for page in PAGES:
        st = make_streamlit(press, radio)
        sys.modules["streamlit"] = st
        sys.modules["altair"] = make_altair()
        for m in [m for m in sys.modules if m == "bap" or m.startswith("bap.")]:
            del sys.modules[m]
        try:
            runpy.run_path(str(ROOT / page), run_name="__main__")
            status = "ok"
        except _Rerun:
            status = "ok (rerun)"
        except Exception:
            fails += 1
            status = "FAIL\n" + traceback.format_exc()
        metrics = [c for c in st._calls if c[0] == "metric"]
        charts = [c for c in st._calls if c[0] == "chart"]
        print(f"[buttons={press} radio={radio}] {page}: {status} | metrics={len(metrics)} charts={len(charts)}")
        if press and page == "app.py":
            print("   sample metrics:", [(m[1], m[2]) for m in metrics[:4]])
    return fails


def run_with_mock_gemini() -> int:
    import os
    from tests.test_core import _start_mock
    srv = _start_mock()
    os.environ["GEMINI_API_KEY"] = "test-key"
    import bap.llm as L
    url = f"http://127.0.0.1:{srv.server_port}/v1beta/models/{{model}}:generateContent"
    orig = L.API
    # patch the module source default so re-imports pick it up
    src = (ROOT / "bap" / "llm.py").read_text()
    try:
        (ROOT / "bap" / "llm.py").write_text(src.replace(orig, url))
        return run(True)
    finally:
        (ROOT / "bap" / "llm.py").write_text(src)
        os.environ.pop("GEMINI_API_KEY", None)
        srv.shutdown()


if __name__ == "__main__":
    total = run(False) + run(True) + run(True, radio=1)
    print("--- with mock Gemini API ---")
    total += run_with_mock_gemini()
    from bap import green, llm
    green.reset(); llm.clear_cache()
    sys.exit(1 if total else 0)
