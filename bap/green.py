"""Green-cloud accounting: token usage, cache savings and estimated energy / CO2.

The energy factors are ILLUSTRATIVE ASSUMPTIONS for the prototype and are
configurable. Published per-query estimates for LLM inference vary by more than
an order of magnitude depending on model size, hardware and data centre.
"""
from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import asdict, dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / ".cache" / "green_metrics.json"

# --- assumptions (documented in the presentation) ---
WH_PER_1K_TOKENS_LIGHT = 0.3   # small "Flash-Lite" class model
WH_PER_1K_TOKENS_LARGE = 3.0   # large frontier model, for comparison
GRID_KG_CO2_PER_KWH = 0.25     # approx. EU-average grid intensity
GREEN_REGION_KG_CO2_PER_KWH = 0.03  # region powered mostly by renewables (e.g. Sweden)

_lock = threading.Lock()


@dataclass
class GreenMetrics:
    llm_calls: int = 0
    cache_hits: int = 0
    offline_answers: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_saved_by_cache: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        n = self.llm_calls + self.cache_hits
        return self.cache_hits / n if n else 0.0

    def energy_wh(self, wh_per_1k: float = WH_PER_1K_TOKENS_LIGHT) -> float:
        return self.total_tokens / 1000 * wh_per_1k

    def saved_wh(self, wh_per_1k: float = WH_PER_1K_TOKENS_LIGHT) -> float:
        return self.tokens_saved_by_cache / 1000 * wh_per_1k

    def co2_g(self, wh: float, intensity: float = GRID_KG_CO2_PER_KWH) -> float:
        return wh / 1000 * intensity * 1000  # Wh -> kWh -> kg -> g


def load() -> GreenMetrics:
    try:
        return GreenMetrics(**json.loads(METRICS_FILE.read_text()))
    except Exception:
        return GreenMetrics()


def record(**increments: int) -> None:
    with _lock:
        m = load()
        for k, v in increments.items():
            setattr(m, k, getattr(m, k) + int(v))
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.write_text(json.dumps(asdict(m)))


def reset() -> None:
    METRICS_FILE.unlink(missing_ok=True)


def scenario(queries_per_day: int, avg_tokens: int, cache_rate: float) -> dict:
    """Annual energy/CO2 for a baseline vs. the optimised green design."""
    yearly_tokens = queries_per_day * avg_tokens * 365
    base_kwh = yearly_tokens / 1000 * WH_PER_1K_TOKENS_LARGE / 1000
    green_tokens = yearly_tokens * (1 - cache_rate)
    green_kwh = green_tokens / 1000 * WH_PER_1K_TOKENS_LIGHT / 1000
    return {
        "baseline_kwh": base_kwh,
        "green_kwh": green_kwh,
        "baseline_kg_co2": base_kwh * GRID_KG_CO2_PER_KWH,
        "green_kg_co2": green_kwh * GREEN_REGION_KG_CO2_PER_KWH,
        "reduction_pct": 100 * (1 - (green_kwh * GREEN_REGION_KG_CO2_PER_KWH) / (base_kwh * GRID_KG_CO2_PER_KWH)) if base_kwh else 0.0,
    }
