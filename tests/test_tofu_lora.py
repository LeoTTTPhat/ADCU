from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcu.tofu_lora import _tofu_eval_limit


def test_tofu_eval_limit_default_scores_forget01_fully(monkeypatch) -> None:
    monkeypatch.delenv("ADCU_TOFU_EVAL_LIMIT", raising=False)
    monkeypatch.delenv("ADCU_TOFU_FULL_SPLITS", raising=False)

    assert _tofu_eval_limit("forget01", 40) == 40
    assert _tofu_eval_limit("retain99", 3960) == 5


def test_tofu_eval_limit_full_scores_every_split(monkeypatch) -> None:
    monkeypatch.setenv("ADCU_TOFU_EVAL_LIMIT", "full")

    assert _tofu_eval_limit("retain99", 3960) == 3960
    assert _tofu_eval_limit("world_facts", 117) == 117
