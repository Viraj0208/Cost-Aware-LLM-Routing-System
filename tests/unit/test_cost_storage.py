"""Tests for SQLite-backed cost storage."""

from src.cost.calculator import CostCalculator
from src.cost.storage import CostStorage


def test_storage_starts_empty(tmp_path):
    storage = CostStorage(str(tmp_path / "costs.db"))

    assert storage.get_request_count() == 0
    assert storage.get_total_cost() == 0
    assert storage.get_total_savings() == 0
    assert storage.get_recent() == []


def test_storage_records_single_cost_event(tmp_path):
    storage = CostStorage(str(tmp_path / "costs.db"))
    estimate = CostCalculator().calculate("phi-2", 100, 50)

    storage.record(estimate, complexity_score=0.25, routing_decision="small")

    assert storage.get_request_count() == 1
    assert storage.get_total_cost() == estimate.cost_usd
    assert storage.get_total_savings() == estimate.savings_usd


def test_storage_recent_events_are_returned_newest_first(tmp_path):
    storage = CostStorage(str(tmp_path / "costs.db"))
    calc = CostCalculator()

    storage.record(calc.calculate("phi-2", 100, 50), complexity_score=0.2)
    storage.record(calc.calculate("llama-2-70b", 200, 100), complexity_score=0.8)

    recent = storage.get_recent(2)
    assert len(recent) == 2
    assert recent[0]["model_name"] == "llama-2-70b"
    assert recent[1]["model_name"] == "phi-2"


def test_storage_limit_recent_results(tmp_path):
    storage = CostStorage(str(tmp_path / "costs.db"))
    calc = CostCalculator()

    for _ in range(5):
        storage.record(calc.calculate("phi-2", 10, 10))

    assert len(storage.get_recent(3)) == 3


def test_storage_clear_removes_all_events(tmp_path):
    storage = CostStorage(str(tmp_path / "costs.db"))
    storage.record(CostCalculator().calculate("phi-2", 100, 50))

    storage.clear()

    assert storage.get_request_count() == 0
    assert storage.get_total_cost() == 0
    assert storage.get_recent() == []


def test_storage_reopens_existing_database(tmp_path):
    db_path = tmp_path / "costs.db"
    estimate = CostCalculator().calculate("phi-2", 100, 50)

    CostStorage(str(db_path)).record(estimate)
    reopened = CostStorage(str(db_path))

    assert reopened.get_request_count() == 1
    assert reopened.get_total_cost() == estimate.cost_usd
