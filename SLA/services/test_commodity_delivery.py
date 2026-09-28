from pathlib import Path

from SLA.dsl import load_problem
from SLA.services import compile_service


EXAMPLE = Path(__file__).parents[1] / "dsl" / "examples" / "refuelling.json"


def test_refuelling_requirements():
    problem = load_problem(EXAMPLE)
    requirements = compile_service(problem)
    assert requirements.commodity == "xenon"
    assert requirements.quantity == 80.0
    assert requirements.unit == "kg"
    assert requirements.service_duration_s == 86400.0
    assert requirements.inventory_shortfall == 0.0
    assert not requirements.needs_resupply
    assert [s.servicer_id for s in requirements.candidate_servicers] == ["SERVICER-1"]
    assert requirements.inventory_candidates[0].available_quantity == 100.0
