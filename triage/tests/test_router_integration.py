"""End-to-end routing on the built-in mock provider (zero keys, no network).

conftest forces the mock provider and redirects all state to a temp dir, so these
exercise the real pipeline (tier assignment, tool route, long-context flag, the
response contract) without touching committed data/ or a real API.
"""
import asyncio

from app.router.router import TIER_NAMES, route_and_answer
from app.schemas import ChatRequest, ChatResponse


def _run(req):
    return asyncio.run(route_and_answer(req))


def test_calculator_tool_route_is_free_and_exact():
    resp, telem = _run(ChatRequest(message="what is 2 + 2?"))
    assert resp.route.tier_name == "tool"
    assert resp.status == "OK"
    assert resp.cost.est_cost_usd == 0.0
    assert "4" in resp.answer
    assert telem["tier_name"] == "tool"


def test_confident_question_ships_without_abstain():
    resp, _ = _run(ChatRequest(message="What is the capital of France?", model=None))
    assert isinstance(resp, ChatResponse)
    assert resp.status == "OK"
    assert resp.route.abstained is False
    assert resp.route.tier in TIER_NAMES
    assert resp.model


def test_long_context_prompt_sets_the_flag():
    resp, _ = _run(ChatRequest(message="summarize this: " + ("word " * 20000)))
    assert resp.route.long_context is True


def test_response_contract_is_wellformed():
    resp, telem = _run(ChatRequest(message="What is the capital of Japan?"))
    assert resp.request_id and len(resp.request_id) == 12
    assert resp.route.tier_name == TIER_NAMES[resp.route.tier]
    assert resp.cost.est_cost_usd >= 0.0
    assert set(["tier", "status", "abstained", "est_cost_usd"]).issubset(telem.keys())


def test_no_escalation_into_mock_target():
    # escalate_to a real catalog id, but under force-mock it resolves to mock;
    # the router must refuse to 'escalate' into a mock target.
    resp, _ = _run(ChatRequest(message="What is the capital of Japan?",
                               model=None, escalate_to="gpt-4o"))
    assert resp.route.escalated is False
