"""Tests for the RBAC engine.

This is the part of the system where a regression is most painful (data
leak), so it gets the most test coverage. We test the engine directly --
unit-level -- because the integration test of "ask via the pipeline" is
slower and depends on the vector store being populated.

Run with:
    python -m pytest tests/

or, without pytest installed:
    python tests/test_rbac.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_models import DocumentMetadata
from src.rbac import RBACEngine


def _meta(department: str, sensitivity: str,
          tags: list[str] | None = None) -> DocumentMetadata:
    """Convenience: build a DocumentMetadata with sensible defaults."""
    return DocumentMetadata(
        doc_id="TEST",
        source_path="/tmp/test",
        source_type="pdf",
        department=department,
        sensitivity=sensitivity,
        title="test doc",
        tags=tags or [],
    )


def test_engineer_blocked_from_salary_data():
    """An engineer with restricted clearance still can't see salaries:
    the salary_data policy is role-gated, not clearance-gated."""
    eng = RBACEngine()
    bob = eng.get_user("bob@acme.com")
    md = _meta("hr", "confidential", tags=["salary", "directory"])
    ok, reason = eng.is_allowed(bob, md)
    assert not ok, f"Bob (engineer) must not see salary data, got: {reason}"
    assert "salary_data" in reason


def test_hr_can_see_salary_data():
    """HR manager has the right role -> salary access is allowed."""
    eng = RBACEngine()
    alice = eng.get_user("alice@acme.com")
    md = _meta("hr", "confidential", tags=["salary", "directory"])
    ok, _ = eng.is_allowed(alice, md)
    assert ok


def test_finance_blocked_from_engineering_security():
    """Finance can't read engineering's security docs. Either the
    clearance gate (restricted > confidential), the department gate
    (engineering not in finance's accessible list), or the
    security_incidents policy can block -- any one is fine."""
    eng = RBACEngine()
    carol = eng.get_user("carol@acme.com")
    md = _meta("engineering", "restricted", tags=["incident"])
    ok, reason = eng.is_allowed(carol, md)
    assert not ok
    assert any(k in reason for k in ("clearance", "department",
                                     "security_incidents"))


def test_finance_blocked_from_engineering_at_confidential_level():
    """The pure department gate: same clearance, wrong silo."""
    eng = RBACEngine()
    carol = eng.get_user("carol@acme.com")
    md = _meta("engineering", "confidential", tags=["incident"])
    ok, reason = eng.is_allowed(carol, md)
    assert not ok
    assert ("department" in reason) or ("security_incidents" in reason)


def test_ceo_can_see_everything():
    """Executive role with empty accessible_departments acts as wildcard."""
    eng = RBACEngine()
    david = eng.get_user("david@acme.com")
    for dept in ("hr", "finance", "engineering"):
        for sens in ("public", "internal", "confidential", "restricted"):
            md = _meta(dept, sens, tags=["salary", "revenue", "incident"])
            ok, reason = eng.is_allowed(david, md)
            assert ok, f"CEO blocked from {dept}/{sens}: {reason}"


def test_clearance_gate_blocks_underclassified():
    """Alice (confidential clearance) can't read restricted docs even
    in her own department."""
    eng = RBACEngine()
    alice = eng.get_user("alice@acme.com")
    md = _meta("hr", "restricted")
    ok, reason = eng.is_allowed(alice, md)
    assert not ok
    assert "clearance" in reason


def test_chroma_filter_for_normal_user():
    """The pre-filter should include both sensitivity AND department."""
    eng = RBACEngine()
    alice = eng.get_user("alice@acme.com")
    f = eng.chroma_filter(alice)
    # Confidential clearance -> 4 levels allowed (public..confidential)
    assert "$and" in f
    sens_clause = next(c for c in f["$and"] if "sensitivity" in c)
    assert set(sens_clause["sensitivity"]["$in"]) == {
        "public", "internal", "confidential",
    }


def test_chroma_filter_for_executive_is_wildcard_on_department():
    """Execs should NOT have a department filter (they see all silos)."""
    eng = RBACEngine()
    david = eng.get_user("david@acme.com")
    f = eng.chroma_filter(david)
    # Only sensitivity filter, no department filter
    assert "department" not in str(f)


# ---------------------------------------------------------------------------
# Lightweight runner when pytest isn't available
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
