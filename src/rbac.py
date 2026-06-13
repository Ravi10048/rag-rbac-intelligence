"""RBAC engine.

This is the security boundary of the system. Every retrieved chunk passes
through here twice:

    1. PRE-RETRIEVAL  - we compute a metadata filter from the user's
       permissions and hand it to ChromaDB so denied documents never even
       enter the candidate set. Cheap and reduces noise.

    2. POST-RETRIEVAL - we re-validate each surviving chunk against the
       full policy set. This is defense-in-depth: if a future code change
       weakens the pre-filter (or someone passes the user object around
       wrong) the post-check still blocks the leak.

Both checks call the same `is_allowed` predicate so they can't drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import config
from src.data_models import DocumentMetadata, User


class RBACEngine:
    """Encapsulates user lookup and access-control decisions.

    Construct once at pipeline startup, then call `is_allowed` for every
    (user, document_metadata) pair you want to authorise.
    """

    def __init__(self,
                 users_path: Path | None = None,
                 policies_path: Path | None = None) -> None:
        users_path = users_path or config.POLICIES_DIR / "user_roles.json"
        policies_path = policies_path or config.POLICIES_DIR / "access_policies.json"

        self._users_raw: dict[str, dict[str, Any]] = json.loads(users_path.read_text())
        self.policies: dict[str, dict[str, Any]] = json.loads(policies_path.read_text())

    # ------------------------------------------------------------------
    # User lookup helpers
    # ------------------------------------------------------------------
    def get_user(self, email: str) -> User:
        """Look up a user by email. Raises KeyError if unknown.

        Returning a typed `User` (rather than a raw dict) keeps the rest
        of the pipeline honest about what fields it depends on.
        """
        if email not in self._users_raw:
            raise KeyError(f"Unknown user: {email}")
        u = self._users_raw[email]
        return User(
            user_id=u["user_id"],
            name=u["name"],
            department=u["department"],
            role=u["role"],
            clearance=u["clearance"],
            accessible_departments=u.get("accessible_departments", []),
        )

    def list_users(self) -> list[User]:
        """Used by the demo to iterate over the available personas."""
        return [self.get_user(e) for e in self._users_raw]

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------
    def is_allowed(self, user: User,
                   meta: DocumentMetadata) -> tuple[bool, str]:
        """Return (allowed, human-readable reason).

        Resolution order (first failure wins so we can give a precise reason):

            1. Clearance check  - user must hold sensitivity >= document's.
            2. Department check - document's dept must be in user's
               accessible_departments (empty list = wildcard).
            3. Tag-based policies - if the document carries tags matching
               a specific policy (e.g. "salary"), that policy's allowed
               roles/departments must include the user.

        The "executive" department + empty accessible_departments combine
        to give CEO-style users a true wildcard.
        """

        # --- 1. clearance gate ----------------------------------------
        user_level = config.SENSITIVITY_LEVELS.get(user.clearance, -1)
        doc_level = config.SENSITIVITY_LEVELS.get(meta.sensitivity, 999)
        if user_level < doc_level:
            return False, (
                f"clearance '{user.clearance}' is below document "
                f"sensitivity '{meta.sensitivity}'"
            )

        # --- 2. department gate ---------------------------------------
        # An empty list is intentionally the wildcard (only executives have
        # this). We *don't* want to default to wildcard for normal users
        # who forgot to set the field, so caller code should always supply
        # at least one department for non-execs.
        depts = user.accessible_departments
        is_wildcard = (not depts) or user.department == "executive"
        if not is_wildcard and meta.department not in depts:
            return False, (
                f"user's department '{user.department}' is not authorised "
                f"to read documents from '{meta.department}'"
            )

        # --- 3. tag-driven specific policies --------------------------
        # Salary data: even HR's clearance doesn't help if you're not in
        # an HR-aligned or executive role.
        if "salary" in meta.tags:
            ok, why = self._check_named_policy("salary_data", user)
            if not ok:
                return False, why

        if "revenue" in meta.tags or "earnings" in meta.tags:
            ok, why = self._check_named_policy("financial_reports", user)
            if not ok:
                return False, why

        if "incident" in meta.tags or "siem" in meta.tags:
            ok, why = self._check_named_policy("security_incidents", user)
            if not ok:
                return False, why

        return True, "all policy checks passed"

    def _check_named_policy(self, name: str,
                            user: User) -> tuple[bool, str]:
        """Check a single named policy block (salary_data, etc.).

        Returns OK if any of the policy's allow-lists matches the user.
        If a policy declares both roles and departments, satisfying either
        is enough (logical OR) - that matches how people actually write
        access rules in real systems.
        """
        policy = self.policies.get(name, {})
        if not policy:
            return True, ""  # policy not declared - nothing to enforce

        allowed_roles = policy.get("allowed_roles", [])
        allowed_depts = policy.get("allowed_departments", [])

        if allowed_roles and user.role in allowed_roles:
            return True, ""
        if allowed_depts and user.department in allowed_depts:
            return True, ""
        if not allowed_roles and not allowed_depts:
            return True, ""

        return False, (
            f"policy '{name}' restricts access to "
            f"{allowed_roles or allowed_depts}; user role='{user.role}' "
            f"department='{user.department}'"
        )

    # ------------------------------------------------------------------
    # Pre-retrieval filter
    # ------------------------------------------------------------------
    def chroma_filter(self, user: User) -> dict[str, Any] | None:
        """Build a ChromaDB `where` clause that pre-filters at search time.

        We can express the *cheap* checks (clearance level, department)
        natively in Chroma's metadata filter language; the tag-based
        policies are too fine-grained and run in the post-filter instead.

        Returning None means "no pre-filter" - used for executive users
        whose access is wide enough that filtering is a waste.
        """
        max_sensitivity = user.clearance
        # Sensitivities the user is cleared for (everything <= their level)
        allowed_sensitivities = [
            s for s, lvl in config.SENSITIVITY_LEVELS.items()
            if lvl <= config.SENSITIVITY_LEVELS[max_sensitivity]
        ]

        # Wildcard user (exec) - skip the where clause entirely
        if not user.accessible_departments or user.department == "executive":
            return {"sensitivity": {"$in": allowed_sensitivities}}

        # Normal user - intersect sensitivity AND department.
        # ChromaDB's filter DSL uses $and / $in - see
        # https://docs.trychroma.com/usage-guide#using-where-filters
        return {
            "$and": [
                {"sensitivity": {"$in": allowed_sensitivities}},
                {"department": {"$in": user.accessible_departments}},
            ]
        }
