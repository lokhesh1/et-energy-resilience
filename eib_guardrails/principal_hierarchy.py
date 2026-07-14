"""
Principal Hierarchy — WHO may invoke WHAT, at the board's trust boundaries.

The constitutions guard what an agent's OUTPUT may contain; this module guards
which PRINCIPAL may trigger an action in the first place. Three principals, most
to least trusted, with strictly nested grants (everything a less-trusted principal
may do, a more-trusted one may too):

  * system         — the board's own internals (twin loop, distiller pod). Everything.
  * operator       — a human at the REST API / dashboard. Everything except direct
                     memory writes (those belong to the learning loop, not a request).
  * external_agent — an A2A peer at the front door. May discover the card, run the
                     board, and read the twin — may NOT force our compute clock
                     (refresh_twin), read the internal audit log, or write memory.

Enforcement points are the two real trust boundaries:
  * `protocols/a2a_server.py` — every `tasks/send` passes `check_permission(
    "external_agent", "run_board")` before the board runs; a denial comes back as
    a `failed` A2A Task (protocol surface stays stable, same as a board failure).
  * `api/main.py` — POST /twin/refresh checks `("operator", "refresh_twin")`.

The REST surface has no auth yet, so the principal is fixed by WHERE the request
arrives (REST = operator, A2A = external_agent). When auth lands, derive the
principal from the credential instead — the table and checks don't change.

Closed-world, least-privilege: an unknown principal or capability is DENIED (same
philosophy as the 8-corridor closed world in the constitutions). Deterministic,
no LLM, no I/O — a permission check can never fail open.
"""

# Most → least trusted. Grants are strictly nested along this order.
PRINCIPALS = ("system", "operator", "external_agent")

CAPABILITIES = frozenset({
    "discover",          # read the capability cards (A2A discovery)
    "run_board",         # run a full crisis-board query (/query, /scenario, A2A)
    "read_twin",         # read the latest twin snapshot / corridor status
    "refresh_twin",      # force an immediate twin recompute
    "read_audit",        # read the durable audit log
    "trigger_learning",  # fire the distiller pod after a run
    "write_memory",      # direct writes into xMemory (learning loop only)
})

_GRANTS: dict[str, frozenset] = {
    "external_agent": frozenset({
        "discover", "run_board", "read_twin", "trigger_learning",
    }),
    "operator": frozenset({
        "discover", "run_board", "read_twin", "trigger_learning",
        "refresh_twin", "read_audit",
    }),
    "system": CAPABILITIES,
}


def grants(principal: str) -> frozenset:
    """The capability set a principal holds. Unknown principal → empty set."""
    return _GRANTS.get(principal, frozenset())


def check_permission(principal: str, capability: str) -> dict:
    """Least-privilege check, never raises. Returns
    {"allowed": bool, "principal", "capability", "reason"} so a denial can be
    surfaced (audit entry, failed task) rather than swallowed."""
    if principal not in _GRANTS:
        return {"allowed": False, "principal": principal, "capability": capability,
                "reason": f"unknown principal '{principal}' — denied (least privilege)"}
    if capability not in CAPABILITIES:
        return {"allowed": False, "principal": principal, "capability": capability,
                "reason": f"unknown capability '{capability}' — denied (closed world)"}
    if capability not in _GRANTS[principal]:
        return {"allowed": False, "principal": principal, "capability": capability,
                "reason": f"'{principal}' is not granted '{capability}'"}
    return {"allowed": True, "principal": principal, "capability": capability,
            "reason": "granted"}


def require(principal: str, capability: str) -> None:
    """Raise PermissionError on denial — for call sites that must hard-stop."""
    result = check_permission(principal, capability)
    if not result["allowed"]:
        raise PermissionError(result["reason"])
