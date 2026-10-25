"""Single-anchor mutations of the disposable candidate copy only."""

MUTATIONS = {
    "alias-admission": ('path == "/v1/runs" and not starting and selected', "False"),
    "alias-stop": (
        'and path.startswith("/v1/runs/")\n                        and path.endswith("/stop")',
        'and request.path.startswith("/v1/runs/")\n'
        '                        and path.endswith("/stop")',
    ),
    "alias-status": (
        'and path.startswith("/v1/runs/")\n                        and len(path.split("/")) == 4',
        'and request.path.startswith("/v1/runs/")\n'
        '                        and len(path.split("/")) == 4',
    ),
    "budget-binding": ("row is not self.members[run_id][2]", "False"),
    "refusal-tombstone": ("run_id in self.refused or", "False or"),
    "search-limit": ("SEARCH_LIMIT = 5", "SEARCH_LIMIT = 6"),
    "batch-names": ('names == {"web_search"}', "True"),
    "batch-phase": ('row["phase"] == "new"', "True"),
    "query-shape": ('type(args.get("query")) is str and args["query"]', "True"),
    "extract-limit": ("EXTRACT_LIMIT = 3", "EXTRACT_LIMIT = 4"),
    "url-unique": ("len(set(urls)) != len(urls)", "False"),
    "url-membership": ('any(url not in row["urls"] for url in urls)', "False"),
    "final-arguments": ("ticket[3:] != (name, frozen(args))", "False"),
    "cancel-close": ('row["closed"] = row["cancelled"] = True', "pass"),
    "authority-anchor": ("self.members is not self._membership_authority", "False"),
    "run-key": (
        'run_id = get_current_session_key(default="")',
        'run_id = "shared-synthetic-session"',
    ),
    "response-mode": ('kw.get("api_mode") != "chat_completions"', "False"),
    "model-downstream": (
        '        response = kw["next_call"](kw["request"])',
        '        try:\n            response = kw["next_call"](kw["request"])\n'
        "        except BaseException:\n            return self.refusal()",
    ),
    "tool-downstream": (
        '            return kw["next_call"](kw["args"])',
        '            try:\n                return kw["next_call"](kw["args"])\n'
        "            except BaseException:\n                return REFUSAL",
    ),
    "search-error": (
        "            result = original(args, **kw)",
        "            try:\n                result = original(args, **kw)\n"
        "            except BaseException:\n                self.refuse(run_id)\n"
        "                return REFUSAL",
    ),
    "extract-error": (
        "            result = await original(args, **kw)",
        "            try:\n                result = await original(args, **kw)\n"
        "            except BaseException:\n                self.refuse(run_id)\n"
        "                return REFUSAL",
    ),
    "http-error": ("                if native_error:", "                if False:"),
    "outcome-truth": ('"true" if run_id in self.refused else "false"', '"false"'),
    "outcome-auth": (
        'request.method == "GET"\n                        and response.status == 200',
        'request.method == "GET"\n                        and True',
    ),
    "effect-closed": (
        'row["closed"]\n                or ticket is None',
        "False\n                or ticket is None",
    ),
    "late-result": (
        '                        row["settled"] += 1\n                        if row["closed"]:',
        '                        row["settled"] += 1\n                        if False:',
    ),
}
