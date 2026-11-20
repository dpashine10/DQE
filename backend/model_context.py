"""Bound untrusted dataset context before sending it to a model provider."""
import json

MAX_CONTEXT_BYTES = 48000


def bounded_json(value):
    """Return valid JSON within a hard UTF-8 budget, with explicit truncation.

    This limits dataset-derived context, not provider protocol/schema overhead or
    the response. Long identifiers may be truncated; callers must validate output.
    """
    def compact(item, string_limit, list_limit):
        if isinstance(item, dict):
            return {str(k)[:string_limit]: compact(v, string_limit, list_limit)
                    for k, v in list(item.items())[:list_limit]}
        if isinstance(item, list):
            return [compact(v, string_limit, list_limit) for v in item[:list_limit]]
        if isinstance(item, str):
            return item if len(item) <= string_limit else item[:string_limit] + '…'
        return item

    for strings, items in ((256, 100), (160, 40), (100, 15), (80, 5)):
        result = json.dumps({'context': compact(value, strings, items),
                             'truncation_notice': 'Context is bounded; local audit contains full findings.'},
                            ensure_ascii=True, default=str)
        if len(result.encode('utf-8')) <= MAX_CONTEXT_BYTES:
            return result
    return json.dumps({'truncation_notice': 'Context exceeded the budget; use local deterministic results.'})
