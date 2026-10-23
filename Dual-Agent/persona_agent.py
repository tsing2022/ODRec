import json
from questionnaire_agent import normalize_profile, fingerprint


def validate_persona(candidate, source, role):
    if not isinstance(candidate, dict):
        return ["Persona must be a JSON object"]
    values, inferred, evidence = candidate.get("profile"), candidate.get("inferred_fields"), candidate.get("evidence")
    errors = []
    if not isinstance(values, dict) or set(values) != set(source):
        errors.append("Profile must preserve the source field names")
    if (not isinstance(inferred, list) or any(not isinstance(x, str) or x not in source for x in inferred) or
            len(inferred) != len(set(x for x in inferred if isinstance(x, str)))):
        errors.append("inferred_fields must list distinct existing field names")
    if not isinstance(evidence, dict) or any(x not in source for x in evidence):
        errors.append("evidence must map source field names to quotations")
    if errors:
        return errors
    missing = ("", "unknown", "secrecy", "none", "null")
    for field, value in values.items():
        known = source[field].strip().lower() not in missing
        if value is not None and not isinstance(value, str):
            errors.append("Profile values must be strings or null")
            continue
        if role == "self" and known:
            if value != source[field] or field in inferred:
                errors.append("Observed self attribute changed: " + field)
            continue
        if value is None:
            if field in inferred or evidence.get(field):
                errors.append("Unknown attributes cannot carry an inferred value or evidence: " + field)
            continue
        quotes = evidence.get(field, [])
        if field not in inferred or not isinstance(quotes, list) or not quotes:
            errors.append("Inferred attribute requires its flag and source evidence: " + field)
            continue
        for quote in quotes:
            if (not isinstance(quote, dict) or not isinstance(quote.get("field"), str) or quote["field"] not in source or
                    not isinstance(quote.get("quote"), str) or not quote["quote"].strip() or
                    quote["quote"] not in source[quote["field"]]):
                errors.append("Inferred attribute evidence must quote the source")
    return errors


def construct_persona(profile, role, transport, prompts, max_rounds=3):
    if role not in ("self", "partner"):
        raise ValueError("role must be self or partner")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be 1..10")
    source = normalize_profile(profile)
    memory = json.dumps({"source_profile": source, "source_sha256": fingerprint(source)}, ensure_ascii=False)
    feedback, attempts = [], []
    for round_id in range(1, max_rounds + 1):
        attempt = {"round": round_id}
        try:
            candidate = json.loads(transport([
                {"role": "system", "content": prompts["persona_" + role] + "\nImmutable source profile:\n" + memory},
                {"role": "user", "content": json.dumps({"revision_feedback": feedback})}]))
            attempt["candidate"] = candidate
            errors = validate_persona(candidate, source, role)
            review = None
            if not errors:
                review = json.loads(transport([
                    {"role": "system", "content": prompts["persona_reviewer"] + "\nImmutable source profile:\n" + memory},
                    {"role": "user", "content": json.dumps({"role": role, "candidate": candidate}, ensure_ascii=False)}]))
                if (not isinstance(review, dict) or review.get("approve") is not True or
                        review.get("issues") != []):
                    errors = ["Reflector rejected the persona"]
                    if isinstance(review, dict) and isinstance(review.get("issues"), list):
                        errors.extend(x for x in review["issues"] if isinstance(x, str))
            attempt["review"], attempt["issues"] = review, errors
        except (ValueError, TypeError, KeyError, AttributeError):
            errors = ["Persona response must match the required JSON schema"]
            attempt["issues"] = errors
        attempts.append(attempt)
        if not errors:
            return {"status": "complete", "role": role, "source_sha256": fingerprint(source),
                    "persona": candidate, "attempts": attempts}
        feedback = errors
    return {"status": "needs_review", "role": role, "source_sha256": fingerprint(source),
            "attempts": attempts}
