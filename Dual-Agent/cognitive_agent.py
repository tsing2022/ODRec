import argparse
import json
from pathlib import Path
from persona_agent import construct_persona, validate_persona
from questionnaire_agent import (QuestionnaireAgent, OpenAIChatTransport, HERE,
    normalize_profile, fingerprint, read_json, atomic_json, export_proxy, validate_bank)

VERSION = "cognitive-agents-v1"


def persona_text(profile):
    return "\n".join("{}: {}".format(k, v if v is not None else "unknown")
                     for k, v in profile.items())


class CognitiveAgent:
    def __init__(self, transport, bank, memory_dir, prompts=None, max_rounds=3,
                 model_settings=None):
        self.prompts = prompts or read_json(HERE / "prompts.json")
        self.bank, self.transport = bank, transport
        self.settings = model_settings or {"transport": "injected"}
        self.max_rounds = max_rounds
        self.memory_dir = Path(memory_dir)
        self.questionnaire = QuestionnaireAgent(transport, bank, self.memory_dir / "questionnaire",
            prompts=self.prompts, max_rounds=max_rounds, model_settings=self.settings)

    def _persona(self, subject_id, source, role):
        identity = {"subject_id": str(subject_id), "role": role}
        path = self.memory_dir / "personas" / (fingerprint(identity) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(identity, version=VERSION, source_sha256=fingerprint(source),
            prompts_sha256=fingerprint(self.prompts), settings_sha256=fingerprint(self.settings),
            max_rounds=self.max_rounds)
        with self.questionnaire._lock(path):
            if path.exists():
                cached = read_json(path)
                if (not isinstance(cached, dict) or cached.get("provenance") != provenance or
                        cached.get("source_profile") != source):
                    raise ValueError("Persona memory inputs changed; use a separate memory directory")
                result = cached.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Invalid persona memory")
                if result.get("status") == "complete":
                    attempts = result.get("attempts")
                    review = attempts[-1].get("review") if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else None
                    if (result.get("role") != role or result.get("source_sha256") != fingerprint(source) or
                            validate_persona(result.get("persona"), source, role) or
                            not isinstance(review, dict) or review.get("approve") is not True or review.get("issues") != []):
                        raise ValueError("Invalid accepted persona memory")
                    return result
            result = construct_persona(source, role, self.transport, self.prompts, self.max_rounds)
            atomic_json(path, {"provenance": provenance, "source_profile": source, "result": result})
            return result

    def run(self, subject_id, profile):
        source = normalize_profile(profile)
        personas, kernels = {}, {}
        for role in ("self", "partner"):
            result = self._persona(subject_id, source, role)
            if result["status"] != "complete":
                return {"status": "needs_review", "stage": role + "_persona", "result": result}
            personas[role] = result["persona"]

            subject_profile = source if role == "self" else personas[role]["profile"]
            state = self.questionnaire.run(subject_id, subject_profile, agent_role=role)
            if state["status"] != "complete":
                return {"status": "needs_review", "stage": role + "_questionnaire",
                        "subject_id": str(subject_id)}
            kernels[role] = export_proxy(state, self.bank)
        return {"status": "complete", "subject_id": str(subject_id), "version": VERSION,
            "source_sha256": fingerprint(source), "questionnaire_sha256": fingerprint(self.bank),
            "personas": personas, "kernels": kernels,
            "semantic_texts": {"self_persona": persona_text(personas["self"]["profile"]),
                "self_kernel": kernels["self"]["proxy_text"],
                "partner_persona": persona_text(personas["partner"]["profile"]),
                "partner_kernel": kernels["partner"]["proxy_text"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--questionnaire", default=str(HERE / "questionnaire.json"))
    parser.add_argument("--memory-dir", default="agent_memory")
    parser.add_argument("--output-dir", default="agent_output")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--max-rounds", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    profiles, bank = read_json(args.profiles), read_json(args.questionnaire)
    validate_bank(bank)
    if not isinstance(profiles, dict) or not profiles:
        parser.error("--profiles must contain a nonempty object")
    for profile in profiles.values():
        normalize_profile(profile)
    settings = {"model": args.model, "temperature": 0.8, "top_p": 0.1, "max_tokens": 800}
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "subjects": len(profiles),
            "agent_roles": ["self", "partner"], "items_per_agent": len(bank["questions"]),
            "maximum_logical_model_calls": len(profiles) * args.max_rounds * (4 + 4 * len(bank["questions"])),
            "outputs": "four semantic texts; no embeddings", "version": VERSION}, indent=2))
        return 0
    agent = CognitiveAgent(OpenAIChatTransport(**settings), bank, args.memory_dir,
        max_rounds=args.max_rounds, model_settings=settings)
    incomplete = False
    for subject_id, profile in profiles.items():
        result = agent.run(subject_id, profile)
        incomplete = incomplete or result["status"] != "complete"
        atomic_json(Path(args.output_dir) / (fingerprint(str(subject_id)) + ".json"), result)
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
