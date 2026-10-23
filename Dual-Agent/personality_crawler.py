import argparse
import copy
from html.parser import HTMLParser
import json
import re
import time
from urllib.parse import unquote
from pathlib import Path

from questionnaire_agent import HERE, atomic_json, fingerprint, read_json, validate_bank

BASE_URL = "https://www.16personalities.com"
ANSWER_SCORES = {1: -3, 2: -2, 3: -1, 4: 1, 5: 2, 6: 3}


class CrawlerError(RuntimeError):
    pass


def validate_answers(answers, bank):
    validate_bank(bank)
    if [q["id"] for q in bank["questions"]] != list(range(1, 61)):
        raise ValueError("The crawler requires the 60 questions in ID order 1..60")
    if isinstance(answers, str):
        if re.fullmatch(r"[1-6]{60}", answers) is None:
            raise ValueError("answer_string must contain exactly 60 ASCII digits 1..6")
        values = [int(x) for x in answers]
    elif isinstance(answers, (list, tuple)):
        values = list(answers)
    else:
        raise ValueError("Answers must be a 60-digit string or a list of 60 integers")
    if len(values) != 60 or any(type(x) is not int or x not in ANSWER_SCORES for x in values):
        raise ValueError("Every answer must be an integer 1..6; null answers cannot be submitted")
    return values


def build_payload(answers, bank):
    values = validate_answers(answers, bank)
    return {
        "questions": [{"text": q["text"], "answer": ANSWER_SCORES[a]}
                      for q, a in zip(bank["questions"], values)],
        "gender": None, "inviteCode": "", "teamInviteKey": None, "extraData": []
    }


class _ReportText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1

    def handle_data(self, text):
        if not self.hidden and text.strip():
            self.parts.append(text.strip())


def plain_text(value):
    if not isinstance(value, str):
        return ""
    parser = _ReportText()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


def render_report(report):
    if not isinstance(report, dict):
        raise CrawlerError("The report response must be a JSON object")
    content = report.get("content", {})
    personality = report.get("personality", {})
    if not isinstance(content, dict) or not isinstance(personality, dict):
        raise CrawlerError("The report response has an invalid structure")
    intro = content.get("intro", {})
    if not isinstance(intro, dict):
        raise CrawlerError("The report introduction has an invalid structure")
    traits = personality.get("traits", [])
    if not isinstance(traits, list):
        raise CrawlerError("The report traits must be a list")
    if not (intro.get("traitsTitle") or intro.get("parts") or traits):
        raise CrawlerError("The response does not contain a personality report")
    lines = []
    def add(value):
        text = plain_text(value)
        if text:
            lines.append(text)
    add(intro.get("traitsTitle"))
    for part in intro.get("parts", []):
        add(part)
    if traits:
        lines.append("Personality Traits")
    for trait in traits:
        if not isinstance(trait, dict):
            raise CrawlerError("A report trait must be an object")
        name, pct = plain_text(trait.get("trait")), trait.get("pct")
        if name:
            lines.append(name + (": {}%".format(pct) if pct is not None else ""))
        add(trait.get("label"))
        add(trait.get("snippet"))
    for name in ("career", "personalGrowth", "relationships"):
        section = content.get(name, {})
        if not isinstance(section, dict):
            raise CrawlerError("A report section must be an object")
        add(section.get("title"))
        add(section.get("intro"))
        for key, heading in (("traits", "Influential Traits"), ("strengths", "Your Strengths"),
                             ("weaknesses", "Your Weaknesses")):
            entries = section.get(key, [])
            if not isinstance(entries, list):
                raise CrawlerError("Report section entries must be a list")
            if entries:
                lines.append(heading)
            for entry in entries:
                if not isinstance(entry, dict):
                    raise CrawlerError("A report entry must be an object")
                add(entry.get("title"))
                add(entry.get("subtitle"))
    if not lines:
        raise CrawlerError("The personality report text is empty")
    return "\n".join(lines)


class PersonalityCrawler:
    def __init__(self, session=None, timeout=30, attempts=3, delay=1, sleep=time.sleep):
        import requests
        if type(attempts) is not int or not 1 <= attempts <= 3 or timeout <= 0 or delay < 0:
            raise ValueError("Use 1..3 attempts, a positive timeout and a nonnegative delay")
        self.session = session if session is not None else requests.Session()
        self.timeout, self.attempts, self.delay, self.sleep = timeout, attempts, delay, sleep
        self.session.headers.update({
            "User-Agent": "ODRec-questionnaire-report/1.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": BASE_URL, "Referer": BASE_URL + "/free-personality-test"
        })

    def _request(self, method, path, payload=None, as_json=True):
        import requests
        for attempt in range(self.attempts):
            try:
                kwargs = {"timeout": self.timeout}
                if payload is not None:
                    kwargs["json"] = payload
                response = self.session.request(method, BASE_URL + path, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt + 1 == self.attempts:
                    raise CrawlerError("Questionnaire service connection failed") from None
                self.sleep(self.delay * (attempt + 1))
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < self.attempts:
                retry_after = response.headers.get("Retry-After", "")
                wait = min(float(retry_after), 30) if retry_after.isdigit() else self.delay * (attempt + 1)
                self.sleep(wait)
                continue
            if not 200 <= response.status_code < 300:
                raise CrawlerError("Questionnaire service returned HTTP {}".format(response.status_code))
            if not as_json:
                return response
            try:
                value = response.json()
            except ValueError:
                raise CrawlerError("Questionnaire service returned a non-JSON response") from None
            if not isinstance(value, dict) or value.get("success") is False or value.get("errors"):
                raise CrawlerError("Questionnaire service did not accept the request")
            return value
        raise CrawlerError("Questionnaire request did not complete")

    def submit(self, answers, bank, subject_id, role, output_dir):
        if role not in ("self", "partner"):
            raise ValueError("role must be self or partner")
        values = validate_answers(answers, bank)
        payload = build_payload(values, bank)
        identity = {"subject_id": str(subject_id), "agent_role": role}
        provenance = {"questionnaire_sha256": fingerprint(bank), "answers_sha256": fingerprint(values)}
        path = Path(output_dir) / (fingerprint(identity) + ".json")
        if path.exists():
            cached = read_json(path)
            if (not isinstance(cached, dict) or cached.get("status") != "complete" or
                    any(cached.get(k) != v for k, v in dict(identity, **provenance).items()) or
                    cached.get("answer_string") != "".join(map(str, values)) or
                    cached.get("report_text") != render_report(cached.get("report"))):
                raise ValueError("Report cache does not match the answers and questionnaire")
            return cached
        self._request("GET", "/free-personality-test", as_json=False)
        token = self.session.cookies.get("XSRF-TOKEN")
        if token:
            self.session.headers["X-XSRF-TOKEN"] = unquote(token)
        self._request("POST", "/test-results", payload)
        self.sleep(self.delay)
        report = self._request("POST", "/api/type-descriptions/results", payload)
        result = dict(identity, **provenance, status="complete",
                      answer_string="".join(map(str, values)),
                      source=BASE_URL, report=report, report_text=render_report(report))
        atomic_json(path, result)
        return result


def attach_reports(result, bank, crawler, output_dir):
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise ValueError("Only a completed cognitive-agent result can be submitted")
    ready = {}
    for role in ("self", "partner"):
        kernel = result.get("kernels", {}).get(role, {})
        if kernel.get("questionnaire_sha256") != fingerprint(bank):
            raise ValueError("Agent output uses a different questionnaire")
        values = validate_answers(kernel.get("answer_string"), bank)
        if values != kernel.get("item_responses"):
            raise ValueError("answer_string and item_responses disagree")
        ready[role] = values
    updated = copy.deepcopy(result)
    updated["questionnaire_reports"] = {}
    for role in ("self", "partner"):
        report = crawler.submit(ready[role], bank, result["subject_id"], role, output_dir)
        updated["questionnaire_reports"][role] = report
        updated["semantic_texts"][role + "_kernel"] = report["report_text"]
    updated["kernel_text_source"] = "16personalities_report"
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Submit 60 questionnaire answers and retrieve the report")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--answers-json", help="JSON mapping subject IDs to 60 digits or 60 integers")
    inputs.add_argument("--agent-result", help="Completed CognitiveAgent output JSON")
    parser.add_argument("--questionnaire", default=str(HERE / "questionnaire.json"))
    parser.add_argument("--output-dir", default="personality_reports")
    parser.add_argument("--role", choices=("self", "partner"), default="self")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    bank = read_json(args.questionnaire)
    if args.agent_result:
        result = read_json(args.agent_result)
        if not isinstance(result, dict) or result.get("status") != "complete":
            parser.error("--agent-result must contain a completed cognitive-agent result")
        for role in ("self", "partner"):
            kernel = result.get("kernels", {}).get(role, {})
            values = validate_answers(kernel.get("answer_string"), bank)
            if kernel.get("questionnaire_sha256") != fingerprint(bank) or values != kernel.get("item_responses"):
                parser.error("Agent output does not match the questionnaire or answer string")
        subjects = {str(result["subject_id"]): None}
    else:
        subjects = read_json(args.answers_json)
        if not isinstance(subjects, dict) or not subjects:
            parser.error("--answers-json must contain a nonempty subject-to-answers object")
        for answers in subjects.values():
            validate_answers(answers, bank)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "subjects": len(subjects), "items_per_agent": 60,
                          "agent_roles": ["self", "partner"] if args.agent_result else [args.role]}))
        return 0
    crawler = PersonalityCrawler()
    if args.agent_result:
        updated = attach_reports(result, bank, crawler, args.output_dir)
        path = Path(args.output_dir) / (fingerprint(str(result["subject_id"])) + "_agent.json")
        atomic_json(path, updated)
    else:
        for subject_id, answers in subjects.items():
            crawler.submit(answers, bank, subject_id, args.role, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
