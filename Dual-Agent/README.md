# Cognitive agents

`CognitiveAgent` constructs two profile-conditioned agents:

- Self-Agent: preserves observed profile attributes and answers the questionnaire from the original profile.
- Inner Twin Soul: proposes a partner persona and answers the same questionnaire from that persona.

## Processing

1. Construct the persona and review its attributes and quoted evidence.
2. Ask one questionnaire item at a time.
3. Validate the answer schema, response range, confidence, and quotations.
4. Have the Reflector review grounding, answer direction, strength, and consistency.
5. Return feedback for up to three rounds and store accepted responses.

Each call receives the complete subject profile. The Reflector also receives the accepted-answer history. Memory is separated by user and role and checked against profile, prompt, questionnaire, and model-setting hashes.

`questionnaire.json` contains 60 items from 16Personalities. The six options run from strongly agree (1) to strongly disagree (6). Unsupported items retain a null answer. A response string is exported when all items have supported answers.

## Usage

From this directory:

```bash
python cognitive_agent.py --profiles example_profiles.json
```

For generation, configure `OPENAI_API_KEY` and select a model:

```bash
python cognitive_agent.py --profiles /private/profiles.json \
  --memory-dir /private/odrec/memory --output-dir /private/odrec/output \
  --model gpt-3.5-turbo-0125 --execute
```

`OPENAI_BASE_URL` selects an HTTPS OpenAI-compatible endpoint. The agent runner uses the Python standard library; the optional crawler uses Requests.

Outputs include reviewed personas, item responses, and four semantic texts. Keep profiles, memory, and generated outputs outside the public repository.

## Tests

```bash
python -m unittest -v test_questionnaire_agent test_cognitive_agent
```

## Questionnaire reports

Each item answer is an integer from 1 to 6, or null when the profile does not support an answer. Completed numeric responses are exported in question order as both a 60-element `item_responses` list and a 60-character `answer_string`. The crawler accepts only 60 digits in `1..6` and does not fill null answers.

Install the crawler dependency in this directory:

```bash
python -m pip install -r requirements.txt
```

Generate both agents and retrieve their questionnaire reports:

```bash
python cognitive_agent.py --profiles /path/to/profiles.json \
  --execute --crawl --report-dir personality_reports
```

The crawler uses the questionnaire order in `questionnaire.json` and maps `1,2,3,4,5,6` to `-3,-2,-1,1,2,3`. It submits the 60 question/answer pairs to `/test-results`, then retrieves the report from `/api/type-descriptions/results` in the same session. Reports retain their answer and questionnaire hashes for resumption. The `--crawl` path writes the returned report text into `semantic_texts.self_kernel` and `semantic_texts.partner_kernel`; the numeric answers remain in `kernels`.

Retrieve reports from an existing completed agent result:

```bash
python personality_crawler.py --agent-result agent_output/subject.json \
  --output-dir personality_reports --execute
```

The crawler also accepts a JSON object mapping subject IDs to 60-digit strings through `--answers-json`. Without `--execute`, this command validates the answers and prints its plan. Report text includes the service's personality traits, career, personal-growth and relationship sections.
