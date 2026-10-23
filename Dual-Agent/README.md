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

`OPENAI_BASE_URL` selects an HTTPS OpenAI-compatible endpoint. The runner uses the Python standard library.

Outputs include reviewed personas, item responses, and four semantic texts. Keep profiles, memory, and generated outputs outside the public repository.

## Tests

```bash
python -m unittest -v test_questionnaire_agent test_cognitive_agent
```
