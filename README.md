# ODRec

Agent construction and evaluation for ODRec directed follow recommendation.

## Structure

- `Dual-Agent/`: Self-Agent and Inner Twin Soul construction with Analyst/Reflector review.
- `odrec_eval/model.py`: collaborative memory, semantic projections, compatibility scoring and decision arbiter.
- `evaluate.py`: checkpoint scoring and metric evaluation.
- `config/evaluation.json`: inference settings.

## Setup

Use Python 3.9 or later and install `requirements.txt`.

## Agents

```bash
cd Dual-Agent
python cognitive_agent.py --profiles example_profiles.json
```

For generation, configure `OPENAI_API_KEY` and add `--execute`. Add `--crawl` to retrieve questionnaire reports from each agent's 60-digit answer string and use the report text as its kernel representation. See [Dual-Agent/README.md](Dual-Agent/README.md).

## Evaluate saved scores

```bash
python evaluate.py --scores /path/to/scores.npz --output metrics.json
```

The `scores` array has shape `[records, 1 + negatives]`. Column zero is the observed positive target. Each row is one evaluation record. Candidate counts of 20 and 100 correspond to 19 and 99 negatives.

## Evaluate a checkpoint

```bash
python evaluate.py --checkpoint /path/to/odrec.pth \
  --graph /path/to/memory_graph.npz --records /path/to/records.npz \
  --feature-dir /path/to/datasets/SCD --config config/evaluation.json
```

See [the input specification](docs/DATA_FORMAT.md).

## Checks

```bash
python -m unittest discover -s tests
python -m unittest discover -s Dual-Agent -p 'test_*.py'
```

See [PAPER_IMPLEMENTATION.md](PAPER_IMPLEMENTATION.md) for the method components and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
