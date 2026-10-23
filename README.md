# ODRec

Evaluation code for ODRec directed follow recommendation.

## Setup

Use Python 3.9 or later and install `requirements.txt`.

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

Checkpoint evaluation uses externally supplied model weights, a normalized collaborative role graph, candidate records and four semantic feature matrices. These inputs are not bundled. See [the input specification](docs/DATA_FORMAT.md).

The public release contains inference and metric evaluation. Training, optimization and agent-generation executables are not included. Questionnaire items and prompts are supplied in `Dual-Agent/` as method materials.

## Checks

```bash
python -m unittest discover -s tests
```

The checks use synthetic temporary inputs. They do not reproduce the paper's dataset results.

See [PAPER_IMPLEMENTATION.md](PAPER_IMPLEMENTATION.md) for the inference components and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
