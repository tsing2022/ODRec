import argparse
import json
from pathlib import Path
import numpy as np
from odrec_eval.metrics import evaluate_scores


def predict(args):
    import torch
    from odrec_eval.inputs import load_config, load_graph, load_records, load_checkpoint
    from odrec_eval.model import ODRec
    config = load_config(args.config, args.device)
    graph = load_graph(args.graph, args.device)
    model = load_checkpoint(ODRec(graph, config, args.feature_dir), args.checkpoint).to(args.device)
    users, positives, negatives = load_records(args.records, graph.user_num)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(users), args.batch_size):
            arrays = [a[start:start + args.batch_size] for a in (users, positives, negatives)]
            batch = [torch.from_numpy(a.astype(np.int64)).to(args.device) for a in arrays]
            positive, negative = model.score_compatibility_blocks(*batch)
            pieces.append(torch.cat([positive, negative], dim=1).cpu().numpy())
    return np.concatenate(pieces)


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved ODRec scores or a fixed checkpoint')
    parser.add_argument('--scores', help='NPZ containing scores [R,1+M], positive in column zero')
    parser.add_argument('--checkpoint')
    parser.add_argument('--graph', help='Normalized collaborative role graph in SciPy NPZ format')
    parser.add_argument('--records', help='NPZ containing users, positives and negatives')
    parser.add_argument('--feature-dir')
    parser.add_argument('--config', default='config/evaluation.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--output')
    args = parser.parse_args()
    required = [args.checkpoint, args.graph, args.records, args.feature_dir]
    if args.scores and any(required):
        parser.error('Choose saved scores or checkpoint inputs')
    if not args.scores and not all(required):
        parser.error('Supply --scores, or all of --checkpoint --graph --records --feature-dir')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.scores:
        with np.load(args.scores, allow_pickle=False) as data:
            scores = data['scores'].copy()
    else:
        scores = predict(args)
    result = evaluate_scores(scores, tuple(args.k))
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
