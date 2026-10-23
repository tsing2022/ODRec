import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odrec_eval.metrics import evaluate_scores


class EvaluationTests(unittest.TestCase):
    def test_known_positive_ranks(self):
        scores = np.zeros((2, 20)); scores[0, 0] = 3
        scores[1, :3] = [1, 3, 2]
        result = evaluate_scores(scores)
        self.assertEqual(result['HR@1'], 0.5)
        self.assertEqual(result['HR@5'], 1.0)
        self.assertEqual(result['NDCG@5'], 0.75)

    def test_nonfinite_and_invalid_shapes_rejected(self):
        for scores in [np.zeros((0, 20)), np.array([[np.nan, 0]]), np.zeros(20)]:
            with self.assertRaises(ValueError): evaluate_scores(scores)

    def test_saved_score_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); np.savez(p/'scores.npz', scores=np.arange(40).reshape(2, 20))
            result = subprocess.run([sys.executable, str(ROOT/'evaluate.py'), '--scores', str(p/'scores.npz'), '--output', str(p/'metrics.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['records'], 2)
            self.assertEqual(json.loads((p/'metrics.json').read_text())['records'], 2)


if __name__ == '__main__': unittest.main()
