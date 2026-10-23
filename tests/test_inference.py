import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odrec_eval.model import ODRec
from odrec_eval.inputs import load_checkpoint


class InferenceTests(unittest.TestCase):
    def model(self):
        config = {'data': {'name': 'SCD'}, 'model': {'embedding_size': 4, 'layer_num': 2},
                  'gpt': 2, 'mbti': 2, 'social': True, 'profile': True, 'MBTI': True,
                  'relu': True, 'w_1': 'kexuexi', 'channel': '0,1,2', 'device': 'cpu'}
        graph = SimpleNamespace(torch_adj=torch.eye(18).to_sparse(), user_num=9, item_num=9)
        arrays = np.random.RandomState(421).normal(size=(9, 6)).astype('float32')
        with patch('odrec_eval.model.np.load', return_value=arrays):
            return ODRec(graph, config, 'unused')

    def test_fixed_model_scores_candidate_blocks(self):
        model = self.model()
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with torch.no_grad():
            scores, tensor = model.predict_candidate_block(
                torch.tensor([0, 2]), torch.tensor([[1, 3, 5], [0, 4, 6]]), return_tensor=True)
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertEqual(tuple(tensor.shape), (2, 3, 3))
        self.assertTrue(torch.isfinite(scores).all())
        weights = model.adaptive_channel_weights(tensor.mean(1), torch.tensor([0, 2]))
        np.testing.assert_allclose(weights.sum(-1).numpy(), np.ones(2), atol=1e-6)

    def test_checkpoint_reader_uses_weights_only_and_strict_shapes(self):
        model = self.model()
        state = copy.deepcopy(model.state_dict())
        state['denoiseNet.unused'] = torch.zeros(1)
        with patch('torch.load', return_value={'state_dict': state}) as read:
            loaded = load_checkpoint(model, 'external.pth')
        read.assert_called_once_with('external.pth', map_location='cpu', weights_only=True)
        self.assertIs(loaded, model)
        self.assertFalse(model.training)
        del state['w']
        with patch('torch.load', return_value=state):
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, 'incompatible.pth')


if __name__ == '__main__':
    unittest.main()
