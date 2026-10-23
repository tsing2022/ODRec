import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def load_config(path, device):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    config['device'] = device
    enabled = sum(bool(config[k]) for k in ('social', 'profile', 'MBTI'))
    channels = [int(v) for v in config['channel'].split(',')]
    if not enabled or sorted(channels) != list(range(enabled)):
        raise ValueError('channel must contain every enabled channel once')
    if config['data']['name'] not in ('SCD', 'MGD'):
        raise ValueError('dataset must be SCD or MGD')
    return config


def load_graph(path, device):
    import scipy.sparse as sp
    import torch
    graph = sp.load_npz(path).tocoo()
    if graph.shape[0] != graph.shape[1] or graph.shape[0] % 2 or not np.isfinite(graph.data).all():
        raise ValueError('graph must be a finite square normalized seeker/candidate role graph')
    n = graph.shape[0] // 2
    indices = torch.from_numpy(np.vstack([graph.row, graph.col]).astype(np.int64))
    values = torch.from_numpy(graph.data.astype(np.float32))
    adj = torch.sparse_coo_tensor(indices, values, graph.shape).coalesce().to(device)
    return SimpleNamespace(torch_adj=adj, user_num=n, item_num=n)


def load_records(path, n):
    with np.load(path, allow_pickle=False) as data:
        users, positives, negatives = [data[k].copy() for k in ('users', 'positives', 'negatives')]
    if users.ndim != 1 or positives.shape != users.shape or negatives.ndim != 2 or len(negatives) != len(users) or not len(users) or negatives.shape[1] < 1:
        raise ValueError('records require aligned users [R], positives [R], negatives [R,M]')
    for values in (users, positives, negatives):
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= n).any():
            raise ValueError('record IDs must be integer row indices within the graph')
    if (negatives == positives[:, None]).any():
        raise ValueError('positive targets cannot also occur as negatives')
    return users, positives, negatives


def load_checkpoint(model, path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('checkpoint must contain a state_dict mapping')
    state = payload.get('state_dict', payload.get('model_state_dict', payload))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith('denoiseNet.')}
    model.load_state_dict(state, strict=True)
    model.final_embeds = None
    model.eval()
    return model
