import numpy as np


def evaluate_scores(scores, ks=(1, 5, 10)):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not scores.shape[0] or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite [records, 1 + negatives] array')
    if not ks or any(type(k) is not int or k < 1 or k > scores.shape[1] for k in ks):
        raise ValueError('Every k must be between 1 and the candidate count')
    order = np.argsort(-scores, axis=1)
    ranks = np.argmax(order == 0, axis=1) + 1
    result = {'records': len(scores), 'candidates_per_record': scores.shape[1]}
    for k in ks:
        hit = ranks <= k
        result['HR@' + str(k)] = float(hit.mean())
        result['NDCG@' + str(k)] = float(np.mean(hit / np.log2(ranks + 1)))
    from sklearn.metrics import roc_auc_score, average_precision_score
    negatives = scores[:, 1:].reshape(-1)
    positives = scores[:, 0].tolist() * (scores.shape[1] - 1)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    pooled = np.concatenate([positives, negatives])
    result['AUC'] = float(roc_auc_score(labels, pooled))
    result['AP'] = float(average_precision_score(labels, pooled))
    return result
