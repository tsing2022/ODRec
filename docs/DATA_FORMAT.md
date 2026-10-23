# Evaluation inputs

## Scores

`scores.npz` contains a finite floating-point array `scores` of shape `[R,1+M]`. The first column is the positive target. Metrics average over R evaluation records, with no prior user averaging.

## Checkpoint and graph

A checkpoint contains the ODRec state dictionary, either directly or under `state_dict` or `model_state_dict`. It is loaded strictly with PyTorch's weights-only reader. Auxiliary `denoiseNet.*` entries are omitted from the inference state.

`memory_graph.npz` is a SciPy sparse normalized adjacency matrix with shape `[2N,2N]`, ordered as N seekers followed by N candidate role copies. Supply the graph associated with the checkpoint. It uses the original scorer's symmetric degree normalization; the evaluator does not reconstruct or modify it.

`records.npz` contains integer arrays `users [R]`, `positives [R]` and `negatives [R,M]`. All IDs are zero-based account indices shared by the graph and feature rows. M can be 19 or 99.

## Feature files

Set `data.name` to SCD or MGD in the evaluation configuration. Each feature matrix has N rows and 1,536 columns from `text-embedding-3-small`.

SCD:

```text
contextual/user_profile_pure_emb.npy
contextual/gpt_req_embedding.npy
mental/converted_user_mbti_analyse_scd_gpt_emb.npy
mental/converted_soul_mbti_analyse_emb.npy
```

MGD:

```text
contextual/user_profile_emb.npy
contextual/gpt_req_emb.npy
mental/converted_user_mbti_mgd_analyse_emb.npy
mental/converted_soul_mbti_mgd_analyse_emb.npy
```

The order is self persona, partner persona, self kernel and partner kernel. The configuration must match the checkpoint's dimensions, enabled components and gate settings. The supplied default describes the complete three-channel model.
