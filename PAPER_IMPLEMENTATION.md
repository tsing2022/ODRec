# Inference components

`odrec_eval/model.py` retains the ODRec forward scorer and parameter names for loading a fixed state dictionary.

- Collaborative memory propagates seeker and candidate role embeddings through the supplied normalized graph.
- SP and PK each share a `1536 -> 16 -> 8` projection between their self and partner inputs.
- External and internal compatibility pair seeker-partner and candidate-self vectors.
- `build_score_tensor` constructs the requested compatibility blocks.
- `decision_arbiter` pools each score block and normalizes the mixed sigmoid gates across channels before fusion.
- `score_compatibility_blocks` scores the positive and negative blocks separately, preserving the existing sampled evaluator's pooling scopes.

`odrec_eval/metrics.py` computes record-level HR and NDCG with the positive in column zero, using NumPy's default descending argsort. Its pooled AUC and AP repeat each positive score once per negative, following the existing evaluation protocol. No training loop or objective is included.

The self SP input represents the observed profile; the partner SP and two PK inputs are externally prepared representations. Questionnaire and prompt materials describe the Analyst/Reflector process.
