# Initial-action noise visualization

RoboDojo can collect the initial action-noise tensor used by a diffusion or
flow-matching policy and create two shared UMAP/t-SNE views after evaluation.
Collection is enabled with `--action-noise-viz` on `eval`, `client`, `smoke`,
or `benchmark`.

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CKPT> \
  --policy-env <POLICY_ENV> \
  --eval-num 20 \
  --action-noise-viz \
  --noise-viz-method umap \
  --noise-viz-k 5
```

Outputs are written below `eval_result/action_noise/<run-id>/` by default:

- `raw/<task>/*.npz`: one simulator-labelled file per completed rollout;
- `plots/<method>_by_outcome.png`: successful points are deep blue; failed
  points progress from deep blue to deep red as the rollout step increases;
- `plots/<method>_by_task.png`: points use a discrete color per task ID;
- `plots/<method>_coordinates.csv`: coordinates and source metadata;
- `plots/manifest.json`: reduction and highlighted-rollout metadata.

The selected successful and failed rollouts always come from the same task.
Up to `k` inference points are selected at equal time intervals, connected,
and labelled `S1...Sk` and `F1...Fk` in both plots. If the collected data has
no task with both outcomes, plotting still succeeds but omits these markers.

## Policy response contract

The policy owns noise generation, so its XPolicyLab deployment adapter must
return the exact tensor it passed into the diffusion/flow sampler. Add
`initial_noise_actions` beside `actions` in the inference result:

```python
initial_noise = make_initial_noise(...)  # [horizon, action_dim], or more axes
actions = sampler.sample_actions(..., noise=initial_noise)
return {
    "actions": format_action_chunk(actions),
    "initial_noise_actions": np.asarray(initial_noise, dtype=np.float32),
}
```

One visualization point represents one inference/action chunk: RoboDojo
flattens the complete `initial_noise_actions` tensor into one feature vector.
All points in one plot must therefore have the same tensor shape. The optional
field does not affect non-diffusion policies when visualization is disabled.

For action chunks, return one noise tensor for the whole generated chunk, not
one copy per deployed action. RoboDojo associates it with the rollout step at
which the chunk begins execution.

`XPolicyLab/policy/Pi_05` implements this contract directly. Its OpenPI
inference policy creates the noise before invoking the JAX or PyTorch flow
sampler and returns that exact tensor. Base Pi0.5, BCP, and post-trained action
heads expose the noise used for their reference Pi0.5 chunk; adaptive action
chunking exposes only the selected candidate's noise. This is confined to
inference and does not alter the Pi0.5 training loss, training batches,
replay-buffer schema, or checkpoint format used by `scripts/posttrain`.
