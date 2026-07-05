# 07 Multi-Generation-Seed Evaluation Protocol

Status: active evaluation-protocol plan  
Decision date: 2026-07-05  
Scope: D-OPSD / few-step adaptation method comparison  
Non-scope: this is not RC-MC B2/B3, not Step 5, and not a PTRW method continuation plan.

## 0. Decision: close PTRW as a method-development branch

PTRW is closed as a method-development branch.

PTRW should be retained only as weak development evidence / archival diagnostic evidence. It should not receive additional training seeds under the current single-generation-seed evaluation setup.

Reason:

- The observed image-metric deltas are too small to interpret under a single generation seed.
- Additional PTRW training seeds would not fix the main evaluation problem.
- Before judging any method with small deltas, the project needs an evaluation protocol that estimates generation variability.

Allowed future use of PTRW:

- archive its existing results as development evidence;
- optionally re-score frozen PTRW checkpoints under the multi-generation-seed protocol if they are needed as a historical reference;
- do not use PTRW as a primary method candidate unless a separate proposal reopens it with a new objective and a new evaluation gate.

## 1. Motivation

The current evaluation path supports fixed-seed image generation, for example through a single `--base-seed` in the evaluation generation command. That is acceptable for smoke tests and qualitative checks, but it is not enough for method claims when the expected image-metric delta is on the order of `1e-3`.

The immediate problem is not the absence of another image-quality metric. The immediate problem is that single-seed point estimates cannot separate method signal from generation-seed variability.

Therefore, the next evaluation change should add uncertainty and reliability indicators around the existing metrics, rather than add another standalone image metric.

## 2. Evaluation unit

The atomic evaluation row should be:

```text
method_id × checkpoint_id × case_id × prompt_id × generation_seed
```

Each generated image should preserve enough metadata to make paired comparisons possible:

```text
method_id
checkpoint_id
case_id
prompt_id
generation_seed
image_path
num_inference_steps
lora_path / adapter_id
base_model
metric values
```

A single metric average from one generation seed must not be treated as a primary method result.

## 3. Protocol

For each method/checkpoint pair:

1. freeze the evaluation case set;
2. freeze prompt generation outputs;
3. freeze metric code and metric model versions;
4. generate `K` samples per case using multiple generation seeds;
5. use the same generation seeds across methods whenever possible;
6. compute paired deltas at `case_id × prompt_id × generation_seed` granularity;
7. aggregate with confidence intervals and seed-variability diagnostics.

Recommended seed counts:

```text
pilot protocol:  K = 4 generation seeds
formal protocol: K = 8 or 16 generation seeds, depending on pilot variance
```

Generation seeds should be addressed before adding more training seeds. Training-seed reruns are justified only after the generation-seed noise floor is known.

## 4. Existing metrics to retain

Keep the current image/task metric families, including available DINO/LPIPS/CLIP/VLM-judge style metrics.

The protocol does not require a new perceptual or semantic image metric. New additions should be uncertainty and reliability indicators computed from the existing metrics.

Metric direction must be normalized before aggregation:

```text
higher_is_better: CLIP-like, DINO-like, reward/judge-like metrics
lower_is_better: LPIPS-like distance metrics
```

For lower-is-better metrics, convert deltas to the common signed convention:

```text
signed_delta = baseline_metric - method_metric
```

For higher-is-better metrics:

```text
signed_delta = method_metric - baseline_metric
```

After sign normalization, positive signed deltas mean the method is better.

## 5. New reliability indicators

These are the new indicators to add to summaries. They are meta-evaluation metrics, not new image-quality metrics.

| Indicator | Definition | Purpose |
|---|---|---|
| `metric_mean` | Mean over cases, prompts, and generation seeds | Average method score |
| `metric_seed_std` | Standard deviation across generation seeds after case/prompt aggregation | Direct estimate of generation variability |
| `metric_ci_low`, `metric_ci_high` | Bootstrap or t-interval confidence interval for the metric mean | Uncertainty of absolute score |
| `paired_delta_mean` | Mean signed paired delta against baseline | Primary direction and magnitude |
| `paired_delta_ci_low`, `paired_delta_ci_high` | Confidence interval for paired delta | Whether the method delta is distinguishable from noise |
| `paired_win_rate` | Fraction of paired rows with signed delta `> 0` | Case/seed-level consistency |
| `seed_noise_floor` | Empirical seed-level fluctuation for the same method/metric | Minimum practical detectable effect |
| `noise_floor_ratio` | `abs(paired_delta_mean) / (seed_noise_floor + eps)` | Whether observed delta exceeds seed noise |
| `case_flip_rate` | Fraction of cases whose win/loss sign changes across generation seeds | Stability of per-case conclusion |
| `valid_case_count` | Number of cases with valid metric values | Coverage check |
| `cost_normalized_delta` | `paired_delta_mean` per GPU-hour, teacher call, or generation cost unit | Cost-aware comparison |

Default interpretation:

```text
noise_floor_ratio < 1:   indistinguishable from generation variability
1 <= ratio < 2:          weak evidence only
ratio >= 2:              potentially meaningful, pending CI and sparsity checks
```

The primary comparison should use `paired_delta_mean`, `paired_delta_ci_*`, `paired_win_rate`, and `noise_floor_ratio` together. Do not make method claims from `metric_mean` alone.

## 6. Primary decision rule

A method can be called a clear positive result only if all of the following hold:

1. `paired_delta_mean > 0` under the signed convention;
2. `paired_delta_ci_low > 0`, or an explicitly approved equivalent uncertainty test is positive;
3. `noise_floor_ratio >= 2` for the primary metric or aggregate metric;
4. `paired_win_rate` shows directional consistency rather than a few outlier wins;
5. VLM/Judge-style metrics are not supported by only a tiny number of positive cases;
6. the method remains acceptable under cost-normalized comparison.

Use these labels in reports:

```text
clear_positive
weak_positive
inconclusive
negative
```

Any result with a confidence interval crossing zero and `noise_floor_ratio < 1` should be labeled `inconclusive`, even if the point estimate is positive.

## 7. Aggregate uncertainty metric

If an aggregate score is needed, define it as an uncertainty-aware signed aggregate, not as a single-seed mean.

Recommended minimal aggregate:

```text
aggregate_signed_delta = mean_zscore_over_metric_families(signed_paired_delta)
```

Where each metric family is first normalized by its empirical seed noise floor:

```text
z_delta_metric = signed_paired_delta_metric / (seed_noise_floor_metric + eps)
```

The aggregate report must include:

```text
aggregate_signed_delta_mean
aggregate_signed_delta_ci_low
aggregate_signed_delta_ci_high
aggregate_paired_win_rate
```

This aggregate is a decision aid, not a replacement for per-metric reporting.

## 8. VLM/Judge case-level sparsity report

VLM/Judge-style metrics must not be reported only as a scalar average. They need a case-level sparsity table.

Required fields:

```text
vlmj_valid_case_count
vlmj_positive_case_count
vlmj_positive_rate
vlmj_gain_case_count
vlmj_loss_case_count
vlmj_unchanged_case_count
vlmj_seed_flip_case_count
vlmj_all_methods_fail_case_count
vlmj_single_method_win_case_count
```

Required interpretation:

- If the positive delta comes from only one or two cases, the result is weak.
- If the winning cases flip across generation seeds, the result is seed-sensitive.
- If most cases are invalid or all methods fail, the scalar average should not be used as primary evidence.

## 9. Recent rerun plan

Previous single-generation-seed experiments should be treated as development diagnostics until rerun under this protocol.

Near-term plan:

### P0. Close PTRW

- Mark PTRW as closed for method development.
- Preserve existing PTRW outputs as archival weak evidence only.
- Do not allocate new PTRW training seeds.

### P1. Implement multi-seed evaluation outputs

Add or adapt evaluation logging so that each row records:

```text
method_id
checkpoint_id
case_id
prompt_id
generation_seed
metric values
```

Expected output files:

```text
multi_seed_samples.jsonl
metric_summary.csv
paired_delta_summary.csv
vlmj_case_sparsity.csv
run_manifest.json
```

### P2. Pilot rerun for generation variability

Rerun the evaluation generation step with `K = 4` generation seeds for:

- the same-step raw baseline / public frontier used for comparison;
- the strongest non-PTRW candidate, if one is still under consideration;
- frozen PTRW checkpoints only if a historical reference is needed.

The goal is to estimate `seed_noise_floor`, not to relaunch PTRW.

### P3. Formal rerun only where justified

Move to `K = 8` or `K = 16` only for methods whose pilot results show plausible signal:

```text
paired_delta_mean > 0
and noise_floor_ratio >= 1 in pilot
```

If a method has `noise_floor_ratio < 1` in the pilot, keep it as inconclusive unless there is a strong external reason to rerun.

### P4. Reclassify earlier tables

Any earlier table based on a single generation seed should be relabeled as one of:

```text
smoke_test
qualitative_check
development_diagnostic
historical_archival_result
```

It should not be used as a primary method comparison until rerun with multi-generation-seed uncertainty.

## 10. Implementation notes

A minimal analysis script should compute paired deltas from a table with columns:

```text
method_id
baseline_method_id
checkpoint_id
case_id
prompt_id
generation_seed
metric_name
metric_value
baseline_metric_value
metric_direction
```

For each metric, report:

```text
metric_mean
metric_seed_std
metric_ci_low
metric_ci_high
paired_delta_mean
paired_delta_ci_low
paired_delta_ci_high
paired_win_rate
seed_noise_floor
noise_floor_ratio
case_flip_rate
valid_case_count
```

Use bootstrap over `case_id` as the default conservative option. If cases contain multiple prompts, resample cases first and preserve prompts/seeds inside the sampled case to avoid overestimating effective sample size.

## 11. Acceptance criteria

This protocol is considered active only when:

1. the evaluation run manifest records generation seeds explicitly;
2. metric summaries report confidence intervals;
3. paired baseline deltas are available;
4. VLM/Judge-style metrics include case-level sparsity reporting;
5. previous single-seed results are no longer used as primary method evidence;
6. at least one baseline pilot estimates the generation-seed noise floor.

Until these are satisfied, small image-metric deltas should be treated as inconclusive.
