# Experiment 09 scripts

## Collect a benchmark run

```bash
python experiments/09_vec3_vec4_padding_costs/scripts/collect_run.py --input experiments/09_vec3_vec4_padding_costs/results/tables/benchmark_results.json
```

This stores timestamped JSON files under `experiments/09_vec3_vec4_padding_costs/runs/<gpu_slug>/`.

## Generate summary artifacts

```bash
python experiments/09_vec3_vec4_padding_costs/scripts/analyze_vec3_vec4_padding_costs.py
```

This writes:
- `experiments/09_vec3_vec4_padding_costs/results/tables/vec3_vec4_padding_summary.csv`
- `experiments/09_vec3_vec4_padding_costs/results/charts/vec3_vec4_padding_median_gpu_ms.svg`

