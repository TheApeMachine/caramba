# Timeline

This file shows the exact timeline how things ran.

## Exploratory Runs

These runs were done to select the best decoupled model.

It turned out that the sem16 geo32 model underperformed when benchmarked, and the sem8 geo32 v40 model was the best.

GPU: A100 80GB
LAYERS: 22
STEPS: 10k

manifest: ./config/presets/dba_paper_rerun.yml

make benchmark10k

━━━ Multi-Checkpoint Compare • 3 models • device=mps dtype=torch.float16 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Loading: baseline
ℹ   Config: TransformerModel
ℹ   Loading: research/dba/checkpoints/a100_fw1b_l22_baseline_s1337_10k.pt
ℹ   Converted 22 separate Q/K/V projections to fused qkv_proj
ℹ   Marked as baseline
  baseline: 1336.5023M params
── Loading: sem16
ℹ   Config: TransformerModel
ℹ   Loading: research/dba/checkpoints/a100_fw1b_l22_decoupled_s1337_10k.pt
  sem16: 1244.2276M params
── Loading: sem8
ℹ   Config: TransformerModel
ℹ   Loading: research/dba/checkpoints/a100_fw1b_l22_decoupled_sem8geo32v40_s1337_10k.pt
  sem8: 1198.0902M params
✓ Loaded 3 models

ARTIFACTS: ./research/dba/benchmark10k/

## Full Runs

These runs directly compare the sem8 geo32 v40 model with the baseline.

GPU: A100 80GB
LAYERS: 22
STEPS: 100k

manifest: ./config/presets/dba_paper_rerun.yml

While the ablations were running, we gather initial benchmarks, some of which are reported in the paper.
Most notably there is a mention of 117 behavior probes. This is based on the original in-house benchmark suite.
They showed that the sem8 geo32 model was able to match, if not beat the baseline model.
We then decided to expand the benchmark suite to include 18 categories, each with 30 probes.

### Original

━━━ Behavior Benchmark (Multi): behavior_sanity ━━━
Models: baseline, sem8 | Baseline: baseline

  [10/117] fewshot_simple_suffix
  [20/117] distractor_implicit_numbers
  [30/117] reason_double_negation
  [40/117] math_sub_two_digit
  [50/117] seq_geometric_x3
  [60/117] fact_months_year
  [70/117] format_json_nested
  [80/117] context_multiple_facts
  [90/117] edge_single_digit
  [100/117] attention_recent_vs_distant
  [110/117] instruct_uppercase

━━━ Weighted Scoring Summary ━━━
Difficulty distribution: easy=1 medium=37 hard=79
  baseline: hard=0.9% │ soft=32.5% │ weighted=12.2%
  sem8: hard=0.0% │ soft=39.3% │ weighted=17.6%

  behavior_multi_weighted: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_multi_weighted.json
  behavior_multi_markdown: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_multi_detailed.md
  behavior_multi_detailed_log: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_multi_detailed.log
  behavior_multi_csv: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_multi_weighted.csv
  behavior_multi_latex: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_multi_table.tex
  viz_weighted_scores_comparison: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_weighted_scores_comparison.png
  viz_difficulty_breakdown: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_difficulty_breakdown.png
  viz_match_type_distribution: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_match_type_distribution.png
  viz_weighted_ranking_table: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_weighted_ranking_table.png
  viz_weighted_ranking_table_latex: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavior_weighted_ranking_table.tex
  baseline_hard: 0.8547%
  baseline_soft: 32.4786%
  baseline_weighted: 12.1795%
  sem8_hard: 0.0000%
  sem8_soft: 39.3162%
  sem8_weighted: 17.6282%

### V2 Behavioral Benchmark

── behavioral_v2 (BenchmarkType.BEHAVIORAL_V2)
ℹ Generating test suite (seed=1337)...
ℹ Running 600 behavioral tests on 2 models...
ℹ   [10/600] copy_simple_lower_echo_start_20110
ℹ   [20/600] copy_simple_alpha_copy_start_31225
ℹ   [30/600] copy_simple_upper_repeat_middle_40170
ℹ   [40/600] fewshot_ing_suffix_certain
ℹ   [50/600] fewshot_symbol_f
ℹ   [60/600] fewshot_relation_dog
ℹ   [70/600] noisy_easy_start_45206
ℹ   [80/600] noisy_easy_middle_71301
ℹ   [90/600] noisy_easy_end_68528
ℹ   [100/600] reason_transitive_4step
ℹ   [110/600] reason_conditional_3
ℹ   [120/600] reason_spatial_1
ℹ   [130/600] math_sub_55_36
ℹ   [140/600] math_chain_3
ℹ   [150/600] math_compare_2
ℹ   [160/600] seq_geom_4_2
ℹ   [170/600] seq_fib_3_3
ℹ   [180/600] seq_square_3
ℹ   [190/600] fact_science_8
ℹ   [200/600] fact_geo_Mariana_Tr
ℹ   [210/600] fact_lang_Icelandic
ℹ   [220/600] semantic_ant_generous
ℹ   [230/600] semantic_cat_vegetable
ℹ   [240/600] semantic_assoc_book
ℹ   [250/600] format_table_1
ℹ   [260/600] format_bracket_3
ℹ   [270/600] format_xml_3
ℹ   [280/600] context_late_laptop_desk
ℹ   [290/600] context_update_total_2
ℹ   [300/600] context_pattern_2
ℹ   [310/600] robust_case_cold
ℹ   [320/600] robust_punct_24
ℹ   [330/600] robust_order_2
ℹ   [340/600] edge_single_2
ℹ   [350/600] edge_negative_1
ℹ   [360/600] edge_identity_3
ℹ   [370/600] attn_noisy_4801
ℹ   [380/600] attn_binding_1
ℹ   [390/600] attn_interfere_2
ℹ   [400/600] pattern_count_9
ℹ   [410/600] pattern_reverse_3
ℹ   [420/600] pattern_extract_2
ℹ   [430/600] consist_copy_XYZ
ℹ   [440/600] consist_order_1
ℹ   [450/600] consist_style_Charlie
ℹ   [460/600] inject_hard_start_11609
ℹ   [470/600] poison_medium_36094
ℹ   [480/600] format_hijack_hard_97483
ℹ   [490/600] dread_forced_repeat_easy_25839
ℹ   [500/600] topic_switch_easy_92099
ℹ   [510/600] escalate_easy_49236
ℹ   [520/600] paired_hijack_frequency_easy_s0_26750
ℹ   [530/600] paired_hijack_position_easy_s1_11043
ℹ   [540/600] paired_sink_easy_s2_44180
ℹ   [550/600] binding_two_easy_end_93251
ℹ   [560/600] binding_temporal_easy_after_53083
ℹ   [570/600] binding_three_medium_middle_66396
ℹ   [580/600] multihop_comp_easy_45686
ℹ   [590/600] multihop_comp_medium_85927
ℹ   [600/600] multihop_comp_hard_1941
  baseline_exact: 0.6122%
  sem8_exact: 1.2245%
  baseline_hard: 1.0204%
  baseline_soft: 32.0408%
  baseline_weighted: 12.0031%
  sem8_hard: 0.8163%
  sem8_soft: 29.7959%
  sem8_weighted: 13.0734%

ℹ Difficulty distribution (by baseline): easy=5 medium=152 hard=333
  behavioral_v2_multi_summary: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_multi_summary.json
  behavioral_v2_multi_detailed: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_multi_detailed.json
  behavioral_v2_weighted_scores: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_weighted_scores.json
  behavioral_v2_weighted_csv: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_weighted_summary.csv
  behavioral_v2_latex: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_table.tex
  behavioral_v2_markdown: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_detailed.md
  viz_weighted_scores_comparison: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_weighted_scores_comparison.png
  viz_difficulty_breakdown: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_difficulty_breakdown.png
  viz_match_type_distribution: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_match_type_distribution.png
  viz_weighted_ranking_table: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_weighted_ranking_table.png
  viz_weighted_ranking_table_latex: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/behavioral_v2_weighted_ranking_table.tex
  behavioral_v2_multi_log: research/dba/benchmark100k/dba_l22_100k_benchmark/multi_checkpoint_compare/20260118_061513/multi_behavioral_v2_log.txt
  baseline: 0.6122% exact
  sem8: 1.2245% exact

## Ablations

These runs were to satisfy the reviewer request to show that the results were not due to other factors.
These also revealed a pretty big win for the gated DBA variant.

GPU: A100 40GB
LAYERS: 12
STEPS: 10k

manifest: ./config/presets/dba_paper_local.yml

## Gated Full Run

Given the success of the gated DBA variant in the ablations, we are running a full-length 100k apples-to-apples run to compare it with the baseline.

GPU: A100 80GB
LAYERS: 22
STEPS: 100k

manifest: ./config/presets/dba_paper_gated.yml

## NOTES

We should reduce the amount of individual figures we're creating and just make one big aggregate, advanced figure for each run type (e.g. one figure for the 10k runs, one figure for the 100k runs, etc.) that shows as much interesting data visualized as possible.

We should definitely include any of the attention heatmaps, attention last layer head, and attention mass visualizations we're creating during the behavioral benchmarks.

#0DB6AE
#F68512
#FFC600
#DE3D82