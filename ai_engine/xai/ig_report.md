Checkpoint train_count: 1176 | baseline: -5910.58039908
# Integrated Gradients for the PPO join-order policy
Query: SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1990 LIMIT 10
Fixed random seed: 20261026
[Vectorizer] Fetched stats for 'title': rows=2,562,145  pages=36,008  cols=12  indexes=1  pk=1  fk=0  size=295.0MB  growth_rate=0.0000
[Vectorizer] Fetched stats for 'cast_info': rows=36,203,904  pages=252,720  cols=7  indexes=1  pk=1  fk=0  size=2070.3MB  growth_rate=0.0000
[Vectorizer] Fetched stats for 'name': rows=4,170,977  pages=55,656  cols=9  indexes=1  pk=1  fk=0  size=455.9MB  growth_rate=0.0000
[Vectorizer] Built state tensor: shape=[3, 11]
[Vectorizer] Fetched stats for 'movie_keyword': rows=4,523,930  pages=24,480  cols=3  indexes=1  pk=1  fk=0  size=200.5MB  growth_rate=0.0000
[Vectorizer] Fetched stats for 'movie_companies': rows=2,609,129  pages=18,824  cols=5  indexes=1  pk=1  fk=0  size=154.2MB  growth_rate=0.0000
[Vectorizer] Fetched stats for 'keyword': rows=134,170  pages=1,008  cols=3  indexes=1  pk=1  fk=0  size=8.3MB  growth_rate=0.0000
[Vectorizer] Fetched stats for 'company_name': rows=234,997  pages=3,040  cols=7  indexes=1  pk=1  fk=0  size=24.9MB  growth_rate=0.0000
[Vectorizer] Built state tensor: shape=[7, 11]
Saved query and seven-table states to: D:\College\TY SEM-I\EDI\ai query optimizer\ai_engine\xai\sample_state.pt
State source: live PostgreSQL via build_state_tensor

## Seven-table mean feature vector
| Feature | Mean |
|---|---:|
| log_row_count | 0.82981884 |
| log_page_count | 0.77869469 |
| log_col_count | 0.63771516 |
| log_index_count | 0.28906485 |
| has_primary_key | 1.00000000 |
| has_foreign_key | 0.00000000 |
| log_table_size_bytes | 0.87022316 |
| avg_row_width_norm | 0.56831795 |
| row_density | 0.23519094 |
| is_in_query | 1.00000000 |
| row_growth_rate | 0.00000000 |
## Feature variance across the seven IMDb tables
| Feature | Population variance |
|---|---:|
| log_row_count | 0.0100746592507 |
| log_page_count | 0.0182262584567 |
| log_col_count | 0.018433811143 |
| log_index_count | 0 |
| has_primary_key | 0 |
| has_foreign_key | 0 |
| log_table_size_bytes | 0.00619274238124 |
| avg_row_width_norm | 0.00214129523374 |
| row_density | 0.00654671434313 |
| is_in_query | 0 |
| row_growth_rate | 0 |

Observed constant features: log_index_count, has_primary_key, has_foreign_key, is_in_query, row_growth_rate
Expected constants to check: has_primary_key, has_foreign_key, log_index_count (index_count), is_in_query, row_growth_rate.
With the mean baseline, a constant feature has attribution 0 by construction. With the zero baseline, it can show nonzero attribution purely because its input differs from 0.
Varying features used by the second faithfulness ranking: log_row_count, log_page_count, log_col_count, log_table_size_bytes, avg_row_width_norm, row_density
Constant features detected from these states: log_index_count, has_primary_key, has_foreign_key, is_in_query, row_growth_rate

## Connected-prefix order probabilities
Skipped t → n → ci: ValueError: 'n' is not in list
Skipped n → t → ci: ValueError: 't' is not in list
- [t → ci → n]: 0.024471670389
- [ci → t → n]: 0.311204463243
- [ci → n → t]: 0.615892887115
- [n → ci → t]: 0.048430934548
Probability sum: 0.999999955297
Sum-to-1 check: PASS (tolerance 1e-5)
Most probable order: ci → n → t (π=0.615892887115)
Policy would sample most often: ci → n → t (highest exact sequence probability; π=0.615892887115)
Order explanation (conditional policy choices): P(ci next)=0.92709744 × P(n next)=0.66432381 × P(t next)=1.00000000

Attention weights for the input state (comparison for both baselines):
| Table | Alias | Attention weight |
|---|---|---:|
| title | t | 0.02447166 |
| cast_info | ci | 0.92709738 |
| name | n | 0.04843093 |

Explained order log π(x): -0.484682202339

## All-zero baseline
Integrated Gradients steps used: 50
| Table | log_row_count | log_page_count | log_col_count | log_index_count | has_primary_key | has_foreign_key | log_table_size_bytes | avg_row_width_norm | row_density | is_in_query | row_growth_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| title (t) | -2.612321 | -2.142406 | 1.180467 | 0.017360 | 0.200953 | 0.000000 | -2.023031 | 2.098960 | -0.465491 | 0.226430 | -0.000000 |
| cast_info (ci) | 1.700389 | 1.416222 | -0.532662 | -0.004194 | -0.100097 | -0.000000 | 1.232615 | -0.997060 | 0.517416 | -0.120119 | 0.000000 |
| name (n) | 1.163859 | 0.975431 | -0.457741 | -0.002235 | -0.077806 | -0.000000 | 0.897883 | -0.904446 | 0.210669 | -0.086794 | 0.000000 |

Per-table totals of |attribution|:
- title (t): 10.96741962
- cast_info (ci): 6.62077427
- name (n): 4.77686357

Per-feature totals of |attribution|:
- log_row_count: 5.47656870
- log_page_count: 4.53405857
- log_col_count: 2.17087030
- log_index_count: 0.02378934
- has_primary_key: 0.37885559
- has_foreign_key: 0.00000000
- log_table_size_bytes: 4.15352869
- avg_row_width_norm: 4.00046635
- row_density: 1.19357622
- is_in_query: 0.43334341
- row_growth_rate: 0.00000000

Completeness check:
- sum(attributions): 1.312251091003
- f(x) - f(baseline): 1.307077288628
- difference (sum(attr) - output difference): 0.005173802376
- Captum convergence delta: 0.005173802376
- Completeness within 1e-2: PASS

### Faithfulness mini-test
Drop = log π(original order | x) − log π(order | masked x).
Each random comparison is the mean over 20 fixed-seed random position sets.

#### Ranked over ALL positions
| k | Top-k drop | Mean random drop (20 sets) |
|---:|---:|---:|
| 1 | -0.43019965 | 0.81004893 |
| 2 | -0.43368164 | 0.68567274 |
| 3 | -0.43036129 | 2.18591665 |

#### Ranked over VARYING-feature positions
| k | Top-k drop | Mean random drop (20 sets) |
|---:|---:|---:|
| 1 | -0.43019965 | 1.29265893 |
| 2 | -0.43368164 | 3.56532315 |
| 3 | -0.43036129 | 1.83695138 |

## Seven-table mean baseline
Integrated Gradients steps used: 50
| Table | log_row_count | log_page_count | log_col_count | log_index_count | has_primary_key | has_foreign_key | log_table_size_bytes | avg_row_width_norm | row_density | is_in_query | row_growth_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| title (t) | -0.039632 | -0.136796 | 0.277053 | 0.000000 | 0.000000 | 0.000000 | -0.067728 | 0.191600 | 0.290230 | 0.000000 | -0.000000 |
| cast_info (ci) | 0.279569 | 0.302092 | -0.034668 | -0.000000 | -0.000000 | -0.000000 | 0.149538 | 0.063929 | 0.090717 | -0.000000 | 0.000000 |
| name (n) | 0.053545 | 0.097972 | -0.069691 | -0.000000 | -0.000000 | -0.000000 | 0.048502 | -0.073378 | -0.115776 | -0.000000 | 0.000000 |

Per-table totals of |attribution|:
- title (t): 1.00303757
- cast_info (ci): 0.92051268
- name (n): 0.45886368

Per-feature totals of |attribution|:
- log_row_count: 0.37274545
- log_page_count: 0.53685999
- log_col_count: 0.38141075
- log_index_count: 0.00000000
- has_primary_key: 0.00000000
- has_foreign_key: 0.00000000
- log_table_size_bytes: 0.26576784
- avg_row_width_norm: 0.32890698
- row_density: 0.49672291
- is_in_query: 0.00000000
- row_growth_rate: 0.00000000

Completeness check:
- sum(attributions): 1.307077169418
- f(x) - f(baseline): 1.307077288628
- difference (sum(attr) - output difference): -0.000000119209
- Captum convergence delta: -0.000000119209
- Completeness within 1e-2: PASS

### Faithfulness mini-test
Drop = log π(original order | x) − log π(order | masked x).
Each random comparison is the mean over 20 fixed-seed random position sets.

#### Ranked over ALL positions
| k | Top-k drop | Mean random drop (20 sets) |
|---:|---:|---:|
| 1 | 0.11128795 | 0.04169868 |
| 2 | 0.35703689 | 0.01722820 |
| 3 | 0.61127853 | 0.09474160 |

#### Ranked over VARYING-feature positions
| k | Top-k drop | Mean random drop (20 sets) |
|---:|---:|---:|
| 1 | 0.11128795 | 0.03927688 |
| 2 | 0.35703689 | 0.11885209 |
| 3 | 0.61127853 | 0.10732673 |

Full output saved to `xai/ig_report.md`.
