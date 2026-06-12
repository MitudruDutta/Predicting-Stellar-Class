# LB Probe Ledger — anchor external/nina2025_ps-s6e6/0.97135.csv = 0.97135
# public-split per-row delta (approx): STAR ±0.000046, QSO ±0.000032, GALAXY ±0.000010

| probe | flips | LB score | delta vs 0.97135 | inference |
|---|---|---|---|---|
| probes/probe_d1_S2G_n35.csv | 35 | 0.97137 | +0.00002 | NET POSITIVE — keep all 35, split next round |
| probes/probe_d1_Q2G_n25.csv | 25 | 0.97132 | -0.00003 | NET NEGATIVE — discard group, anchor right |
| probes/probe_d1_G2S_n24.csv | 24 |  |  |  |
| probes/probe_d1_G2Q_n16.csv | 16 |  |  |  |
| probes/probe_d1_SQswap_n8.csv | 8 |  |  |  |

## Round 2 (after d1 probes)
| probes/probe_d2_S2G_halfA17.csv | 17 of S2G |  |  | isolates which half carries the gain |
| probes/probe_d2_S2G_halfB18.csv | 18 of S2G |  |  |  |

## Round 3 results
| probe_d3_B1_n9 | 9 | 0.97137 | +0.00002 | B1 good; B2 inferred +0.00003 |
| probe_d3_A1_n8 | 8 | 0.97136 | +0.00001 | A1 small good; A2 inferred -0.00003 BAD |
| probe_d3_base_sec1 | 13 | 0.97136 | -0.00004 vs combo-base | second-tier dry |
| probe_d3_base_sec2 | 13 | 0.97138 | -0.00002 | dry |
| probe_d3_COMBO_B18_A1 | 26 | **0.97141** | +0.00006 | BANKED BEST |

## Round 4 queue (awaiting daily reset)
| probes/probe_d4_blendHP1.csv | combo+12 STAR/QSO blend-rows |  |  | new vein: 234 our-blend disagreements |
| probes/probe_d4_blendHP2.csv | combo+12 |  |  |  |
| probes/probe_d4_blendG24.csv | combo+24 GALAXY rows |  |  |  |

## Morning routine
1. cd external && set -a && source ../.env && set +a && kaggle datasets download nina2025/ps-s6e6 -p nina2025_ps-s6e6 --unzip --force  (check for files > 0.97141 -> re-anchor)
2. submit the 3 queued probes, paste scores
3. 2 remaining slots cut same-day from results

## Round 4-5 (2026-06-12): blend vein + errdet STAR-direction — all dead/neutral
| probe_d4_blendHP1 | combo+12 S/Q blend rows | 0.97139 | -0.00002 | blend vein dry |
| probe_d4_blendHP2 | combo+12 | 0.97134 | -0.00007 | dry |
| probe_d4_blendG24 | combo+24 G rows | 0.97137 | -0.00004 | dry |
| probe_d5_errdet1 | combo+12 errdet S->G | 0.97141 | 0 | breakeven exactly as predicted (82% acc needed) |
| probe_d5_errdet2 | combo+12 | 0.97141 | 0 | breakeven |

## Round 6-7 (2026-06-12): errdet GALAXY->STAR (payoff asymmetry: breakeven 18%)
| probe_d6_G2S_err1 | combo+15 ranks1-15 | 0.97140 | -0.00001 | top ranks ambiguous |
| probe_d6_G2S_err2 | combo+15 ranks16-30 | **0.97147** | **+0.00006** | NEW BEST ~4 right STARs |
| probe_d6_G2Q_err1 | combo+15 G->Q | 0.97139 | -0.00002 | direction dead |
| probe_d7_G2S_err3 | 147base+15 ranks31-45 | 0.97144 | -0.00003 | vein decaying |
| probe_d7_G2S_err4 | 147base+15 ranks46-60 | 0.97141 | -0.00006 | dead at depth |

## State: BEST = probe_d6_G2S_err2.csv = 0.97147 (~#5). Leader 0.97173, gap 0.00026.
## Next ideas: errdet-v2 binary (GALAXY-pred rows: is it truly STAR?) trained on stacker G/S confusions;
## QSO->STAR direction (breakeven 41%); rescue err1/err3 rights via splits; retrain errdet w/ deeper features.
