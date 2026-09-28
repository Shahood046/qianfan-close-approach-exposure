# Figure 7 numbers (25_fig7_model.py)

113 inter-plane pairs with SGP4 minimum <= 50 km; reference epoch from 09_model_checks.py part A.

| version | r (log d) | median abs error (km) | 90th pct (km) | max (km) |
|---|---|---|---|---|
| V4 | 0.923 | 1.36 | 12.79 | 40.8 |
| V6 | 0.987 | 0.24 | 1.00 | 10.4 |

Six largest V4 errors:

| pair | planes | SGP4 (km) | V4 (km) | V6 (km) |
|---|---|---|---|---|
| 2024-232E/2025-046A | P1-P6 | 16.1 | 56.8 | 16.8 |
| 2024-232C/2025-046D | P1-P6 | 24.1 | 56.1 | 24.2 |
| 2024-232U/2025-046Q | P1-P6 | 41.2 | 69.2 | 42.1 |
| 2025-016Q/2026-124M | P4-P9 | 43.7 | 68.3 | 43.9 |
| 2024-232P/2025-046R | P1-P6 | 25.2 | 46.0 | 26.0 |
| 2025-016L/2026-124D | P4-P9 | 13.2 | 33.0 | 13.7 |
