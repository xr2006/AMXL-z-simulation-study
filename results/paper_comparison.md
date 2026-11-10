# Reproduced results vs. the paper

Values are rounded to three decimals, as in the paper. A cell matches if |difference| <= 0.001.

| Table | Cells | Identical after rounding | Within tolerance | Max abs. difference |
|---|---:|---:|---:|---:|
| Table 2 | 216 | 216 | 216 | 0.000 |
| Table 3 | 120 | 120 | 120 | 0.000 |
| Table A.2 | 64 | 64 | 64 | 0.000 |

## Table 4 (machine dependent; reproduced on Apple M2 Pro (Darwin x86_64))

| Model | Quantity | sigma | C | Paper | Reproduced | Ratio |
|---|---|---|---|---:|---:|---:|
| MNL | CPU time (s) | 0.2 | 5 | 3.0 | 1.1 | 0.36 |
| MNL | CPU time (s) | 0.2 | 10 | 2.9 | 1.2 | 0.42 |
| MNL | CPU time (s) | 0.5 | 5 | 2.8 | 1.1 | 0.40 |
| MNL | CPU time (s) | 0.5 | 10 | 2.8 | 1.1 | 0.40 |
| MNL | Peak memory (MB) | max | 5 | 226.0 | 165.0 | 0.73 |
| MNL | Peak memory (MB) | max | 10 | 233.0 | 164.6 | 0.71 |
| MXL | CPU time (s) | 0.2 | 5 | 5.6 | 3.7 | 0.66 |
| MXL | CPU time (s) | 0.2 | 10 | 8.7 | 6.8 | 0.78 |
| MXL | CPU time (s) | 0.5 | 5 | 6.8 | 4.5 | 0.67 |
| MXL | CPU time (s) | 0.5 | 10 | 12.3 | 10.3 | 0.84 |
| MXL | Peak memory (MB) | max | 5 | 1,099.0 | 1,035.3 | 0.94 |
| MXL | Peak memory (MB) | max | 10 | 1,833.0 | 1,586.8 | 0.87 |
| TasteNet | CPU time (s) | 0.2 | 5 | 525.0 | 536.4 | 1.02 |
| TasteNet | CPU time (s) | 0.2 | 10 | 869.0 | 910.7 | 1.05 |
| TasteNet | CPU time (s) | 0.5 | 5 | 425.0 | 462.1 | 1.09 |
| TasteNet | CPU time (s) | 0.5 | 10 | 630.0 | 689.3 | 1.09 |
| TasteNet | Peak memory (MB) | max | 5 | 465.0 | 337.7 | 0.73 |
| TasteNet | Peak memory (MB) | max | 10 | 462.0 | 337.9 | 0.73 |
| AMXL-linear | CPU time (s) | 0.2 | 5 | 133.0 | 125.8 | 0.95 |
| AMXL-linear | CPU time (s) | 0.2 | 10 | 143.0 | 136.7 | 0.96 |
| AMXL-linear | CPU time (s) | 0.5 | 5 | 123.0 | 116.4 | 0.95 |
| AMXL-linear | CPU time (s) | 0.5 | 10 | 143.0 | 133.8 | 0.94 |
| AMXL-linear | Peak memory (MB) | max | 5 | 367.0 | 306.8 | 0.84 |
| AMXL-linear | Peak memory (MB) | max | 10 | 367.0 | 322.8 | 0.88 |
| AMXL-quadratic | CPU time (s) | 0.2 | 5 | 200.0 | 201.7 | 1.01 |
| AMXL-quadratic | CPU time (s) | 0.2 | 10 | 173.0 | 178.1 | 1.03 |
| AMXL-quadratic | CPU time (s) | 0.5 | 5 | 144.0 | 148.2 | 1.03 |
| AMXL-quadratic | CPU time (s) | 0.5 | 10 | 166.0 | 171.4 | 1.03 |
| AMXL-quadratic | Peak memory (MB) | max | 5 | 367.0 | 309.1 | 0.84 |
| AMXL-quadratic | Peak memory (MB) | max | 10 | 372.0 | 270.7 | 0.73 |
