# Migration from MATLAB scripts

| MATLAB script             | Python API                                                  |
| ------------------------- | ----------------------------------------------------------- |
| `make2DGini.m`            | `polargini.pgc.polar_gini_curve`                            |
| `computeGini.m`           | `polargini.metrics.gini`                                    |
| `ComputePVal.m`           | `polargini.stats.compute_pvalues`                            |
| `spatial_MOB_analysis.py` | `polargini.cli` (`pgc` command)                             |
