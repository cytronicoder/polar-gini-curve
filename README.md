# Python Polar Gini Curve

[![PyPI version](https://badge.fury.io/py/polargini.svg)](https://badge.fury.io/py/polargini)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/release/python-390/)

This is a Python port of the Polar Gini Curve introduced by Nguyen et al. (2021). I've implemented the core functionality in Python, making it easier to integrate into Python-based data analysis workflows. The original MATLAB code is available in the [aimed-lab/Polar-Gini-Curve repository](https://github.com/aimed-lab/Polar-Gini-Curve).

## Quickstart

To get started with the Polar Gini Curve in Python, you can install the package via pip, then use the CLI to compute the Polar Gini Curve from a CSV file containing your data. Here's a quick example:

```bash
pip install polargini
pgc --csv data.csv --plot
```

For more detailed usage, refer to the [original documentation](https://github.com/aimed-lab/Polar-Gini-Curve/blob/main/README.md).

## Citation

Nguyen, T.M., Jeevan, J.J., Xu, N. and Chen, J.Y., "Polar Gini Curve: a technique to discover gene expression spatial patterns from single-cell RNA-seq data," _Genomics, Proteomics & Bioinformatics_ 19(3), 493-503 (2021).
