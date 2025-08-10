# Legacy Data Conversion Guide

This guide explains how to convert legacy MATLAB datasets from the original polar gini curve implementation to formats compatible with the new Python package.

## Installation

To use the legacy conversion features, install polargini with the legacy dependencies:

```bash
pip install -e .[legacy]
```

This installs the required packages:

- `scipy` - for loading MATLAB `.mat` files
- `h5py` - for newer MATLAB file formats (v7.3+)
- `openpyxl` - for reading Excel files

## Legacy Data Format

The legacy implementation expects the following MATLAB files:

- **`coordinate.mat`** - Cell coordinates (usually tSNE or UMAP)
- **`Expression.mat`** - Gene expression matrix (cells × genes)
- **`geneList.mat`** - List of gene names
- **`ClusterID.mat`** - Cluster assignments for each cell
- **`RSMD_cluster*.xlsx`** - RSMD results for each cluster (optional)

## Quick Start

### 1. Inspect Legacy Dataset

```python
from polargini import load_legacy_dataset

# Load and inspect the dataset
data = load_legacy_dataset("path/to/legacy/data/")

print(f"Coordinates shape: {data['coordinates'].shape}")
print(f"Expression shape: {data['expression'].shape}")
print(f"Number of genes: {len(data['genes'])}")
print(f"Number of clusters: {len(np.unique(data['clusters']))}")
```

### 2. Convert to CSV Files

```python
from polargini import legacy_to_csv

# Convert entire dataset to CSV format
legacy_to_csv("path/to/legacy/data/", "output/csv/")

# This creates:
# - coordinates.csv (x, y, cluster)
# - expression.csv (gene1, gene2, ..., cluster)
# - genes.csv (gene names)
# - clusters.csv (cluster assignments)
```

### 3. Load Data for Analysis

```python
from polargini import load_legacy_for_pgc, polar_gini_curve

# Load using cluster labels
coordinates, labels = load_legacy_for_pgc("path/to/legacy/data/")

# Load specific gene expression
coordinates, expression = load_legacy_for_pgc(
    "path/to/legacy/data/",
    gene_name="Actc1"
)

# Filter to specific clusters
coordinates, labels = load_legacy_for_pgc(
    "path/to/legacy/data/",
    cluster_filter=[1, 2, 3]
)

# Run polar gini curve analysis
result = polar_gini_curve(coordinates, labels)
```

## Detailed Usage

### Loading Individual MATLAB Files

```python
from polargini.io import load_mat_file

# Load a single variable
coordinates = load_mat_file("coordinate.mat", "coordinates")

# Load all variables
all_data = load_mat_file("Expression.mat")
```

### Working with Gene Expression

```python
from polargini import load_legacy_dataset, polar_gini_curve
import numpy as np

# Load dataset
data = load_legacy_dataset("legacy_data/")

# Find genes of interest
gene_names = data['genes']
actc1_idx = gene_names.index('Actc1')

# Get expression for specific gene
actc1_expression = data['expression'][:, actc1_idx]

# Filter out non-clusterable cells
valid_cells = data['clusters'] != -1
coordinates = data['coordinates'][valid_cells]
expression = actc1_expression[valid_cells]

# Analyze spatial expression pattern
result = polar_gini_curve(coordinates, expression)
print(f"Actc1 spatial pattern RSMD: {result['mean_rsmd']:.4f}")
```

### Batch Processing Multiple Genes

```python
from polargini import load_legacy_dataset, polar_gini_curve
import pandas as pd

# Load dataset
data = load_legacy_dataset("legacy_data/")

# Analyze multiple genes
genes_of_interest = ['Actc1', 'Myh6', 'Tnnt2']
results = []

for gene_name in genes_of_interest:
    if gene_name in data['genes']:
        coordinates, expression = load_legacy_for_pgc(
            "legacy_data/",
            gene_name=gene_name
        )

        result = polar_gini_curve(coordinates, expression)
        results.append({
            'gene': gene_name,
            'mean_rsmd': result['mean_rsmd'],
            'min_rsmd': result['min_rsmd'],
            'optimal_angle': result['optimal_angle']
        })

# Create results DataFrame
results_df = pd.DataFrame(results)
print(results_df)
```

### Converting RSMD Results

```python
from polargini import convert_rsmd_results

# Convert Excel RSMD results to CSV
convert_rsmd_results("legacy_data/", "output/rsmd_csv/")

# This converts files like:
# RSMD_cluster1.xlsx -> RSMD_cluster1.csv
# RSMD_cluster2.xlsx -> RSMD_cluster2.csv
# etc.
```

## Command Line Interface

Use the provided example script for common tasks:

```bash
# Inspect dataset
python examples/legacy_converter.py inspect legacy_data/

# Convert to CSV
python examples/legacy_converter.py convert legacy_data/ output/

# Analyze specific gene
python examples/legacy_converter.py gene legacy_data/ Actc1 --output actc1_analysis.png

# Analyze clusters
python examples/legacy_converter.py clusters legacy_data/ --clusters 1 2 3 --output clusters.png
```

## Error Handling

### Missing Dependencies

If you get import errors, install the required dependencies:

```bash
# For basic .mat file support
pip install scipy

# For MATLAB v7.3 files
pip install h5py

# For Excel file support
pip install openpyxl
```

### File Format Issues

```python
from polargini.io import load_mat_file

try:
    data = load_mat_file("problematic_file.mat")
except NotImplementedError:
    print("File is MATLAB v7.3 format, trying with h5py...")
    # The function will automatically retry with h5py
except FileNotFoundError:
    print("File not found. Check the path.")
except KeyError as e:
    print(f"Variable not found in file: {e}")
```

### Data Validation

```python
import numpy as np

# Check for common issues
data = load_legacy_dataset("legacy_data/")

# Check dimensions match
n_cells_coord = data['coordinates'].shape[0]
n_cells_expr = data['expression'].shape[0]
n_cells_clusters = len(data['clusters'])

if not (n_cells_coord == n_cells_expr == n_cells_clusters):
    print("Warning: Dimension mismatch!")
    print(f"Coordinates: {n_cells_coord} cells")
    print(f"Expression: {n_cells_expr} cells")
    print(f"Clusters: {n_cells_clusters} cells")

# Check for missing values
if np.any(np.isnan(data['coordinates'])):
    print("Warning: NaN values in coordinates")

if np.any(np.isnan(data['expression'])):
    print("Warning: NaN values in expression")
```

## Integration with Existing Workflows

### From Legacy MATLAB to New Analysis

```python
# 1. Convert legacy data
from polargini import legacy_to_csv
legacy_to_csv("matlab_data/", "csv_data/")

# 2. Load with standard CSV loader
from polargini import load_csv, polar_gini_curve
coordinates, labels = load_csv("csv_data/coordinates.csv", label_col="cluster")

# 3. Run analysis
result = polar_gini_curve(coordinates, labels)

# 4. Visualize
from polargini.plotting import plot_pgc
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_pgc(result, ax=ax)
plt.show()
```

### Comparative Analysis

```python
# Compare legacy and new implementations
from polargini import load_legacy_for_pgc, polar_gini_curve

# Load same data in both formats
legacy_coords, legacy_labels = load_legacy_for_pgc("legacy/")
csv_coords, csv_labels = load_csv("converted/coordinates.csv", label_col="cluster")

# Run analysis on both
legacy_result = polar_gini_curve(legacy_coords, legacy_labels)
csv_result = polar_gini_curve(csv_coords, csv_labels)

# Compare results
print(f"Legacy RSMD: {legacy_result['mean_rsmd']:.6f}")
print(f"CSV RSMD: {csv_result['mean_rsmd']:.6f}")
print(f"Difference: {abs(legacy_result['mean_rsmd'] - csv_result['mean_rsmd']):.6f}")
```

## Tips and Best Practices

1. **Always inspect your data first** using `load_legacy_dataset()` to understand the structure
2. **Check cluster assignments** - cells with cluster ID -1 are typically excluded from analysis
3. **Validate gene names** before analysis - MATLAB cell arrays can have encoding issues
4. **Use cluster filtering** to focus on specific cell populations
5. **Convert to CSV once** and reuse the converted files for faster loading
6. **Save analysis results** to avoid recomputing expensive operations

## Troubleshooting

### Common Issues

**Issue**: "scipy is required for loading .mat files"
**Solution**: `pip install scipy`

**Issue**: "h5py is required for MATLAB v7.3 files"
**Solution**: `pip install h5py`

**Issue**: Gene names appear garbled
**Solution**: Check the encoding of the original MATLAB file, or manually clean the gene list

**Issue**: Dimension mismatch between files
**Solution**: Verify that all MATLAB files come from the same analysis and have consistent cell ordering

**Issue**: Empty or corrupted .mat files
**Solution**: Re-export from MATLAB using `save('filename.mat', 'variable', '-v7')` for better compatibility
