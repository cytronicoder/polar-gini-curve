"""Tests for legacy data conversion functions."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from polargini.io import (
        convert_rsmd_results,
        legacy_to_csv,
        load_legacy_dataset,
        load_legacy_for_pgc,
        load_mat_file,
    )
except ImportError:
    print("PGC package not found. Skipping tests.")
    pytest.skip("PGC package not found", allow_module_level=True)


class TestLoadMatFile:
    """Tests for load_mat_file function."""

    def test_load_mat_file_no_scipy(self):
        """Test that appropriate error is raised when scipy is not available."""
        with patch("polargini.io.HAS_SCIPY", False):
            with pytest.raises(ImportError, match="scipy is required"):
                load_mat_file("dummy.mat")

    def test_load_mat_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_mat_file("nonexistent.mat")

    @patch("polargini.io.scipy.io.loadmat")
    @patch("polargini.io.os.path.exists")
    def test_load_mat_file_single_variable(self, mock_exists, mock_loadmat):
        """Test loading a single variable from a mat file."""
        mock_exists.return_value = True
        mock_data = {
            "__header__": b"header",
            "__version__": "1.0",
            "__globals__": [],
            "test_var": np.array([[1, 2], [3, 4]]),
        }
        mock_loadmat.return_value = mock_data

        result = load_mat_file("test.mat", "test_var")
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    @patch("polargini.io.scipy.io.loadmat")
    @patch("polargini.io.os.path.exists")
    def test_load_mat_file_all_variables(self, mock_exists, mock_loadmat):
        """Test loading all variables from a mat file."""
        mock_exists.return_value = True
        mock_data = {
            "__header__": b"header",
            "__version__": "1.0",
            "__globals__": [],
            "var1": np.array([1, 2, 3]),
            "var2": np.array([4, 5, 6]),
        }
        mock_loadmat.return_value = mock_data

        result = load_mat_file("test.mat")
        expected = {"var1": np.array([1, 2, 3]), "var2": np.array([4, 5, 6])}

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    @patch("polargini.io.scipy.io.loadmat")
    @patch("polargini.io.os.path.exists")
    def test_load_mat_file_variable_not_found(self, mock_exists, mock_loadmat):
        """Test error when requested variable is not found."""
        mock_exists.return_value = True
        mock_data = {"__header__": b"header", "var1": np.array([1, 2, 3])}
        mock_loadmat.return_value = mock_data

        with pytest.raises(KeyError, match="Variable 'missing_var' not found"):
            load_mat_file("test.mat", "missing_var")


class TestLoadLegacyDataset:
    """Tests for load_legacy_dataset function."""

    @patch("polargini.io.load_mat_file")
    def test_load_legacy_dataset_basic(self, mock_load_mat):
        """Test basic loading of legacy dataset."""

        # Mock the mat file loading
        def mock_load_side_effect(path):
            if "coordinate.mat" in path:
                return np.array([[1.0, 2.0], [3.0, 4.0]])
            elif "Expression.mat" in path:
                return np.array([[0.1, 0.2], [0.3, 0.4]])
            elif "geneList.mat" in path:
                # Simulate MATLAB cell array
                return np.array([["Gene1"], ["Gene2"]], dtype=object)
            elif "ClusterID.mat" in path:
                return np.array([[1], [2]])
            return {}

        mock_load_mat.side_effect = mock_load_side_effect

        result = load_legacy_dataset("/fake/path")

        # Check coordinates
        np.testing.assert_array_equal(result["coordinates"], [[1.0, 2.0], [3.0, 4.0]])

        # Check expression
        np.testing.assert_array_equal(result["expression"], [[0.1, 0.2], [0.3, 0.4]])

        # Check genes
        assert result["genes"] == ["Gene1", "Gene2"]

        # Check clusters
        np.testing.assert_array_equal(result["clusters"], [1, 2])

    @patch("polargini.io.load_mat_file")
    def test_load_legacy_dataset_dict_format(self, mock_load_mat):
        """Test loading when mat files return dictionaries."""

        def mock_load_side_effect(path):
            if "coordinate.mat" in path:
                return {"coords": np.array([[1.0, 2.0]])}
            elif "Expression.mat" in path:
                return {"expr": np.array([[0.1, 0.2]])}
            elif "geneList.mat" in path:
                return {"genes": np.array(["Gene1"], dtype=object)}
            elif "ClusterID.mat" in path:
                return {"clusters": np.array([[1]])}
            return {}

        mock_load_mat.side_effect = mock_load_side_effect

        result = load_legacy_dataset("/fake/path")

        assert result["coordinates"].shape == (1, 2)
        assert result["expression"].shape == (1, 2)
        assert len(result["genes"]) == 1
        assert len(result["clusters"]) == 1


class TestLoadLegacyForPgc:
    """Tests for load_legacy_for_pgc function."""

    @patch("polargini.io.load_legacy_dataset")
    def test_load_legacy_for_pgc_clusters(self, mock_load_dataset):
        """Test loading legacy data using cluster labels."""
        mock_data = {
            "coordinates": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "expression": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "genes": ["Gene1", "Gene2"],
            "clusters": np.array([1, 2, -1]),  # Include non-clusterable cell
        }
        mock_load_dataset.return_value = mock_data

        coords, labels = load_legacy_for_pgc("/fake/path")

        # Should exclude non-clusterable cells (-1)
        expected_coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_labels = np.array([1, 2])

        np.testing.assert_array_equal(coords, expected_coords)
        np.testing.assert_array_equal(labels, expected_labels)

    @patch("polargini.io.load_legacy_dataset")
    def test_load_legacy_for_pgc_gene_expression(self, mock_load_dataset):
        """Test loading legacy data using gene expression."""
        mock_data = {
            "coordinates": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "expression": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "genes": ["Gene1", "Gene2"],
            "clusters": np.array([1, 2]),
        }
        mock_load_dataset.return_value = mock_data

        coords, labels = load_legacy_for_pgc("/fake/path", gene_name="Gene1")

        expected_coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_labels = np.array([0.1, 0.3])  # First gene expression values

        np.testing.assert_array_equal(coords, expected_coords)
        np.testing.assert_array_equal(labels, expected_labels)

    @patch("polargini.io.load_legacy_dataset")
    def test_load_legacy_for_pgc_gene_not_found(self, mock_load_dataset):
        """Test error when requested gene is not found."""
        mock_data = {
            "coordinates": np.array([[1.0, 2.0]]),
            "expression": np.array([[0.1, 0.2]]),
            "genes": ["Gene1", "Gene2"],
            "clusters": np.array([1]),
        }
        mock_load_dataset.return_value = mock_data

        with pytest.raises(ValueError, match="Gene 'MissingGene' not found"):
            load_legacy_for_pgc("/fake/path", gene_name="MissingGene")

    @patch("polargini.io.load_legacy_dataset")
    def test_load_legacy_for_pgc_cluster_filter(self, mock_load_dataset):
        """Test filtering by specific clusters."""
        mock_data = {
            "coordinates": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "expression": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "genes": ["Gene1", "Gene2"],
            "clusters": np.array([1, 2, 3]),
        }
        mock_load_dataset.return_value = mock_data

        coords, labels = load_legacy_for_pgc("/fake/path", cluster_filter=[1, 3])

        # Should only include clusters 1 and 3
        expected_coords = np.array([[1.0, 2.0], [5.0, 6.0]])
        expected_labels = np.array([1, 3])

        np.testing.assert_array_equal(coords, expected_coords)
        np.testing.assert_array_equal(labels, expected_labels)


class TestLegacyToCsv:
    """Tests for legacy_to_csv function."""

    @patch("polargini.io.load_legacy_dataset")
    def test_legacy_to_csv(self, mock_load_dataset):
        """Test conversion of legacy dataset to CSV files."""
        mock_data = {
            "coordinates": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "expression": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "genes": ["Gene1", "Gene2"],
            "clusters": np.array([1, 2]),
        }
        mock_load_dataset.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_to_csv("/fake/legacy", temp_dir)

            # Check that files were created
            output_path = Path(temp_dir)
            assert (output_path / "coordinates.csv").exists()
            assert (output_path / "expression.csv").exists()
            assert (output_path / "genes.csv").exists()
            assert (output_path / "clusters.csv").exists()

            # Check coordinates file content
            coord_df = pd.read_csv(output_path / "coordinates.csv")
            assert list(coord_df.columns) == ["x", "y", "cluster"]
            assert len(coord_df) == 2

            # Check expression file content
            expr_df = pd.read_csv(output_path / "expression.csv")
            assert "Gene1" in expr_df.columns
            assert "Gene2" in expr_df.columns
            assert "cluster" in expr_df.columns

            # Check genes file content
            gene_df = pd.read_csv(output_path / "genes.csv")
            assert list(gene_df["gene"]) == ["Gene1", "Gene2"]


class TestConvertRsmdResults:
    """Tests for convert_rsmd_results function."""

    @patch("polargini.io.pd.read_excel")
    @patch("polargini.io.Path.glob")
    def test_convert_rsmd_results(self, mock_glob, mock_read_excel):
        """Test conversion of RSMD Excel files to CSV."""
        # Mock finding Excel files
        mock_file1 = Mock()
        mock_file1.stem = "RSMD_cluster1"
        mock_file1.name = "RSMD_cluster1.xlsx"
        mock_file2 = Mock()
        mock_file2.stem = "RSMD_cluster2"
        mock_file2.name = "RSMD_cluster2.xlsx"
        mock_glob.return_value = [mock_file1, mock_file2]

        # Mock DataFrame
        mock_df = pd.DataFrame({"gene": ["Gene1", "Gene2"], "value": [0.1, 0.2]})
        mock_read_excel.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            convert_rsmd_results("/fake/legacy", temp_dir)

            # Check that CSV files were created
            output_path = Path(temp_dir)
            assert (output_path / "RSMD_cluster1.csv").exists()
            assert (output_path / "RSMD_cluster2.csv").exists()
