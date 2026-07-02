"""Structural tests for the Databricks notebooks, Job template and sample data.

No live Databricks: these validate that the templates are well-formed and
parameterized so a user can run them with minimal edits.
"""

from __future__ import annotations

import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NB_DIR = os.path.join(REPO_ROOT, "notebooks", "databricks")
SAMPLE_DIR = os.path.join(REPO_ROOT, "notebooks", "sample_data")
JOB_TEMPLATE = os.path.join(REPO_ROOT, "jobs", "job_template.json")

EXPECTED_NOTEBOOKS = [
    "00-setup", "01-ingest-and-prepare", "02-batch-inference",
    "03-streaming-inference", "04-deploy-job", "05-export-to-repo",
]
# Notebooks that must expose the widget-backed CONFIG cell.
CONFIG_NOTEBOOKS = [
    "00-setup", "01-ingest-and-prepare", "02-batch-inference",
    "03-streaming-inference", "04-deploy-job",
]


def _load_nb(name):
    with open(os.path.join(NB_DIR, name + ".ipynb"), encoding="utf-8") as fh:
        return json.load(fh)


def _cell_text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_is_valid_json_nbformat4(name):
    nb = _load_nb(name)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_passes_nbformat_validate(name):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(os.path.join(NB_DIR, name + ".ipynb"), as_version=4)
    nbformat.validate(nb)  # raises on an invalid notebook


@pytest.mark.parametrize("name", CONFIG_NOTEBOOKS)
def test_config_notebooks_have_widgets(name):
    text = _cell_text(_load_nb(name))
    assert "dbutils.widgets" in text
    assert "model_name" in text


def test_all_expected_notebooks_present():
    on_disk = {f[:-6] for f in os.listdir(NB_DIR) if f.endswith(".ipynb")}
    assert set(EXPECTED_NOTEBOOKS).issubset(on_disk)


class TestJobTemplate:
    def test_valid_json_and_required_keys(self):
        with open(JOB_TEMPLATE, encoding="utf-8") as fh:
            job = json.load(fh)
        assert job["name"]
        assert job["tasks"], "job must define at least one task"
        task = job["tasks"][0]
        assert "notebook_task" in task
        params = task["notebook_task"]["base_parameters"]
        # The Job must pass the parameters the notebooks read.
        for key in ("input_delta_path", "output_delta_path", "model_name", "batch_size"):
            assert key in params


class TestSampleData:
    @pytest.mark.parametrize("fname", ["tracks.json", "frames.json"])
    def test_sample_json_lines_parse(self, fname):
        path = os.path.join(SAMPLE_DIR, fname)
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        assert lines
        for ln in lines:
            row = json.loads(ln)
            assert "video_id" in row and "frame_index" in row

    def test_tracks_map_to_context(self):
        from dvsa_databricks_connector import row_to_context

        path = os.path.join(SAMPLE_DIR, "tracks.json")
        with open(path, encoding="utf-8") as fh:
            row = json.loads(fh.readline())
        ctx = row_to_context(row)
        assert ctx["tracks"] and "sensor_meta" in ctx
