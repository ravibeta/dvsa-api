"""solution-accelerator IaC: contract checks + `terraform validate`.

The Terraform declares ARM resources; the search index and Foundry agents are
registered by core.azure.deploy in the same apply. These tests pin the seams
between the two so they cannot silently drift apart.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SA = Path(__file__).resolve().parents[1] / "solution-accelerator"
REPO = SA.parent


def _tf(name):
    return (SA / name).read_text(encoding="utf-8")


def _all_tf():
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SA.glob("*.tf")))


def _declared_variables():
    return set(re.findall(r'^variable\s+"([^"]+)"', _tf("variables.tf"), re.M))


# --- the index -----------------------------------------------------------------
def test_legacy_index007_resource_is_gone():
    assert "index007" not in _all_tf()
    # azurerm has no search-index resource; the index is data-plane (deploy.tf).
    assert "azurerm_search_index" not in _all_tf()


def test_base_index_is_dvsa_index_and_registered_by_the_deploy_step():
    assert re.search(r'variable "search_index_name"[^}]*default\s*=\s*"dvsa-index"', _tf("variables.tf"), re.S)
    deploy = _tf("deploy.tf")
    assert "core.azure.deploy" in deploy
    assert 'AZURE_SEARCH_INDEX_NAME' in deploy and "var.search_index_name" in deploy
    assert "index_schema.py" in deploy        # schema changes re-trigger the deploy


def test_search_service_enables_semantic_ranker_for_the_index_semantic_config():
    main = _tf("main.tf")
    assert "semantic_search_sku" in main
    from core.azure import index_schema

    assert index_schema.describe_index("x")["semantic"]["default_configuration"] == "dvsa-semantic"


# --- the Foundry project ---------------------------------------------------------
def test_foundry_is_an_aiservices_resource_with_a_project_and_all_model_deployments():
    f = _tf("foundry.tf")
    assert 'kind                  = "AIServices"' in f
    assert "project_management_enabled = true" in f
    assert "Microsoft.CognitiveServices/accounts/projects@" in f
    v = _tf("variables.tf")
    for model in ("text-embedding-3-large", "mistral-medium-2505"):
        assert model in v
    assert "gpt_deployment" in f and "embedding_deployment" in f


def test_qwen_deployment_is_a_variable_matching_the_agent_yaml():
    import yaml

    y = yaml.safe_load((SA / "agents" / "qwen-agent-in-a-team.yaml").read_text(encoding="utf-8"))
    m = re.search(r'variable "qwen_model_deployment".*?default\s*=\s*"([^"]+)"', _tf("variables.tf"), re.S)
    assert m and m.group(1) == y["definition"]["model"]


def test_search_connection_name_matches_the_agent_yaml_placeholder():
    assert re.search(r'resource "azapi_resource" "search_connection".*?name\s*=\s*var\.search_service_name',
                     _tf("foundry.tf"), re.S)
    for f in ("master-agent-in-a-team.yaml", "scene-search-agent.yaml"):
        assert "connections/${SEARCH_SERVICE}" in (SA / "agents" / f).read_text(encoding="utf-8")


def test_deployer_gets_data_plane_access_before_agents_are_registered():
    f, d = _tf("foundry.tf"), _tf("deploy.tf")
    assert "azurerm_role_assignment" in f and "data.azurerm_client_config.current.object_id" in f
    assert "azurerm_role_assignment.foundry_user" in d      # deploy waits for RBAC
    assert "--retries" in d                                  # and rides out propagation


# --- seams with the python provisioner --------------------------------------------
def test_every_var_the_python_terraform_provisioner_passes_is_declared():
    from types import SimpleNamespace

    from core.azure.terraform import TerraformProvisioner

    cfg = SimpleNamespace(
        resource_group="r", location="l", storage_account="s", input_container="i",
        output_container="o", search_service_name="n", search_sku="standard",
        search_index_name="dvsa-index", vector_dimensions=1536, openai_account="a",
        gpt_deployment="g", gpt_model="gm", embedding_deployment="e", embedding_model="em",
        terraform_dir=str(SA), control_plane_ready=lambda: False)
    passed = set(TerraformProvisioner(cfg)._tf_vars())
    assert passed <= _declared_variables(), passed - _declared_variables()


def test_deploy_cli_registers_index_and_agents_in_one_run():
    import inspect

    from core.azure import deploy

    src = inspect.getsource(deploy.run)
    for step in ("ensure_base_index", "sync_agents", "sync_classic_agents", "verify_deployments"):
        assert step in src, step


def test_agent_yaml_files_all_load_and_reference_only_known_placeholders():
    from core.azure import foundry_provisioning as fp

    known = {"SUBSCRIPTION_ID", "RESOURCE_GROUP", "ACCOUNT", "PROJECT", "SEARCH_SERVICE"}
    for p in (SA / "agents").glob("*.yaml"):
        assert set(re.findall(r"\$\{([A-Z_]+)\}", p.read_text(encoding="utf-8"))) <= known, p.name
    assert len(fp.load_agent_specs(SA / "agents")) == len(list((SA / "agents").glob("*.yaml")))


# --- terraform itself ----------------------------------------------------------
@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_terraform_fmt_and_validate(tmp_path):
    """fmt -check on the real files; init+validate on a scratch copy (no repo pollution)."""
    fmt = subprocess.run(["terraform", "fmt", "-check", "-diff", "-no-color", "-list=false"],
                         cwd=SA, capture_output=True, text=True)
    # fmt exits 3 when files need formatting; only .tf files that we own count.
    assert fmt.returncode == 0, fmt.stdout[-1500:]

    # Mirror the repo layout deploy.tf hashes (solution-accelerator/agents + core/azure).
    module = tmp_path / "solution-accelerator"
    (module / "agents").mkdir(parents=True)
    (tmp_path / "core" / "azure").mkdir(parents=True)
    for f in SA.glob("*.tf"):
        shutil.copy(f, module / f.name)
    for f in (SA / "agents").glob("*.yaml"):
        shutil.copy(f, module / "agents" / f.name)
    for f in (REPO / "core" / "azure").glob("*.py"):
        shutil.copy(f, tmp_path / "core" / "azure" / f.name)
    init = subprocess.run(["terraform", "init", "-backend=false", "-input=false", "-no-color"],
                          cwd=module, capture_output=True, text=True)
    if init.returncode != 0:
        pytest.skip(f"cannot install providers here: {init.stderr[-300:]}")
    val = subprocess.run(["terraform", "validate", "-no-color"], cwd=module,
                         capture_output=True, text=True)
    assert val.returncode == 0, val.stdout + val.stderr
