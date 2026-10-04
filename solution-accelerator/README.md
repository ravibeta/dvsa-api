# DVSA solution accelerator

One `terraform apply` provisions the Azure estate **and** registers everything
that lives on its data plane, so all agents and the search index exist before
the first `/chat` query.

| What | Where | How it's provisioned |
|---|---|---|
| Resource group, storage (`input`/`output`), AI Search (semantic ranker on) | `main.tf`, `storage.tf` | Terraform (azurerm) |
| Foundry resource (`AIServices`), project, model deployments, search connection, RBAC | `foundry.tf` | Terraform (azurerm + azapi) |
| Base search index `dvsa-index` | `core/azure/index_schema.py` | `core.azure.deploy`, run by `deploy.tf` |
| Per-account indexes `dvsa-index-s-<account>` | same schema builder | created at runtime by `core/azure` (`ensure_search_index`) |
| Agents (`agents/*.yaml`) | `core/azure/foundry_provisioning.py` | `core.azure.deploy`, run by `deploy.tf` |

The index schema and agent definitions are **not** duplicated in HCL (azurerm
can't manage search indexes or agents): the runtime and the IaC share the
Python, so they can't drift.

## The index

Every index gets the retrieval pipeline from `Vector-Search-Index-Retrieval-Settings`
at the index level: dense vectors with both an **HNSW** profile (default) and an
**exhaustive-KNN** profile (exact search — pass `exhaustive=True` on a vector
query), an Azure OpenAI vectorizer, **BM25** + a boosting scoring profile, a
**semantic** configuration for reranking, and parent-metadata / adjacent-chunk /
version / ACL fields for context assembly. Fields include `caption`, `location`,
`timestamp`, `tags`, `labels`, `annotations`.

## Agents

`agents/*.yaml` are the definitions. At deploy time:

* `${SUBSCRIPTION_ID}`, `${RESOURCE_GROUP}`, `${ACCOUNT}`, `${PROJECT}`,
  `${SEARCH_SERVICE}` are substituted;
* function tools for `tool-agent-in-a-team` / `fn-agent-in-a-team` are generated
  from `core/azure/analyzer.py` (`analyzer_functions` / `image_user_functions`);
* both agent stores are synced: the versioned agents (portal-visible) and the
  **classic** `asst_*` agents that `core/azure/agents.py` actually executes.

`qwen-agent-in-a-team` is registered on the managed-compute deployment
`qwen--qwen35-08b-2`. That deployment accepts images only through
`chat/completions` (Foundry's Agents/Responses layers reject them), so at query
time `core/azure/qwen_agent.py` sends the agent's instructions plus up to 4
selected frames **inline** (base64) straight to the deployment, and its answer is
rolled into the consolidated `chat-agent-in-a-team` response. The managed-compute
deployment itself is created in the Foundry portal/CLI; deploy only verifies it.

## Running

```powershell
terraform init
terraform apply -var-file=dev.tfvars          # ARM + index + agents in one run

# Data plane only / one-time syncs (needs the "Foundry User" role on the project):
python -m core.azure.deploy --dry-run
python -m core.azure.deploy --only tool-agent-in-a-team   # renew one agent
python -m core.azure.deploy --recreate-index              # rebuild base index even if populated
```

`deploy.tf` re-runs the data-plane step whenever the agent yaml, index schema or
analyzer code change. The deploy never drops an index that holds documents; an
empty legacy base index is rebuilt automatically unless an AI Search knowledge
agent still references it (reported as `blocked`).

`found-vision-1/` is an `aztfexport` snapshot of the existing Foundry resource,
kept for reference; it is not part of this module.
