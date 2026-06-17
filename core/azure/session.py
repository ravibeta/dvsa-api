"""Session-scoped Azure environment: the public setup/teardown API.

Typical use from a request handler / Celery task::

    env = create_session_azure_environment(session_id, user_id=request.user.pk)
    try:
        env.process_frame(frame_id="f0001", image_bytes=..., path="input/f0001.jpg")
    finally:
        teardown_session_azure_environment(session_id)

Global vs. session scope
------------------------
*Global* resources (the ``sadronevideo`` storage account, the AI Search service,
the Foundry/OpenAI account + model deployments) are shared, slow and expensive
to create, and their names are globally unique — so they are provisioned **once**
and only when ``provision_global=True``. *Session* scope is achieved logically:
a per-session search index (``index`` isolation) or session-tagged rows in the
shared index (``filter`` isolation), plus a per-session blob prefix. Teardown
removes only the session-scoped artifacts unless ``delete_global=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import naming
from .config import AzureEnvironmentConfig
from .provisioning import Provisioner, get_provisioner

logger = logging.getLogger("apps.azure")

# In-process registry of live environments so teardown can reuse open clients.
_REGISTRY: Dict[str, "SessionAzureEnvironment"] = {}


@dataclass
class SessionAzureEnvironment:
    """Handle to one user session's Azure resources and clients."""

    session_id: str
    user_id: Optional[str]
    config: AzureEnvironmentConfig
    mode: str
    index_name: str
    input_container: str
    output_container: str
    input_prefix: str
    output_prefix: str
    provisioner: Provisioner
    global_provisioned: bool = False
    operations: List[Dict[str, Any]] = field(default_factory=list)
    # endpoints/keys discovered during provisioning (may be synthetic in dry-run)
    endpoints: Dict[str, Any] = field(default_factory=dict)

    _blob_service: Any = field(default=None, repr=False, compare=False)
    _search_client: Any = field(default=None, repr=False, compare=False)

    # ----- data-plane clients (lazy) -------------------------------------
    def get_blob_service_client(self):
        """Return a cached ``BlobServiceClient`` for ``sadronevideo``."""
        if self._blob_service is None:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

            c = self.config
            if c.storage_connection_string:
                self._blob_service = BlobServiceClient.from_connection_string(
                    c.storage_connection_string
                )
            else:
                cred = c.account_key or self.endpoints.get("storage_key")
                self._blob_service = BlobServiceClient(
                    account_url=f"https://{c.storage_account}.blob.core.windows.net",
                    credential=cred,
                )
        return self._blob_service

    def get_search_client(self):
        """Return a cached ``SearchClient`` bound to this session's index."""
        if self._search_client is None:
            from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
            from azure.search.documents import SearchClient  # noqa: PLC0415

            c = self.config
            self._search_client = SearchClient(
                endpoint=c.search_endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(c.search_admin_key),
            )
        return self._search_client

    # ----- model-backed helpers ------------------------------------------
    def embed(self, text: str) -> List[float]:
        """Return a ``vector_dimensions``-wide embedding for ``text``.

        Uses Azure OpenAI embeddings when configured; otherwise a deterministic
        offline pseudo-embedding (so RAG indexing is testable without a model).
        """
        dims = self.config.vector_dimensions
        c = self.config
        if c.openai_endpoint and c.openai_api_key:
            try:
                return _azure_openai_embedding(c, text)[:dims]
            except Exception as exc:  # noqa: BLE001 - fall back offline
                logger.warning("embedding call failed, using offline vector: %s", exc)
        return _deterministic_embedding(text, dims)

    def annotate(self, description: str) -> Dict[str, Any]:
        """Generate caption/labels/tags for a frame via the configured LLM.

        Reuses :func:`apps.observability.llm.get_llm_client` so the offline
        ``echo`` client keeps this path runnable without credentials.
        """
        try:
            from apps.observability.llm import get_llm_client  # noqa: PLC0415

            client = get_llm_client()
            caption = client.complete(
                f"Caption this aerial drone frame: {description}",
                system="You caption aerial drone imagery in one sentence.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("annotate failed, using fallback: %s", exc)
            caption = f"aerial frame: {description}"[:280]
        tokens = [t for t in description.replace(",", " ").split() if t][:8]
        return {"caption": caption, "labels": tokens[:5], "tags": tokens}

    # ----- end-to-end frame ingestion ------------------------------------
    def index_frame(
        self,
        *,
        frame_id: str,
        vector: List[float],
        caption: str,
        labels: List[str],
        tags: List[str],
        path: str,
    ) -> Dict[str, Any]:
        """Upsert one frame document into this session's search index."""
        doc = {
            "id": f"{naming.session_slug(self.session_id)}-{frame_id}",
            "vector": vector,
            "caption": caption,
            "labels": labels,
            "tags": tags,
            "user": str(self.user_id or ""),
            "session": str(self.session_id),
            "path": path,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        if self.mode == "dryrun":
            logger.info("dryrun index_frame %s -> %s", doc["id"], self.index_name)
            return {"indexed": doc["id"], "dryrun": True}
        self.get_search_client().upload_documents(documents=[doc])
        return {"indexed": doc["id"]}

    def process_frame(
        self, *, frame_id: str, description: str, path: str,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Full RAG ingestion: (optional upload) -> embed -> annotate -> index."""
        blob_path = naming.blob_path(self.session_id, path)
        if image_bytes is not None and self.mode != "dryrun":
            svc = self.get_blob_service_client()
            svc.get_blob_client(self.input_container, blob_path).upload_blob(
                image_bytes, overwrite=True
            )
        ann = self.annotate(description)
        vector = self.embed(f"{ann['caption']} {description}")
        result = self.index_frame(
            frame_id=frame_id, vector=vector, caption=ann["caption"],
            labels=ann["labels"], tags=ann["tags"], path=blob_path,
        )
        return {"frame_id": frame_id, "path": blob_path, **ann, **result}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary (safe for API responses — no secrets)."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "isolation": self.config.session_isolation,
            "storage_account": self.config.storage_account,
            "input_container": self.input_container,
            "output_container": self.output_container,
            "input_prefix": self.input_prefix,
            "output_prefix": self.output_prefix,
            "search_index": self.index_name,
            "vector_dimensions": self.config.vector_dimensions,
            "embedding_deployment": self.config.embedding_deployment,
            "gpt_deployment": self.config.gpt_deployment,
            "global_provisioned": self.global_provisioned,
            "operations": self.operations,
        }


# ============================ public API ================================

def create_session_azure_environment(
    session_id: str,
    user_id: Optional[str] = None,
    *,
    provision_global: bool = False,
    isolation: Optional[str] = None,
    mode: Optional[str] = None,
    config: Optional[AzureEnvironmentConfig] = None,
) -> SessionAzureEnvironment:
    """Provision (or attach to) the Azure resources a session needs.

    Parameters
    ----------
    session_id:
        Stable id for the session (e.g. the DRF session key or a UUID).
    user_id:
        Owning user; stored on every indexed document for filtering.
    provision_global:
        When True, ensure the shared storage account / search service /
        Foundry account + deployments exist first. Off by default because those
        are slow and usually managed out-of-band (Terraform / one-time setup).
    isolation:
        Override ``AZURE_SESSION_ISOLATION`` (``"index"`` or ``"filter"``).
    mode:
        Override the provisioner backend (``sdk`` | ``terraform`` | ``dryrun``).
    """
    cfg = config or AzureEnvironmentConfig.from_settings()
    if isolation:
        cfg = _with(cfg, session_isolation=isolation)
    effective_mode = (mode or cfg.resolve_mode()).lower()

    provisioner = get_provisioner(cfg, effective_mode)
    endpoints: Dict[str, Any] = {}

    if provision_global:
        glob = provisioner.ensure_global()
        endpoints = _collect_endpoints(glob)

    idx = naming.index_name(cfg.search_index_name, session_id, cfg.session_isolation)
    provisioner.ensure_search_index(idx)

    env = SessionAzureEnvironment(
        session_id=str(session_id),
        user_id=str(user_id) if user_id is not None else None,
        config=cfg,
        mode=provisioner.mode,
        index_name=idx,
        input_container=cfg.input_container,
        output_container=cfg.output_container,
        input_prefix=naming.blob_prefix(session_id),
        output_prefix=naming.blob_prefix(session_id),
        provisioner=provisioner,
        global_provisioned=provision_global,
        operations=provisioner.operations,
        endpoints=endpoints,
    )
    _REGISTRY[str(session_id)] = env
    logger.info("created azure session env %s (mode=%s, index=%s)",
                session_id, env.mode, idx)
    return env


def teardown_session_azure_environment(
    session_id: str,
    *,
    delete_global: bool = False,
    config: Optional[AzureEnvironmentConfig] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Release a session's resources.

    Removes session-scoped artifacts (the per-session search index when using
    ``index`` isolation, and the session's blob prefixes). When
    ``delete_global`` is True the shared storage account / search service /
    Foundry account are torn down too (use with care).
    """
    env = _REGISTRY.pop(str(session_id), None)
    cfg = config or (env.config if env else AzureEnvironmentConfig.from_settings())
    provisioner = (
        env.provisioner if env else get_provisioner(cfg, mode or cfg.resolve_mode())
    )

    ops: List[Dict[str, Any]] = []
    # Session-scoped index only when isolation gives each session its own.
    if cfg.session_isolation != "filter":
        idx = naming.index_name(cfg.search_index_name, session_id,
                                cfg.session_isolation)
        ops.append(provisioner.teardown_search_index(idx))

    prefix = naming.blob_prefix(session_id)
    for container in (cfg.input_container, cfg.output_container):
        ops.append(provisioner.teardown_blob_prefix(container, prefix))

    if delete_global:
        ops.append(provisioner._record("delete", "global",
                                      note="delete_global requested"))

    logger.info("torn down azure session env %s (%d ops)", session_id, len(ops))
    return {"session_id": str(session_id), "deleted_global": delete_global,
            "operations": ops}


# ============================ internals =================================

def _with(cfg: AzureEnvironmentConfig, **overrides) -> AzureEnvironmentConfig:
    from dataclasses import replace

    return replace(cfg, **overrides)


def _collect_endpoints(glob: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for section in glob.values():
        if isinstance(section, dict):
            for key in ("endpoint", "admin_key", "api_key"):
                if key in section:
                    out[f"{section.get('resource', 'res')}_{key}"] = section[key]
    return out


def _deterministic_embedding(text: str, dims: int) -> List[float]:
    """Stable pseudo-embedding from a hash stream — offline, unit length-ish."""
    out: List[float] = []
    counter = 0
    while len(out) < dims:
        h = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        # 8 floats per 32-byte digest (4 bytes each).
        for i in range(0, 32, 4):
            out.append(struct.unpack("<I", h[i:i + 4])[0] / 0xFFFFFFFF)
            if len(out) >= dims:
                break
        counter += 1
    return out[:dims]


def _azure_openai_embedding(cfg: AzureEnvironmentConfig, text: str) -> List[float]:
    """Call Azure OpenAI embeddings over REST (stdlib only, like llm.py)."""
    url = (
        f"{cfg.openai_endpoint.rstrip('/')}/openai/deployments/"
        f"{cfg.embedding_deployment}/embeddings?api-version={cfg.openai_api_version}"
    )
    body = json.dumps({"input": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"api-key": cfg.openai_api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["embedding"]
