"""Local Interactive harness — enforce the agent profile around the pipeline.

Loads ``agent_profile.yaml`` and the ``skills/`` directory dynamically, then runs
pipeline steps under the profile's autonomy controls:

* ``dry_run`` — describe intended actions, execute nothing.
* ``confirm`` — run read-only steps freely, prompt before *always-ask* actions.
* ``auto``   — run everything the profile permits without prompting.

*Always-ask* actions (e.g. ``write_output``) require explicit confirmation in any
executing mode; this is how destructive/outward-facing steps are gated.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional

from ..common import (
    LocalFileStorageAdapter,
    MockInferenceAdapter,
    RunInput,
    RunLogger,
    RunOutput,
    SyntheticFrameExtractor,
    LocalVideoFetcher,
)
from ..common._yaml import load_file
from ..common.pipeline import run_pipeline

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODES = ("dry_run", "confirm", "auto")

ConfirmFn = Callable[[str, str], bool]


def _deny(action: str, detail: str) -> bool:
    """Default confirmation: deny (fail-closed) for non-interactive contexts."""
    return False


class GatedActionDenied(RuntimeError):
    """Raised when an always-ask action is not confirmed."""


class LocalAgentHarness:
    """Runs the drone-video pipeline under the local agent profile's controls."""

    def __init__(
        self,
        *,
        profile_path: Optional[str] = None,
        skills_dir: Optional[str] = None,
        mode: Optional[str] = None,
        confirm_fn: Optional[ConfirmFn] = None,
        inference: Optional[MockInferenceAdapter] = None,
    ) -> None:
        self.profile = load_file(profile_path or os.path.join(_HERE, "agent_profile.yaml"))
        self.skills = self._load_skills(skills_dir or os.path.join(_HERE, "skills"))
        self.confirm_fn = confirm_fn or _deny
        self.inference = inference or MockInferenceAdapter()
        self.fetcher = LocalVideoFetcher()
        self.extractor = SyntheticFrameExtractor()
        self.storage = LocalFileStorageAdapter()
        self.agent_id = self.profile.get("agent_id", "local-interactive")
        self.mode = self._resolve_mode(mode)
        self.logger = RunLogger(agent_id=self.agent_id,
                                model_version=self.inference.model_version)

    # ----- profile / skills -------------------------------------------------
    def _resolve_mode(self, requested: Optional[str]) -> str:
        autonomy = self.profile.get("autonomy", {})
        mode = (requested or autonomy.get("default_mode", "confirm")).lower()
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
        # Never exceed the profile's ceiling.
        ceiling = autonomy.get("max_mode", "auto").lower()
        if _MODES.index(mode) > _MODES.index(ceiling):
            mode = ceiling
        return mode

    @staticmethod
    def _load_skills(skills_dir: str) -> Dict[str, Dict[str, Any]]:
        """Dynamically load every ``*.skill`` file in ``skills_dir``."""
        skills: Dict[str, Dict[str, Any]] = {}
        for path in sorted(glob.glob(os.path.join(skills_dir, "*.skill"))):
            spec = load_file(path)
            if isinstance(spec, dict) and spec.get("name"):
                spec["_path"] = path
                skills[spec["name"]] = spec
        return skills

    def list_skills(self) -> List[str]:
        return sorted(self.skills)

    # ----- gating -----------------------------------------------------------
    def _always_ask(self) -> List[str]:
        return list(self.profile.get("always_ask", []))

    def _gate(self, action: str, detail: str) -> None:
        """Enforce always-ask gating for a named action; raise if denied."""
        if action not in self._always_ask():
            return
        if self.mode == "auto":
            allowed = self.profile.get("autonomy", {}).get("allow_skills", [])
            if action in allowed:
                return
        if not self.confirm_fn(action, detail):
            raise GatedActionDenied(f"action '{action}' was not confirmed: {detail}")

    def _check_guardrails(self, run_input: RunInput) -> None:
        rails = self.profile.get("guardrails", {})
        fps = float(run_input.processing_flags.get("fps", 1.0))
        if rails.get("max_fps") and fps > float(rails["max_fps"]):
            raise ValueError(f"fps {fps} exceeds guardrail max_fps {rails['max_fps']}")
        if rails.get("require_local_paths"):
            scheme = run_input.video_uri.split("://", 1)
            if len(scheme) == 2 and scheme[0] not in ("file",):
                raise ValueError(
                    f"remote URI {run_input.video_uri!r} blocked by require_local_paths")

    # ----- steps ------------------------------------------------------------
    def run(
        self, run_input: RunInput, *, store_dest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline; honour dry-run and gate persistence."""
        self._check_guardrails(run_input)
        if self.mode == "dry_run":
            return self._plan(run_input, store_dest)
        run_input.agent_id = run_input.agent_id or self.agent_id
        output = run_pipeline(
            run_input, fetcher=self.fetcher, extractor=self.extractor,
            inference=self.inference, logger=self.logger)
        result: Dict[str, Any] = {"executed": True, "output": output.model_dump()}
        if store_dest:
            self._gate("write_output", f"persist RunOutput to {store_dest}")
            uri = self.storage.store_output(output, store_dest)
            result["stored_uri"] = uri
        return result

    def run_skill(self, name: str, run_input: RunInput, **kwargs: Any) -> Dict[str, Any]:
        """Execute a single named skill against ``run_input``."""
        if name not in self.skills:
            raise KeyError(f"unknown skill {name!r}; have {self.list_skills()}")
        step = self.skills[name].get("step", name)
        self._check_guardrails(run_input)
        if self.mode == "dry_run":
            return {"executed": False, "skill": name, "step": step,
                    "plan": self._plan(run_input, kwargs.get("store_dest"))}
        output = run_pipeline(run_input, fetcher=self.fetcher, extractor=self.extractor,
                              inference=self.inference, logger=self.logger)
        if step == "extract_frames":
            return {"executed": True, "skill": name,
                    "frames_processed": output.summary.get("frames_processed", 0)}
        if step == "detect_objects":
            return {"executed": True, "skill": name,
                    "detections": [d.model_dump() for d in output.detections]}
        if step == "summarize_timeline":
            return {"executed": True, "skill": name, "summary": output.summary}
        return {"executed": True, "skill": name, "output": output.model_dump()}

    def _plan(self, run_input: RunInput, store_dest: Optional[str]) -> Dict[str, Any]:
        """Describe what a run *would* do without executing anything."""
        steps = ["fetch_video", "extract_frames", "run_inference", "summarize_timeline"]
        if store_dest:
            steps.append("write_output (always-ask)")
        return {
            "executed": False,
            "mode": "dry_run",
            "video_uri": run_input.video_uri,
            "fps": run_input.processing_flags.get("fps", 1.0),
            "planned_steps": steps,
            "store_dest": store_dest,
        }


def build_run_output(result: Dict[str, Any]) -> Optional[RunOutput]:
    """Convenience: rebuild a :class:`RunOutput` from a harness ``run`` result."""
    data = result.get("output")
    return RunOutput.model_validate(data) if data else None


__all__ = ["LocalAgentHarness", "GatedActionDenied", "build_run_output"]
