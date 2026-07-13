"""Minimal Pydantic-v2 compatibility shim.

The agent kits use Pydantic for schema definition and JSON-schema export. To keep
the package importable and testable *offline* (and on interpreters where Pydantic
is not installed), this module transparently falls back to a tiny, dependency-free
``BaseModel`` implementing the small subset of the Pydantic v2 API the kits rely
on: ``model_dump``, ``model_validate``, ``model_json_schema`` and required-field
validation.

When real Pydantic (v2) is importable it is used verbatim, so production and CI get
full validation while the fallback keeps local/dev runs unblocked.
"""

from __future__ import annotations

import typing
from typing import Any, Dict, List, Optional, Union

try:  # get_args/get_origin are stdlib on 3.8+.
    from typing import get_args, get_origin
except ImportError:  # pragma: no cover - only on Python < 3.8
    from typing_extensions import get_args, get_origin  # type: ignore

try:  # pragma: no cover - exercised via whichever branch is installed
    from pydantic import BaseModel, Field  # type: ignore

    HAVE_PYDANTIC = True
except Exception:  # noqa: BLE001 - any import failure falls back to the shim
    HAVE_PYDANTIC = False

    class _FieldInfo:
        __slots__ = ("default", "default_factory", "description")

        def __init__(self, default: Any = ..., *, default_factory=None,
                     description: Optional[str] = None) -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description

    def Field(default: Any = ..., *, default_factory=None,  # noqa: N802
              description: Optional[str] = None, **_: Any) -> Any:
        """Subset of :func:`pydantic.Field` used by the kit schemas."""
        return _FieldInfo(default, default_factory=default_factory,
                          description=description)

    _JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

    def _is_optional(annotation: Any) -> bool:
        return get_origin(annotation) is Union and type(None) in get_args(annotation)

    def _non_none(annotation: Any) -> Any:
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
        return args[0] if args else annotation

    def _coerce(annotation: Any, value: Any) -> Any:
        if value is None:
            return None
        if _is_optional(annotation):
            annotation = _non_none(annotation)
        origin = get_origin(annotation)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_validate(value) if isinstance(value, dict) else value
        if origin in (list, List):
            (inner,) = get_args(annotation) or (Any,)
            return [_coerce(inner, v) for v in value]
        return value

    def _json_type(annotation: Any) -> Dict[str, Any]:
        if _is_optional(annotation):
            annotation = _non_none(annotation)
        origin = get_origin(annotation)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
        if origin in (list, List):
            (inner,) = get_args(annotation) or (Any,)
            return {"type": "array", "items": _json_type(inner)}
        if origin in (dict, Dict):
            return {"type": "object"}
        return {"type": _JSON_TYPES.get(annotation, "string")}

    class BaseModel:  # type: ignore[no-redef]
        """Dependency-free stand-in for :class:`pydantic.BaseModel` (v2 subset)."""

        def __init__(self, **data: Any) -> None:
            annotations = self._annotations()
            for name, annotation in annotations.items():
                if name in data:
                    setattr(self, name, _coerce(annotation, data[name]))
                    continue
                default = getattr(type(self), name, ...)
                if isinstance(default, _FieldInfo):
                    if default.default_factory is not None:
                        setattr(self, name, default.default_factory())
                    elif default.default is not ...:
                        setattr(self, name, default.default)
                    elif _is_optional(annotation):
                        setattr(self, name, None)
                    else:
                        raise ValueError(f"{type(self).__name__}: missing field '{name}'")
                elif default is not ...:
                    setattr(self, name, default)
                elif _is_optional(annotation):
                    setattr(self, name, None)
                else:
                    raise ValueError(f"{type(self).__name__}: missing field '{name}'")

        @classmethod
        def _annotations(cls) -> Dict[str, Any]:
            # Resolve real types even under ``from __future__ import annotations``
            # (which stores annotations as strings). Cache per-class.
            cached = cls.__dict__.get("__ak_hints__")
            if cached is not None:
                return cached
            try:
                hints = typing.get_type_hints(cls)
            except Exception:  # noqa: BLE001 - fall back to raw string annotations
                hints = {}
                for klass in reversed(cls.__mro__):
                    hints.update(getattr(klass, "__annotations__", {}))
            # Drop private/dunder helpers that aren't declared fields.
            hints = {k: v for k, v in hints.items() if not k.startswith("_")}
            cls.__ak_hints__ = hints
            return hints

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise ValueError(f"{cls.__name__}.model_validate expects a mapping")
            return cls(**data)

        def model_dump(self, *, exclude_none: bool = False) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for name in self._annotations():
                value = getattr(self, name, None)
                if exclude_none and value is None:
                    continue
                out[name] = self._dump_value(value, exclude_none)
            return out

        @staticmethod
        def _dump_value(value: Any, exclude_none: bool) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(exclude_none=exclude_none)
            if isinstance(value, list):
                return [BaseModel._dump_value(v, exclude_none) for v in value]
            if isinstance(value, dict):
                return {k: BaseModel._dump_value(v, exclude_none) for k, v in value.items()}
            return value

        def model_dump_json(self, **kwargs: Any) -> str:
            import json
            return json.dumps(self.model_dump(**kwargs), default=str)

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            props: Dict[str, Any] = {}
            required: List[str] = []
            for name, annotation in cls._annotations().items():
                props[name] = _json_type(annotation)
                default = getattr(cls, name, ...)
                describe = isinstance(default, _FieldInfo) and default.description
                if describe:
                    props[name]["description"] = default.description
                if not _is_optional(annotation) and not _has_default(default):
                    required.append(name)
            schema: Dict[str, Any] = {
                "title": cls.__name__, "type": "object", "properties": props,
            }
            if required:
                schema["required"] = required
            return schema

    def _has_default(default: Any) -> bool:
        if isinstance(default, _FieldInfo):
            return default.default is not ... or default.default_factory is not None
        return default is not ...


__all__ = ["BaseModel", "Field", "HAVE_PYDANTIC"]

# Keep ``typing`` referenced for static analysers of the shim branch.
_ = typing
