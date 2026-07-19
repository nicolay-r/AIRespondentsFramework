import importlib
import inspect
from pathlib import Path

from src.pipelines.base import Pipeline

_PIPELINES_DIR = Path(__file__).resolve().parent
_SKIP_MODULES = frozenset({"base", "registry"})


def available_pipeline_names() -> list[str]:
    return sorted(
        path.stem.replace("_", "-")
        for path in _PIPELINES_DIR.glob("*.py")
        if path.stem not in _SKIP_MODULES
    )


def _import_pipeline_class(pipeline_name: str) -> type[Pipeline]:
    module_name = pipeline_name.replace("-", "_")
    module_path = f"src.pipelines.{module_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        available = ", ".join(available_pipeline_names())
        raise ValueError(
            f"Unknown pipeline {pipeline_name!r}. Available: {available}"
        ) from exc

    pipeline_classes = [
        obj
        for obj in module.__dict__.values()
        if inspect.isclass(obj)
        and issubclass(obj, Pipeline)
        and obj is not Pipeline
        and obj.__module__ == module.__name__
    ]
    if len(pipeline_classes) != 1:
        raise ImportError(
            f"Expected exactly one Pipeline subclass in {module_path}, "
            f"found {len(pipeline_classes)}"
        )
    return pipeline_classes[0]


def create_pipeline(pipeline_name: str, **context: object) -> Pipeline:
    pipeline_class = _import_pipeline_class(pipeline_name)
    kwargs: dict[str, object] = {}
    for name, param in inspect.signature(pipeline_class.__init__).parameters.items():
        if name == "self":
            continue
        if name in context:
            kwargs[name] = context[name]
        elif param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"Missing required argument {name!r} for pipeline {pipeline_name!r}"
            )
    return pipeline_class(**kwargs)
