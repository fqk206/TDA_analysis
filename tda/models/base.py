#!/usr/bin/env python3
"""Shared estimator contract for every fit model.

Composition, not inheritance, is how a model gets its physics constants and
its preprocessing behaviour: every :class:`FitModel` *holds* a
``physics: PhysicsParams`` and a ``preprocessing: PreprocessingPipeline``
rather than inheriting despike/bin/kernel behaviour through mixins. What *is*
inherited is the estimator contract itself - construction, config
introspection, and (for single-trace models) the prepare -> fit -> fit_file
pipeline - because "is a fit model with a params object" is a genuine is-a
relationship. This mirrors sklearn's own split: ``Pipeline`` composes
transforms, ``BaseEstimator`` is inherited for the common contract.

:class:`SingleTraceModel` is the intermediate base for models that fit one
Taylorgram at a time (CRLI, Laplace, parametric). :class:`GlobalSeriesModel`
(``tda.models.global_series``) fits many traces at once and inherits only
:class:`FitModel` - it composes the same physics/preprocessing but has its own
multi-trace entry point instead of ``fit_file``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from pathlib import Path

import numpy as np

from tda.io.trace import ReaderParams, read_trace
from tda.physics import PhysicsParams
from tda.preprocessing.pipeline import PreparedTrace, PreprocessingPipeline


def _as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _as_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


class FitModel(ABC):
    def __init__(self, physics: PhysicsParams | None = None,
                 preprocessing: PreprocessingPipeline | None = None):
        self.physics = physics if physics is not None else PhysicsParams()
        self.preprocessing = preprocessing if preprocessing is not None else PreprocessingPipeline()

    @property
    def temperature_k(self) -> float:
        return self.physics.temperature_k

    def get_params(self) -> dict:
        """This model's fit-option dataclass as a plain dict (sklearn-style)."""
        return dataclasses.asdict(self.params)

    def set_params(self, **updates) -> "FitModel":
        """Return a copy of this model (same physics/preprocessing/reader) with
        ``updates`` applied to its params - never mutates ``self``, so sweeps
        can build many differently-configured instances side by side."""
        new_params = dataclasses.replace(self.params, **updates)
        kwargs = {"params": new_params, "physics": self.physics, "preprocessing": self.preprocessing}
        if hasattr(self, "reader"):
            kwargs["reader"] = self.reader
        return type(self)(**kwargs)

    def config_dict(self) -> dict:
        """JSON-serializable dump of this model's full configuration."""
        return {
            "model": type(self).__name__,
            "params": _as_jsonable(self.params),
            "physics": _as_jsonable(self.physics),
            "preprocessing": {
                "steps": [_as_jsonable(s) for s in self.preprocessing.steps],
                "baseline": _as_jsonable(self.preprocessing.baseline),
            },
        }

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...


class SingleTraceModel(FitModel):
    """Base for models that fit one prepared Taylorgram at a time."""

    def __init__(self, physics: PhysicsParams | None = None,
                 preprocessing: PreprocessingPipeline | None = None,
                 reader: ReaderParams | None = None):
        super().__init__(physics, preprocessing)
        self.reader = reader if reader is not None else ReaderParams()

    def prepare(self, path: str | Path) -> PreparedTrace:
        path = Path(path)
        t, y = read_trace(path, **dataclasses.asdict(self.reader))
        return self.preprocessing.run(t, y, path)

    @abstractmethod
    def fit(self, prepared: PreparedTrace):
        ...

    def fit_file(self, path: str | Path):
        return self.fit(self.prepare(path))

    def config_dict(self) -> dict:
        out = super().config_dict()
        out["reader"] = _as_jsonable(self.reader)
        return out
