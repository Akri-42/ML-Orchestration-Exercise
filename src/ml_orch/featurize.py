"""Featurization, and the version that binds it to a model.
`fit` learns two things from the training chunk: the mean and standard
deviation of each descriptor, and which fingerprint bits were ever set. Both
are then baked into `version`. Now a model scored with a differently-fitted
featurizer produces genuinely different numbers, `FeaturizerBindingGate` has
something real to catch, and the train/serve skew this system exists to prevent
is a failure that can actually occur.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import SMILES_COLUMN

# A fixed, ordered, named descriptor set. Order is part of the contract: the
# feature matrix is positional, so reordering this list silently invalidates
# every model ever trained. It is therefore hashed into the version.
DESCRIPTORS: tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NumAromaticRings", "NumSaturatedRings", "NumHeteroatoms",
)


class Featurizer(Protocol):
    """Pipeline 1 fits and transforms. Pipeline 2 only ever transforms — with
    the instance loaded from the promoted model's artifact."""

    kind: str

    @property
    def version(self) -> str: ...

    def fit(self, frame: pd.DataFrame) -> Featurizer: ...

    def transform(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FitState:
    """What `fit` learned. Present iff the featurizer has been fitted."""

    descriptor_mean: np.ndarray = field(repr=False)
    descriptor_std: np.ndarray = field(repr=False)
    active_bits: np.ndarray = field(repr=False)     # indices of bits ever set
    n_train_rows: int

    def fingerprint(self) -> str:
        """A hash of the fitted state, so `version` changes when the fit does."""
        h = hashlib.sha256()
        for arr in (self.descriptor_mean, self.descriptor_std, self.active_bits):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(self.n_train_rows).encode())
        return h.hexdigest()[:12]


@dataclass
class RdkitDescriptorFeaturizer:
    

    morgan_bits: int = 1024
    morgan_radius: int = 2
    include_descriptors: bool = True
    kind: str = "rdkit_descriptors"
    state: FitState | None = field(default=None, repr=False)

    # -- identity ----------------------------------------------------------

    def params(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "morgan_bits": self.morgan_bits,
            "morgan_radius": self.morgan_radius,
            "include_descriptors": self.include_descriptors,
            "descriptors": list(DESCRIPTORS) if self.include_descriptors else [],
        }

    @property
    def version(self) -> str:
        """`kind@<params hash>/<fit hash>`.

        Unfitted featurizers get `/unfitted`, which will never match a model's
        recorded binding — so scoring with one fails the gate rather than
        quietly producing unscaled features.
        """
        payload = json.dumps(self.params(), sort_keys=True, separators=(",", ":")).encode()
        p = hashlib.sha256(payload).hexdigest()[:12]
        f = self.state.fingerprint() if self.state is not None else "unfitted"
        return f"{self.kind}@{p}/{f}"

    @property
    def fitted(self) -> bool:
        return self.state is not None

    # -- computation -------------------------------------------------------

    def _raw(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(descriptors, fingerprint bits) before any fitted transformation."""
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius, fpSize=self.morgan_bits)
        calc = {name: getattr(Descriptors, name) for name in DESCRIPTORS}

        smiles = frame[SMILES_COLUMN].tolist()
        desc = np.zeros((len(smiles), len(DESCRIPTORS)), dtype=float)
        bits = np.zeros((len(smiles), self.morgan_bits), dtype=np.uint8)

        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if mol is None:
                # Validation should have removed these. Reaching here means the
                # caller skipped it, so produce zeros rather than raising: a
                # single bad molecule must not kill a batch, and the row is
                # already flagged upstream.
                continue
            for j, name in enumerate(DESCRIPTORS):
                desc[i, j] = calc[name](mol)
            bits[i] = np.frombuffer(
                gen.GetFingerprintAsNumPy(mol).astype(np.uint8).tobytes(), dtype=np.uint8)
        return desc, bits

    def fit(self, frame: pd.DataFrame) -> RdkitDescriptorFeaturizer:
        desc, bits = self._raw(frame)
        std = desc.std(axis=0)
        std[std == 0] = 1.0                       # constant descriptor -> no scaling
        active = np.flatnonzero(bits.any(axis=0))
        self.state = FitState(
            descriptor_mean=desc.mean(axis=0),
            descriptor_std=std,
            active_bits=active,
            n_train_rows=len(frame),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError(
                "featurizer is not fitted — refusing to transform. Load the "
                "featurizer bound to the model instead of constructing one.")
        desc, bits = self._raw(frame)
        parts = []
        if self.include_descriptors:
            parts.append((desc - self.state.descriptor_mean) / self.state.descriptor_std)
        parts.append(bits[:, self.state.active_bits].astype(float))
        return np.hstack(parts) if parts else np.zeros((len(frame), 0))

    @property
    def n_features(self) -> int:
        if self.state is None:
            raise RuntimeError("featurizer is not fitted")
        n = len(self.state.active_bits)
        return n + (len(DESCRIPTORS) if self.include_descriptors else 0)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh)
        return p

    @classmethod
    def load(cls, path: str | Path) -> RdkitDescriptorFeaturizer:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


def featurizer_from_config(cfg: dict[str, Any]) -> RdkitDescriptorFeaturizer:
    """`featurizer:` block -> a Featurizer.

    `kind: from_promoted_model` is deliberately *not* handled here. Pipeline 2
    must load the featurizer out of the promoted model's artifact rather than
    build one from config, and routing that through this factory would make the
    skew bug reachable again. The caller resolves it; see `models.load_bound`.
    """
    kind = cfg.get("kind")
    if kind == "from_promoted_model":
        raise ValueError(
            "featurizer kind 'from_promoted_model' must be resolved from the "
            "registry, not constructed from config — see models.load_bound")
    if kind != "rdkit_descriptors":
        raise ValueError(f"unknown featurizer kind {kind!r}")
    params = cfg.get("params", {})
    return RdkitDescriptorFeaturizer(
        morgan_bits=int(params.get("morgan_bits", 1024)),
        morgan_radius=int(params.get("morgan_radius", 2)),
        include_descriptors=bool(params.get("include_descriptors", True)),
    )


# ---------------------------------------------------------------------------
# Slice columns
# ---------------------------------------------------------------------------


def derive_slice_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the slice gate slices on.

    Derived here rather than written into the chunk files on purpose: a chunk
    arriving from an assay would carry structures and labels, not our
    bookkeeping. Keeping derivation in the pipeline means the arriving-data
    contract stays honest, and the slice definitions can change without
    reissuing the data.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    out = frame.copy()
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None
            for s in frame[SMILES_COLUMN]]
    out["n_heavy_atoms"] = [m.GetNumHeavyAtoms() if m else 0 for m in mols]
    out["has_F"] = [
        bool(m and any(a.GetSymbol() == "F" for a in m.GetAtoms())) for m in mols
    ]
    out["n_rings"] = [m.GetRingInfo().NumRings() if m else 0 for m in mols]
    return out


def slice_masks(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """`evaluation.slices:` config -> named boolean masks over the frame.

    Bins become one slice per interval; boolean columns become two slices. Any
    slice smaller than a handful of rows is dropped rather than reported, since
    a MAE over three molecules is noise wearing a number's clothes.
    """
    masks: dict[str, np.ndarray] = {}
    for name, cfg in (spec or {}).items():
        column = cfg.get("column", name)
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy()
        if "bins" in cfg:
            edges = list(cfg["bins"])
            for lo, hi in pairwise(edges):
                masks[f"{name}[{lo}-{hi})"] = (values >= lo) & (values < hi)
        else:
            masks[f"{name}=true"] = values.astype(bool)
            masks[f"{name}=false"] = ~values.astype(bool)
    return {k: m for k, m in masks.items() if m.sum() >= 20}


def featurize_targets(frame: pd.DataFrame, targets: Sequence[str]) -> np.ndarray:
    """Target matrix, selected **by name**. QM9's CSV order is not task order."""
    missing = [t for t in targets if t not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing target column(s): {missing}")
    return frame[list(targets)].to_numpy(dtype=float)
