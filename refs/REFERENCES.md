# Pinned references — ML Orchestration take-home

Captured 2026-08-21. Verified reachable from the build machine, not from the Cowork sandbox (see "Sandbox egress" below).

## 1. QM9 / MoleculeNet (DeepChem)

- Docs: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
- Loader source: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/qm9_datasets.py

Authoritative task list (order matters — this is the column order the loader returns):

```python
QM9_TASKS = ["mu", "alpha", "homo", "lumo", "gap", "r2",
             "zpve", "cv", "u0", "u298", "h298", "g298"]
```

Units, for the metric-aggregation policy (§3.2 of the brief — these are why naive mean-MAE is meaningless):

| Task | Quantity | Unit |
|---|---|---|
| `mu` | dipole moment | Debye |
| `alpha` | isotropic polarizability | Bohr³ |
| `homo` | HOMO energy | Hartree |
| `lumo` | LUMO energy | Hartree |
| `gap` | HOMO–LUMO gap | Hartree |
| `r2` | electronic spatial extent | Bohr² |
| `zpve` | zero-point vibrational energy | Hartree |
| `cv` | heat capacity @ 298.15 K | cal/(mol·K) |
| `u0` | internal energy @ 0 K | Hartree |
| `u298` | internal energy @ 298.15 K | Hartree |
| `h298` | enthalpy @ 298.15 K | Hartree |
| `g298` | free energy @ 298.15 K | Hartree |

Loader signature (DeepChem master):

```python
load_qm9(featurizer=dc.feat.CoulombMatrix(29), splitter='random',
         transformers=['normalization'], reload=True,
         data_dir=None, save_dir=None, **kwargs)
```

Raw archive: `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz`

**Decision:** do *not* take a DeepChem dependency. It pulls a heavy stack and its default `CoulombMatrix` featurizer is not what we want. Use DeepChem only as the *source of the CSV and the task/unit contract*; featurize with RDKit descriptors. This keeps §8's environment-debugging tripwire from firing on day one.

**Known trap (brief §6):** `u0 / u298 / h298 / g298` are dominated by molecule size and near-trivially predictable from atom counts. Either report atomization energies or explicitly call this out — otherwise the MAE looks spectacular for the wrong reason.

## 2. Chemprop

- Repo: https://github.com/chemprop/chemprop
- Version at capture: **2.3.1**, commit `b904ada665a3a712bbc193cf66ef9d12b4e42c82` (2026-08-21)
- Requires Python `>=3.11,<3.15`

Package surface worth knowing:

```
chemprop/{data, featurizers, models, nn, uncertainty, callbacks, utils, cli}
chemprop/cli/{train, predict, fingerprint, hpopt, convert}.py
```

Two facts that matter for orchestration:

1. **It has a real CLI** (`chemprop train`, `chemprop predict`). That makes it a drop-in *subprocess step* behind the `Trainer` / `Predictor` primitives — a second `ModelBackend` implementation with no change to the DAG. Good material for the "how would this generalize" write-up.
2. **It ships `chemprop/uncertainty`.** If the answer to open question #9 (does the shortlist gate care about uncertainty / applicability domain?) is yes, this is the natural backend for an uncertainty-aware `Gate` — and for the active-learning P3 story.

**Decision:** sklearn is the default backend. Chemprop stays *designed for but not implemented*, unless the time budget turns out to be generous. Wiring it is a §8 tripwire ("swapping model class to improve MAE") unless it's explicitly serving the reusable-backend argument.

## 3. scikit-learn

Default modelling backend. `RandomForestRegressor` or `HistGradientBoostingRegressor` wrapped in `MultiOutputRegressor` (or one model per target — decide when writing `Trainer`). Seconds per fit on 3–5k molecules, deterministic under a fixed `random_state`, no build toolchain. Featurization: RDKit descriptors + Morgan fingerprints.



## Sandbox egress (why the data isn't already here)

This Cowork container's network is allowlisted. Verified 2026-08-21:

| Host | Result |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | reachable |
| `raw.githubusercontent.com`, `git clone github.com` | reachable |
| `*.s3.amazonaws.com` (DeepChem data bucket) | **403 / blocked** |
| `figshare.com` | **blocked** |
| `codeload.github.com` tarballs | **403** |

So QM9 itself cannot be pulled here. `scripts/fetch_qm9.py` does it on the build machine, which has open network. Nothing about this affects the design; it only means data acquisition is the first thing to run locally.
