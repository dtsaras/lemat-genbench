# Modification Notes for Oxidation-State Assignment, BVlain/HamGNN Readiness, and ESW Integration

This document describes the major changes in the current program compared with the original program. The goals of these modifications are: to add an element-validity check during the validity stage and filter out elements that are not accepted for this task; to complete oxidation-state assignment and BVlain readiness checks in advance when a comprehensive run requests `migration_barrier` or a `property` calculation that includes migration barrier; to read the supported elements from the OpenMX `DFT_DATA19` directory on the current server and perform a HamGNN readiness check when HamGNN `band_gap` or a `property` calculation that includes HamGNN band gap is requested; to add a Li-exchange-only ESW property calculation; and to filter out structures that do not support the requested property calculations while allowing supported structures to continue with subsequent calculations.

## 1. Overall behavior changes

The original program workflow was:

```text
load structures
-> validity preprocessing
   -> forbidden-element validity check
-> filter overall_valid structures
-> remaining preprocessors
-> remaining benchmarks
```

The current program adds an optional property-gating step to the comprehensive runner:

```text
load structures
-> validity preprocessing
   -> structural validity
   -> forbidden-element validity check
   -> optional oxidation-state assignment and BVlain readiness check
-> filter overall_valid structures
-> optional property_gating
   -> optional HamGNN/OpenMX element-support readiness check
   -> keep structures ready for the requested property calculations
   -> filter out non-ready structures
-> remaining preprocessors
   -> optional ESW preprocessing for esw/property runs
-> remaining benchmarks
```

The current default configuration is:

```yaml
property_gating:
  enabled: true
  required_families: ["migration_barrier", "band_gap", "property"]
  failure_policy: filter
  check_hamgnn_elements: true
  hamgnn_dft_data: null
```

Therefore, property gating is enabled only when `--families` contains property families that require readiness checks. The default comprehensive run does not include `migration_barrier`, `band_gap`, or `property`, so it is not affected.

## 2. Semantic notes

This modification expands the definition of `overall_valid`.

- `overall_valid` is now jointly determined by charge neutrality, interatomic distance, physical plausibility, and element validity.
- `element_valid=False` means that the structure contains elements forbidden for this task.
- Oxidation-state assignment, BVlain readiness, and HamGNN readiness are not written into `overall_valid`.
- Readiness is used only as a gating condition before subsequent property calculations.
- Property readiness checks are performed only for structures with `overall_valid=True`, avoiding wasted computation on structures that are already invalid.

The current default `failure_policy: filter` means:

- Structures that support the requested property calculations continue to run all subsequent preprocessors/benchmarks.
- Structures that do not support the requested property calculations are skipped.
- The entire run will not stop because a single structure does not support BVlain or HamGNN/OpenMX.

If this is changed to `abort_run`, then as soon as one valid structure is not ready, all subsequent calculations will be stopped. If it is changed to `warn`, issues will only be recorded, and structures will not be filtered.

## 3. Major file changes

| File | Description of changes |
|---|---|
| `src/lemat_genbench/utils/oxidation_state.py` | Added utility functions for oxidation-state assignment and BVlain readiness. |
| `src/lemat_genbench/utils/hamgnn_readiness.py` | Added utility functions for HamGNN/OpenMX readiness; supported elements are read from the current `DFT_DATA19` installation directory. |
| `src/lemat_genbench/preprocess/validity_preprocess.py` | Added forbidden-element validity checks and optional oxidation-state/readiness recording to `ValidityPreprocessor`. |
| `src/lemat_genbench/properties/migration_barrier.py` | Migration barrier calculation now preferentially uses the oxidation-state record saved during the validity stage. |
| `src/lemat_genbench/properties/esw.py` | Added the core Li-exchange-only ESW calculation, including MACE relaxation, MP entries cache, phase diagram, and Li chemical-potential scan. |
| `src/lemat_genbench/preprocess/esw_preprocess.py` | Added an ESW preprocessor that writes ESW, oxidation/reduction potentials, Ehull, and error information into structure properties. |
| `src/lemat_genbench/metrics/esw_metric.py` | Added ESW aggregation metrics. |
| `src/lemat_genbench/benchmarks/esw_benchmark.py` | Added a standalone `esw` benchmark family. |
| `src/lemat_genbench/benchmarks/property_benchmark.py` | Added optional `include_esw` to the `property` benchmark. |
| `scripts/run_benchmarks.py` | Added `property_gating` parsing, filtering, and result metadata output to the comprehensive runner. |
| `src/lemat_genbench/cli.py` | Added lightweight CLI support for the `esw` benchmark and generation of the default `esw.yaml`. |
| `src/config/comprehensive.yaml` | Enabled property gating by default, set `failure_policy: filter`, and added `esw_settings`. |
| `src/config/esw.yaml` | Added standalone ESW benchmark configuration. |
| `src/config/property.yaml` | Added `include_esw` and ESW parameters to the `property` configuration. |
| `EVALUATION.md` | Documented `property_gating` as property-readiness filtering in the configuration reference. |
| `tests/test_validity_preprocess.py` | Added basic tests for readiness records. |
| `tests/test_hamgnn_readiness.py` | Added tests for HamGNN/OpenMX element-support checks. |
| `tests/test_esw_metric.py` | Added lightweight aggregation tests for ESW metric/benchmark. |

## 4. Element-validity check

A forbidden-element check has been added to the validity stage. The default forbidden elements are:

```python
{
    "He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og",
    "Tc", "Pm", "Po", "At", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Hg",
}
```

If a structure contains any of the above elements:

```python
structure.properties["element_valid"] = False
structure.properties["overall_valid"] = False
```

The following fields are also written:

```python
structure.properties["element_check_details"]
structure.properties["validity_details"]["element_filter"]
```

Example:

```json
{
  "element_valid": false,
  "element_check_details": {
    "valid": false,
    "elements": ["Hg", "O"],
    "forbidden_elements_present": ["Hg"],
    "forbidden_elements": ["He", "Ne", "...", "Hg"]
  }
}
```

Therefore, `overall_valid` now means that both structural validity and the element application domain are satisfied.

## 5. Oxidation-state assignment records

The new core record keys are:

```python
OXIDATION_STATE_RECORD_KEY = "oxidation_state_record"
MIGRATION_BARRIER_READY_KEY = "migration_barrier_ready"
```

For each structure that succeeds or fails the readiness check, the following fields are written into `Structure.properties`:

```python
structure.properties["oxidation_state_record"]
structure.properties["migration_barrier_ready"]
structure.properties["oxidation_state_status"]
structure.properties["oxidation_state_confidence"]
```

The main fields in `oxidation_state_record` are:

```json
{
  "status": "success",
  "selected_source": "bvanalyzer",
  "confidence_label": "medium_confidence",
  "mobile_ion": "Li1+",
  "total_charge": 0,
  "mean_abs_mismatch": 0.12,
  "max_abs_mismatch": 0.45,
  "oxidation_states_by_element": {
    "Li": 1,
    "O": -2
  },
  "oxidation_states_by_site": [
    {
      "site_index": 0,
      "element": "Li",
      "oxi_state": 1
    }
  ],
  "bvlain_parameters_complete": true,
  "migration_barrier_ready": true,
  "warnings": [],
  "failure_reasons": [],
  "candidate_summary": []
}
```

On failure, the failure reason is recorded, for example:

```json
{
  "status": "failed",
  "mobile_ion": "Li1+",
  "migration_barrier_ready": false,
  "failure_reasons": [
    "mobile ion Li not present"
  ]
}
```

## 6. Oxidation-state candidate sources

The current oxidation-state assignment generates candidates from the following sources:

| Source | Description |
|---|---|
| `cif` | If the input structure already contains site oxidation states, they are used preferentially as a candidate. |
| `bvanalyzer` | Uses Pymatgen `BVAnalyzer` to assign oxidation states. |
| `composition_guess` | Uses the existing LeMat ICSD oxidation-state prior to perform composition-level charge-balance guessing. |
| `composition_guess_all_states` | Allows a broader oxidation-state search when the normal candidates fail. |

Candidates are checked for:

- Whether the total charge is close to neutral.
- Whether the mobile ion is present.
- Whether the number of element types satisfies `min_num_elements`.
- Whether the mobile-ion oxidation state matches the requirement, for example `Li1+` requires Li to be +1.
- Common chemical constraints, for example alkali metals are +1, alkaline-earth metals are +2, F is -1, and O should not have a positive oxidation state.
- BVS mismatch statistics.
- If `check_bvlain_parameters` is enabled, BVlain is called to perform the parameter/readiness check.

## 7. HamGNN/OpenMX readiness

The new core record keys are:

```python
HAMGNN_READINESS_RECORD_KEY = "hamgnn_readiness_record"
HAMGNN_READY_KEY = "hamgnn_ready"
```

HamGNN readiness does not use a fixed global table of supported elements. Instead, it reads the OpenMX `DFT_DATA19` directory configured on the current server at runtime:

- It first uses `property_gating.hamgnn_dft_data`.
- If not configured, it reads `band_gap_settings.backend_kwargs.dft_data`.
- If the family is `property`, it also reads `property_settings.band_gap_backend_kwargs.dft_data`.
- If none of the above is configured, it reads the environment variable `HAMGNN_DFT_DATA`.

The check logic is:

```text
elements appearing in DFT_DATA19/VPS
∩ elements appearing in DFT_DATA19/PAO
-> element set supported by the current OpenMX installation
```

If a structure contains elements outside this set, then:

```python
structure.properties["hamgnn_ready"] = False
structure.properties["hamgnn_readiness_record"]["unsupported_elements"] = [...]
```

If the `DFT_DATA19` path is not configured, the path does not exist, or supported elements cannot be read from `VPS`/`PAO`, the structure is also marked as not ready, and the reason is written into `failure_reasons`.

## 8. Changes to `ValidityPreprocessor`

`ValidityPreprocessor` now has the following optional parameters:

```python
forbidden_elements: list[str] | tuple[str, ...] | None = None
assign_oxidation_states: bool = False
mobile_ion: str = "Li1+"
require_mobile_ion: bool = False
migration_min_num_elements: int | None = None
check_bvlain_parameters: bool | str = False
oxidation_charge_tolerance: float = 1e-3
bvlain_settings: Dict[str, Any] | None = None
```

Oxidation-state/readiness checks remain disabled by default, so directly using `ValidityPreprocessor()` will not additionally call BVlain. The forbidden-element check is enabled by default; to disable it, pass `forbidden_elements=[]`.

When the comprehensive runner requires migration/property gating, it passes:

```python
assign_oxidation_states=True
mobile_ion="Li1+"
require_mobile_ion=True
migration_min_num_elements=2
check_bvlain_parameters=True
```

The generated validity final scores additionally include:

```json
{
  "element_validity_count": 95,
  "element_validity_ratio": 0.95,
  "oxidation_state_success_count": 82,
  "oxidation_state_success_ratio": 0.82,
  "migration_barrier_ready_count": 82,
  "migration_barrier_ready_ratio": 0.82
}
```

Here, `element_validity_*` always reflects the validity-stage element check; `oxidation_state_*` and `migration_barrier_ready_*` appear only when oxidation-state/readiness checks are actually enabled.

## 9. Changes to the comprehensive runner

`scripts/run_benchmarks.py` adds two main functions:

```python
get_property_gating_settings(...)
apply_property_gating(...)
```

`get_property_gating_settings` parses the following from the configuration:

- Whether gating is enabled.
- Which families trigger gating.
- Whether the current run requires migration/BVlain readiness.
- Whether the current run requires HamGNN/OpenMX readiness.
- The mobile ion.
- The minimum number of element types required for readiness.
- BVlain parameters.
- The HamGNN `DFT_DATA19` path.
- `failure_policy`.

`apply_property_gating` is executed after validity and before other preprocessors:

```text
valid_structures
-> keep structures ready for all requested property calculations
-> record blocked structures and reasons
-> return filtered structures
```

Under the default `filter` policy, filtered structures will not enter subsequent:

- fingerprint/distribution/stability preprocessors
- `migration_barrier`
- `band_gap`
- `property`
- other remaining benchmarks

This satisfies the requirement to “skip only structures that do not support the requested property calculations, while allowing all other normal structures to continue calculation.”

If only HamGNN `band_gap` is requested in the current run, BVlain readiness is not enabled, and the structure is not required to contain Li. If only `migration_barrier` is requested, the HamGNN/OpenMX element-support check is not enabled. The `property` family decides which readiness checks are needed based on `include_band_gap`, `include_migration_barrier`, and `band_gap_backend`.

## 10. Output changes

The result JSON adds `validity_filtering.property_gating`:

```json
{
  "enabled": true,
  "migration_enabled": true,
  "hamgnn_enabled": true,
  "failure_policy": "filter",
  "mobile_ion": "Li1+",
  "check_bvlain_parameters": true,
  "require_mobile_ion": true,
  "min_num_elements": 2,
  "check_hamgnn_elements": true,
  "hamgnn_dft_data": "/path/to/DFT_DATA19",
  "required_families": ["band_gap", "migration_barrier", "property"],
  "input_valid_structures": 100,
  "ready_structures": 82,
  "blocked_structures": 18,
  "migration_ready_structures": 90,
  "hamgnn_ready_structures": 88,
  "ready_structure_ids": [0, 1, 4],
  "blocked_structure_details": [
    {
      "structure_id": 7,
      "original_source": "bad.cif",
      "oxidation_state_status": "failed",
      "hamgnn_status": "success",
      "failure_reasons": ["bvlain parameter check failed: ..."],
      "warnings": []
    }
  ]
}
```

The result JSON also adds `validity_filtering.oxidation_state_records`:

```json
[
  {
    "structure_id": 0,
    "original_source": "LiFePO4.cif",
    "record": {
      "status": "success",
      "selected_source": "bvanalyzer",
      "mobile_ion": "Li1+",
      "migration_barrier_ready": true,
      "oxidation_states_by_site": []
    }
  }
]
```

When HamGNN readiness is enabled, `validity_filtering.hamgnn_readiness_records` is also added:

```json
[
  {
    "structure_id": 0,
    "original_source": "LiFePO4.cif",
    "record": {
      "status": "success",
      "hamgnn_ready": true,
      "dft_data": "/path/to/DFT_DATA19",
      "elements": ["Fe", "Li", "O", "P"],
      "unsupported_elements": [],
      "supported_element_count": 80,
      "failure_reasons": [],
      "warnings": []
    }
  }
]
```

The terminal summary displays:

```text
Property-ready structures: 82 / 100
Migration-barrier ready: 90 / 100
HamGNN/OpenMX ready: 88 / 100
```

The logs display:

```text
Property gating: 82/100 valid structures are property-ready
Property gating filtered 18 structures before remaining benchmarks.
```

## 11. Changes to migration barrier calculation

The original BVlain calculation in `migration_barrier.py` mainly relied on:

- `oxi_check=True`
- `add_oxidation_state_by_guess`

The current program adds a preferred path:

```text
1. If an oxidation_state_record already exists in structure properties, use the site-wise oxidation states in the record to decorate the structure.
2. If no record exists, temporarily call assign_oxidation_states_for_bvlain.
3. If the first two steps fail, fall back to the original BVlain fallback chain.
```

In this way, oxidation states assigned during the validity stage in a comprehensive run can be reused by the subsequent migration barrier calculation, avoiding duplication and inconsistency.

## 12. Configuration changes

`src/config/comprehensive.yaml` adds:

```yaml
validity_settings:
  forbidden_elements:
    - He
    - Ne
    - Ar
    - Kr
    - Xe
    - Rn
    - Og
    - Tc
    - Pm
    - Po
    - At
    - Fr
    - Ra
    - Ac
    - Th
    - Pa
    - U
    - Np
    - Pu
    - Am
    - Cm
    - Bk
    - Cf
    - Es
    - Fm
    - Md
    - No
    - Lr
    - Hg

property_gating:
  enabled: true
  required_families: ["migration_barrier", "band_gap", "property"]
  failure_policy: filter
  mobile_ion: Li1+
  require_mobile_ion: true
  min_num_elements: 2
  check_bvlain_parameters: true
  oxidation_charge_tolerance: 0.001
  check_hamgnn_elements: true
  hamgnn_dft_data: null

migration_barrier_settings:
  mobile_ion: Li1+
  dimensionality: 3d
  r_cut: 10.0
  resolution: 0.2
  k: 100
  encut: 5.0
  fast_threshold: 0.6
  n_jobs: 1
  timeout: 30
```

To change to another strategy later, only modify:

```yaml
failure_policy: warn
```

or:

```yaml
failure_policy: abort_run
```

`hamgnn_dft_data: null` means that the path is not hard-coded in the configuration. At runtime, the program uses `band_gap_settings.backend_kwargs.dft_data`, `property_settings.band_gap_backend_kwargs.dft_data`, or the environment variable `HAMGNN_DFT_DATA`.

## 13. Testing and validation

The following test files have been added:

```text
tests/test_validity_preprocess.py
tests/test_hamgnn_readiness.py
tests/test_esw_metric.py
```

They cover:

- When oxidation-state assignment is enabled, the validity preprocessor can write `oxidation_state_record`.
- Li-containing structures such as LiF obtain `migration_barrier_ready=True`.
- Si structures without Li obtain `migration_barrier_ready=False` and record the failure reason.
- Hg-containing structures with a forbidden element obtain `element_valid=False` and `overall_valid=False`.
- A pure Li structure contains Li, but because the number of element types is less than 2, it obtains `migration_barrier_ready=False`.
- HamGNN readiness reads supported elements from simulated `DFT_DATA19/VPS` and `DFT_DATA19/PAO` directories.
- When a structure contains elements not supported by the current OpenMX installation, it obtains `hamgnn_ready=False` and records `unsupported_elements`.
- The ESW metric correctly aggregates valid/missing ESW values.
- The ESW target window uses `fraction_in_target_window` as the primary metric.
- The ESW benchmark can read existing `structure.properties["esw"]` when `preprocess=False`.

Static validation has been executed:

```text
python -m py_compile ...
git diff --check
```

These checks passed.

The following pytest command has been executed:

```text
uv run pytest tests/test_esw_metric.py tests/test_validity_preprocess.py tests/test_hamgnn_readiness.py
```

The result was `9 passed`. The tests did not run real MACE relaxation or MP API calls; they only covered lightweight aggregation and readiness logic.

## 14. Compatibility notes

- When running a normal validity benchmark directly, oxidation-state assignment is not enabled by default, but forbidden-element validity checks are enabled.
- To fully restore the original element behavior, construct `ValidityPreprocessor(forbidden_elements=[])`.
- When running `lemat-genbench ... migration_barrier` alone, there is no comprehensive validity stage, but migration barrier internally still performs temporary oxidation-state assignment and retains the original fallback.
- In a comprehensive run, property gating is enabled only when `migration_barrier`, `band_gap`, or `property` is requested and the configuration requires it.
- HamGNN readiness is enabled only when band gap is requested with the HamGNN backend. If the band gap backend is changed to `alignn`, the OpenMX `DFT_DATA19` element-support check is not performed.
- `overall_valid` is affected by forbidden-element validity, but not by BVlain/HamGNN readiness, so that “element application domain” and “whether a specific property can be calculated” are not mixed together.

## 15. ESW integration

A new `esw` benchmark family has been added to calculate the Li-exchange-only electrochemical stability window. This implementation follows the definition in `esw_ea_calc.py`:

```text
ESW = widest continuous relative Li chemical-potential interval
      where abs(Li_exchange) <= gpd_stability_tol
```

The current ESW does not include a reaction-energy / grand-potential e_above_hull threshold, so it should be interpreted as a Li-exchange-only ESW.

The ESW calculation workflow is:

```text
input structure
-> MACE relaxation
-> relaxed structure + MACE energy
-> MP entries for chemical system
-> PhaseDiagram + target ComputedStructureEntry
-> GrandPotentialPhaseDiagram Li chemical-potential scan
-> esw / reduction_potential / oxidation_potential / ehull
```

The default device strategy is:

```yaml
device: auto
```

This means that if `torch.cuda.is_available()` is true, `cuda` is used; otherwise, `cpu` is used.

Each structure is written with:

```python
structure.properties["esw"]
structure.properties["reduction_potential"]
structure.properties["oxidation_potential"]
structure.properties["ehull"]
structure.properties["esw_error"]
structure.properties["esw_used_relaxed_structure"]
structure.properties["esw_mace_energy"]
structure.properties["esw_relaxed_structure_path"]
structure.properties["esw_stable_mu_min"]
structure.properties["esw_stable_mu_max"]
```

Aggregated outputs include:

```text
mean_esw
median_esw
min_esw
max_esw
fraction_esw_valid
fraction_in_target_window  # appears only after target_min/target_max are configured
```

The default configuration is located at:

```yaml
esw_settings:
  preprocess: true
  cache_dir: data/esw_cache
  api_key: null
  mace_model: null
  device: auto
  default_dtype: float64
  fmax: 0.05
  max_steps: 500
  mu_min: -5.0
  mu_max: 0.0
  mu_step: 0.01
  gpd_stability_tol: 0.0001
```

Where:

- `api_key: null` means the environment variable `MP_API_KEY` is read at runtime.
- `mace_model: null` means the environment variable `MACE_MODEL_PATH` is read at runtime; if it is not set, the default MACE model path in the original script is used.
- MP entries and MACE relaxation results are cached by default to `data/esw_cache`.
- If MACE relaxation, MP entries retrieval, or ESW calculation fails, the entire run will not crash; the structure is written with `esw=None` and `esw_error`, and the success ratio is reflected by `fraction_esw_valid` during aggregation.

The `property` benchmark adds:

```yaml
include_esw: false
```

Therefore, the default `property` behavior remains band gap + migration barrier. To run ESW at the same time, it must be explicitly set to true.
