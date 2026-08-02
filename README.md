# Stereo CPU-PIV Processing (raw im7 input, DaVis dewarping)

No-GPU counterpart to
[`Stereo_PIV_GPU`](https://github.com/SPC3MN/Stereo_PIV_GPU): processes
raw stereo `.im7` buffers from one or more LaVision/DaVis sets, dewarps
each camera's images onto a shared world grid using the same DaVis
calibration polynomial, runs plain
[`openpiv-python`](https://github.com/OpenPIV/openpiv-python)'s own
multi-pass, window-deformation pipeline (`openpiv.windef`, the same
coarse-to-fine multi-pass + validation + outlier-replacement +
optional-smoothing approach as `piv_gpu`) on each camera's dewarped pair
instead of `piv_gpu`, then combines the two in-plane displacement fields
into 3-component (U, V, W) stereo velocity using the same reconstruction
math as the GPU pipeline. No CUDA, no GPU, nothing beyond a normal Python
+ pip install.

Treat it as a no-GPU fallback or a CPU cross-check against
`Stereo_PIV_GPU`, not a byte-for-byte replacement -- it's the same
multi-pass *idea*, but two different implementations of it
(`openpiv-python` vs. `openpiv-python-gpu`), so exact numeric agreement
isn't guaranteed.

## ⚠️ Before trusting any output

Same calibration/geometry caveats as `Stereo_PIV_GPU` apply here --
`cam0_mapping`/`cam1_mapping`, `alpha1_deg`/`alpha2_deg`/`beta1_deg`/
`beta2_deg`, and `stereo_frame_order` are all placeholders/assumptions
that need confirming against your own calibration and rig before trusting
W (or U/V). See that repo's README for the full discussion.

## What it does

- Reads raw stereo `.im7` images directly via `lvpyio` in one of two
  `input_mode`s:
  - `"set"` -- point `input_path` at either a **single** DaVis image set,
    or a **folder containing several `*.set` entries**, in which case
    every set inside is batch-processed in turn into its own subfolder of
    `output_dir` (see `piv_common.resolve_set_paths()`)
  - `"loose"` -- a plain folder of standalone `.im7` files, auto-detecting
    whether each file already combines both cameras' 4 exposures, or each
    camera's double-frame pair is a separate file matched by
    `suffix_cam0`/`suffix_cam1`
- **Single-set preview:** when `input_path` resolves to exactly one set
  (not a folder of several), the first pair's 3-component velocity field
  is computed, plotted, and opened for review -- the run pauses on a
  terminal `y/N` prompt before processing the rest of that set. Skipped
  entirely in folder-of-sets batch mode and in `"loose"` mode.
- Dewarps each camera's raw images onto a shared world grid using DaVis's
  own 3rd-order polynomial mapping (`CameraMapping`), caching the coordinate
  grid per camera so it's only computed once per run, not once per frame
- Runs `openpiv-python`'s multi-pass pipeline on each camera's dewarped
  pair independently -- coarse-to-fine window sizes with image
  deformation between passes, per-pass sig2noise/global/median
  validation, iterative outlier replacement, and optional `smoothn`
  smoothing (all via `openpiv.settings.PIVSettings`) -- then the same
  outer post-processing options as the GPU pipeline (outlier rejection,
  invalid vector interpolation, smoothing) run on top
- Combines the two cameras' in-plane displacement fields into 3-component
  (U, V, W) displacement via the same least-squares stereo reconstruction
  used by `Stereo_PIV_GPU`
- Saves results per pair as `.npz` (and optionally a stereo quiver plot
  colored by W), plus an optional CSV summary across the batch

## Files

| File | Purpose |
|---|---|
| `CPU_Stereo_Processing.py` | Entry point -- run this |
| `piv_common.py` | Shared config loading, post-processing, GPU/CPU PIV engine adapters, plain im7 frame iteration, set-folder resolution, preview/confirm prompt |
| `stereo_common.py` | Stereo-specific helpers -- `CameraMapping`/dewarping, stereo frame extraction, `reconstruct_stereo`, stereo quiver plot |

## Requirements

- Python 3.9+
- No GPU or CUDA Toolkit needed
- [`openpiv-python`](https://github.com/OpenPIV/openpiv-python) (`pip install openpiv`)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration file

All pipeline settings live in a JSON file -- `stereo_cpu_piv_config.json`
next to `CPU_Stereo_Processing.py` by default, or pass a different path as
the first argument: `python CPU_Stereo_Processing.py my_config.json`. On
first run, if that file doesn't exist, the script writes one out populated
with its built-in defaults and proceeds using them. You only need to
include the keys you're actually changing in the file.

## Usage

1. Run `python CPU_Stereo_Processing.py` once to generate
   `stereo_cpu_piv_config.json` with default values.
2. Fill in `cam0_mapping` / `cam1_mapping` with both cameras' real DaVis
   calibration report coefficients, and the stereo geometry angles
   (`alpha1_deg`/`alpha2_deg`, `beta1_deg`/`beta2_deg`) -- see the warning
   above.
3. Set `input_mode`/`input_path` to point at your stereo set, a folder of
   several stereo sets, or a loose folder, and confirm `stereo_frame_order`
   (or `suffix_cam0`/`suffix_cam1` in loose mode).
4. Edit the rest of the config file, then run:

   ```bash
   python CPU_Stereo_Processing.py
   ```

### Key settings (`stereo_cpu_piv_config.json`)

| Setting | Description |
|---|---|
| `input_mode` | `"set"` (DaVis image set(s)) or `"loose"` (plain folder of `.im7` files) |
| `input_path` | Raw stereo `.im7` source -- a single `.set` file/set folder, a folder containing multiple `*.set` entries, or a plain folder (`"loose"` mode) |
| `stereo_frame_order` | `"camera_major"` (default) or `"frame_major"` -- see `Stereo_PIV_GPU`'s README |
| `suffix_cam0` / `suffix_cam1` | (`"loose"` mode only) filename suffixes used to pair each camera's file |
| `cam0_mapping` / `cam1_mapping` | Each camera's DaVis calibration polynomial coefficients |
| `world_shape` / `world_scale_px_per_mm` / `dewarp_order` | Shared dewarped output grid geometry |
| `cpu_settings` | Keys are `openpiv.settings.PIVSettings` field names, forwarded per camera -- `windowsizes`/`overlap` (one entry per pass; overlap in **pixels**, not a ratio), `dt`, `sig2noise_method`/`sig2noise_threshold`/`sig2noise_validate`, `validation_first_pass`, `replace_vectors`, `filter_method`/`max_filter_iteration`/`filter_kernel_size`, `smoothn`/`smoothn_p`, `deformation_method`, `interpolation_order`, and more (see `PIVSettings`). Default: one pass at 64px/50% overlap, then three passes at 32px/75% overlap. Unrecognized keys are warned about, not silently dropped. |
| `global_outlier_std` | Reject vectors more than N standard deviations from the mean (`None` disables) |
| `replace_invalid` | Interpolate over invalid/NaN vectors, per camera, before combining |
| `smooth_field` / `smooth_sigma` | Gaussian-smooth each camera's field before combining |
| `alpha1_deg` / `alpha2_deg` / `beta1_deg` / `beta2_deg` | Stereo viewing angles used in the U/V/W reconstruction |
| `frame_dt_s` | s between frames; `None` keeps displacement units instead of velocity |
| `apply_v_sign_flip` | Flip the sign of each camera's `v` before combining |
| `save_npz` / `save_plot` / `save_summary_csv` | Which output artifacts to write |

## Output

Same layout as `Stereo_PIV_GPU`: per-pair `<pair_id>_stereo_velocity.npz`
and (optionally) `<pair_id>_stereo_quiver.png` in `output_dir` (or
`output_dir/<set_name>` in batch mode), plus an optional
`stereo_processing_summary.csv` for the whole batch.
