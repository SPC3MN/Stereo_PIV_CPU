"""
Batch stereo CPU-PIV pipeline -- im7 input via lvpyio, dewarped via DaVis's
own calibration polynomial, processed with plain openpiv-python
(https://github.com/OpenPIV/openpiv-python)
================================================================================
No-GPU counterpart to Stereo-PIV.py: reads LaVision .im7 stereo images
directly, dewarps each camera's images onto a shared world grid using the
same DaVis calibration polynomial (see stereo_common.CameraMapping), runs
openpiv-python's basic single-pass `pyprocess.extended_search_area_piv` on
each camera's dewarped pair (instead of piv_gpu), then combines the two
in-plane displacement fields into 3-component (U, V, W) velocity via the
same stereo_common.reconstruct_stereo() used by the GPU pipeline. Use this
when CUDA/cupy/openpiv_gpu aren't available, or as a CPU cross-check
against the GPU result.

Requires: pip install lvpyio openpiv

CONFIG FILE
-----------
Every setting below lives in a JSON file -- CONFIG_PATH, default
"stereo_cpu_piv_config.json" next to this script (or pass a different path
as argv[1]: `python CPU_Stereo_Processing.py my_config.json`). If that
file doesn't exist yet, load_controls() writes one out populated with
DEFAULT_CONFIG's values and proceeds using them. You only need to include
the keys you're actually changing -- anything missing from the file falls
back to DEFAULT_CONFIG. See Stereo-PIV.py's module docstring for the full
discussion of the calibration/geometry placeholders (cam0_mapping /
cam1_mapping / stereo_frame_order / alpha*_deg / beta*_deg) -- they carry
over unchanged here.

INPUT_MODE options (set in the config file):
  "set"   -- point input_path at either a single DaVis stereo image set,
             or a folder containing multiple *.set entries to batch
             through -- see piv_common.resolve_set_paths().
  "loose" -- a plain folder of standalone .im7 files -- see
             stereo_common.iter_stereo_from_loose_files(). Always treated
             as a single run (no folder-of-sets batching).

SINGLE-SET PREVIEW
-------------------
When input_mode="set" and input_path points at exactly one set (not a
folder of several), the FIRST pair's 3-component velocity field is
computed, plotted, and opened for review before the rest of that set is
processed -- see piv_common.preview_first_snapshot(). Declining at the
prompt aborts the run. This step is skipped in folder-of-sets batch mode
and in "loose" mode.
"""

import os
import sys
import csv
import numpy as np

import piv_common as pc
import stereo_common as sc


# ======================================================================
# Config -- all pipeline settings, defaulted here and overridable via a
# JSON file (see load_controls() and the CONFIG FILE note above)
# ======================================================================
CONFIG_PATH = "stereo_cpu_piv_config.json"

DEFAULT_CONFIG = {
    # ---------------- Input source ----------------
    "input_mode": "set",                     # "set" or "loose"
    "input_path": "D:\\messy_data\\Stereo\\6-12_5.set",  # .set file / set folder / plain folder / folder-of-sets

    "multiset_index": 0,
    "stereo_frame_order": "camera_major",
    "suffix_cam0": "_cam1.im7",
    "suffix_cam1": "_cam2.im7",
    "loose_glob": "*.im7",

    # ---------------- Calibration mappings ----------
    # Same DaVis calibration report coefficients as Stereo-PIV.py's
    # DEFAULT_CONFIG -- see that file's module docstring for the
    # calibration-time/plate identifiers and caveats. Fields match
    # stereo_common.CameraMapping.__init__ exactly.
    "cam0_mapping": {
        "x0": 2806.99, "x_span": 4096.00, "y0": 1387.18, "y_span": 3008.00,
        "dx_coefs": {"1": 882.1674, "s": 629.5431, "s2": -74.6835, "s3": -4.4885,
                     "t": -0.6616, "t2": 0.2021, "t3": -0.0545,
                     "st": -0.6915, "s2t": -0.0594, "t2s": -0.1322},
        "dy_coefs": {"1": 19.4802, "s": 17.2524, "s2": 1.4413, "s3": -0.0423,
                     "t": 65.1278, "t2": -0.3800, "t3": -0.5897,
                     "st": -76.3895, "s2t": -3.7352, "t2s": -0.2700},
        "name": "cam0 (Plane 1)",
    },
    "cam1_mapping": {
        "x0": 2806.99, "x_span": 4119.58, "y0": 1387.18, "y_span": 3025.32,
        "dx_coefs": {"1": 846.8601, "s": 633.6056, "s2": -75.5333, "s3": -4.8160,
                     "t": -1.0925, "t2": -0.0019, "t3": 0.6212,
                     "st": -0.6421, "s2t": -0.2899, "t2s": -0.6121},
        "dy_coefs": {"1": 19.2035, "s": 16.9346, "s2": 0.6940, "s3": 0.3712,
                     "t": 67.3521, "t2": -0.5134, "t3": -0.1081,
                     "st": -76.9309, "s2t": -4.0639, "t2s": -0.3362},
        "name": "cam1 (Plane 2)",
    },

    "world_shape": (3067, 5874),
    "world_scale_px_per_mm": 17.92,
    "dewarp_order": 1,

    # ---------------- Output ----------------
    "output_dir": "stereo_piv_output_cpu",

    # ---------------- PIV window size / passes / core settings ----------
    # Forwarded to piv_common.CPUPIVProcess(frame_shape, **cpu_settings)
    # PER CAMERA, which drives openpiv-python's own multi-pass,
    # window-deformation pipeline (openpiv.windef, via an
    # openpiv.settings.PIVSettings object) -- the same coarse-to-fine
    # multi-pass + validation + outlier-replacement + optional-smoothing
    # feature set as piv_gpu. Keys here are PIVSettings field names -- see
    # openpiv.settings.PIVSettings for the full list. Unknown keys are
    # warned about, not silently dropped.
    #
    # Default schedule: one pass at 64px/50% overlap (32px overlap), then
    # three passes at 32px/75% overlap (24px overlap) -- windowsizes/
    # overlap must be the same length (one entry per pass); overlap here
    # is in PIXELS, not a ratio.
    "cpu_settings": {
        "windowsizes": [64, 32, 32, 32],
        "overlap": [32, 24, 24, 24],
        "dt": 1.0,
        "correlation_method": "circular",
        "subpixel_method": "gaussian",
        "deformation_method": "symmetric",
        "interpolation_order": 3,
        "sig2noise_method": "peak2mean",
        "sig2noise_threshold": 1.05,
        "sig2noise_validate": True,
        "validation_first_pass": True,
        "replace_vectors": True,
        "filter_method": "localmean",
        "max_filter_iteration": 4,
        "filter_kernel_size": 2,
        "smoothn": False,
        "smoothn_p": 0.05,
    },

    # ---------------- Per-camera post-processing (before combining) -------
    "global_outlier_std": None,
    "replace_invalid": True,
    "smooth_field": False,
    "smooth_sigma": 1.0,

    # ---------------- Stereo geometry (see Stereo-PIV.py docstring) -------
    "alpha1_deg": -89.53 / 2,
    "alpha2_deg": 89.53 / 2,
    "beta1_deg": 0.0,
    "beta2_deg": 0.0,

    # ---------------- Units ----------------
    "frame_dt_s": None,
    "apply_v_sign_flip": True,

    # ---------------- Output artifacts ----------------
    "save_npz": True,
    "save_plot": True,
    "save_summary_csv": True,
    "plot_dpi": 150,
    "quiver_scale": 1000,
    "show_plots": False,

    "verbose": True,
}


class CONTROLS:
    """Populated at runtime by load_controls() -- see DEFAULT_CONFIG and
    the CONFIG FILE note in the module docstring above."""
    pass


def _fixup_stereo_controls(config, ctrl):
    ctrl.world_shape = tuple(ctrl.world_shape)
    ctrl.cam0_mapping = sc.CameraMapping(**config["cam0_mapping"])
    ctrl.cam1_mapping = sc.CameraMapping(**config["cam1_mapping"])


def load_controls(config_path):
    return pc.load_controls(config_path, DEFAULT_CONFIG, CONTROLS, on_loaded=_fixup_stereo_controls)


# ======================================================================
# Per-camera / per-pair processing
# ======================================================================
def run_camera(frame_a, frame_b, ctrl):
    process, x, y = pc.init_cpu_processor(frame_a.shape, ctrl.cpu_settings)
    u, v, valid, elapsed = pc.process_frames(process, frame_a, frame_b, ctrl, report_gpu_mem=False)
    return u, v, valid, elapsed, x, y


def handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1, ctrl, angles, output_dir):
    if ctrl.verbose:
        print(f"Processing {pair_id} ...", end=" ", flush=True)

    alpha1, alpha2, beta1, beta2 = angles
    u1, v1, valid1, elapsed1, x, y = run_camera(dw_a0, dw_b0, ctrl)
    u2, v2, valid2, elapsed2, _, _ = run_camera(dw_a1, dw_b1, ctrl)
    valid = valid1 & valid2
    elapsed = elapsed1 + elapsed2

    # world grid is in pixels at ctrl.world_scale_px_per_mm px/mm
    u1_mm, v1_mm, u2_mm, v2_mm = (a / ctrl.world_scale_px_per_mm
                                   for a in (u1, v1, u2, v2))

    U, V, W = sc.reconstruct_stereo(u1_mm, v1_mm, u2_mm, v2_mm,
                                     alpha1, alpha2, beta1, beta2)

    if ctrl.frame_dt_s is not None:
        U, V, W = (a / ctrl.frame_dt_s for a in (U, V, W))

    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    W = np.where(valid, W, np.nan)

    n_valid, n_total = int(valid.sum()), int(valid.size)
    if ctrl.verbose:
        print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors")

    if ctrl.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_stereo_velocity.npz"),
                 x=x, y=y, U=U, V=V, W=W, valid=valid)

    if ctrl.save_plot:
        sc.plot_and_save_stereo(x, y, U, V, W, valid,
                                 os.path.join(output_dir, f"{pair_id}_stereo_quiver.png"),
                                 ctrl, title=f"CPU Stereo PIV -- {pair_id}")

    row = (pair_id, elapsed, n_valid, n_total)
    return row, x, y, U, V, W, valid


def process_pairs(pair_source, ctrl, angles, output_dir, interactive_preview):
    summary_rows = []
    for idx, (pair_id, fa0, fb0, fa1, fb1) in enumerate(pair_source):
        dw_a0 = ctrl.cam0_mapping.dewarp_image(fa0, ctrl.world_shape, ctrl.dewarp_order)
        dw_b0 = ctrl.cam0_mapping.dewarp_image(fb0, ctrl.world_shape, ctrl.dewarp_order)
        dw_a1 = ctrl.cam1_mapping.dewarp_image(fa1, ctrl.world_shape, ctrl.dewarp_order)
        dw_b1 = ctrl.cam1_mapping.dewarp_image(fb1, ctrl.world_shape, ctrl.dewarp_order)

        row, x, y, U, V, W, valid = handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1,
                                                  ctrl, angles, output_dir)
        summary_rows.append(row)

        if idx == 0 and interactive_preview:
            preview_path = os.path.join(output_dir, f"{pair_id}_first_snapshot_preview.png")
            sc.plot_and_save_stereo(x, y, U, V, W, valid, preview_path, ctrl,
                                     title=f"First snapshot preview (CPU) -- {pair_id}")
            pc.preview_first_snapshot(preview_path)

    return summary_rows


def write_summary(summary_rows, output_dir, ctrl):
    if not summary_rows:
        return
    if ctrl.save_summary_csv:
        csv_path = os.path.join(output_dir, "stereo_processing_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total"])
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    total_time = sum(row[1] for row in summary_rows)
    print(f"Done: {len(summary_rows)} pair(s) in {total_time:.3f} s "
          f"({total_time / len(summary_rows):.3f} s/pair average)")


# ======================================================================
# Main
# ======================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    ctrl = load_controls(config_path)
    os.makedirs(ctrl.output_dir, exist_ok=True)

    if ctrl.input_mode == "set":
        set_paths, is_batch = pc.resolve_set_paths(ctrl.input_path)
    elif ctrl.input_mode == "loose":
        set_paths, is_batch = [ctrl.input_path], False
    else:
        sys.exit(f"Unknown input_mode: {ctrl.input_mode!r} (use 'set' or 'loose')")

    if is_batch:
        print(f"[info] '{ctrl.input_path}' contains {len(set_paths)} set(s) -- "
              "batch-processing each (no first-snapshot preview in this mode)")

    angles = (np.deg2rad(ctrl.alpha1_deg), np.deg2rad(ctrl.alpha2_deg),
              np.deg2rad(ctrl.beta1_deg), np.deg2rad(ctrl.beta2_deg))

    grand_summary = []
    for set_path in set_paths:
        output_dir = (os.path.join(ctrl.output_dir, pc.set_label(set_path))
                       if is_batch else ctrl.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if ctrl.input_mode == "set":
            print(f"[info] processing set '{set_path}'")
            pair_source = sc.iter_stereo_from_set(ctrl, set_path)
        else:
            pair_source = sc.iter_stereo_from_loose_files(ctrl)

        summary_rows = process_pairs(pair_source, ctrl, angles, output_dir,
                                      interactive_preview=not is_batch)
        if not summary_rows:
            print(f"[warn] no stereo pairs were processed for '{set_path}'")
            continue

        write_summary(summary_rows, output_dir, ctrl)
        grand_summary.extend(summary_rows)

    if not grand_summary:
        sys.exit("No stereo pairs were processed -- check input_mode/input_path")


if __name__ == "__main__":
    main()
