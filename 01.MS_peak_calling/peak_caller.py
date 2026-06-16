import sys
import argparse
import numpy as np
import pandas as pd
from pybaselines import Baseline

def track_to_int(s: str) -> int:
    mapping = {"G": 0, "A": 1, "M": 2, "K": 3, "L": 4}
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown isotype string: {s}")


def get_mz_position(x_track: np.array, mz: float) -> int:
    return int(np.argmin(np.abs(x_track[0, :] - mz)))


def _ria_baseline(x_track: np.array, half_window: int = 40, tol: float = .5) -> np.array:
    new_x_track = x_track.copy()
    baseline_fitter = Baseline(x_track[0, :], check_finite=False)
    synthetic_distrib = baseline_fitter.ria(x_track[1, :], half_window=half_window, tol=tol)[0]
    new_x_track[1, :] = synthetic_distrib
    return new_x_track


def correct_cpu_conc(cpu_conc_pct_raw: float, slope: float = 0.11, intercept: float = 0.007) -> float:
    cpu_conc_pct_raw = np.round(cpu_conc_pct_raw, 5)
    cpu_conc_pct_corrected = ((100 * cpu_conc_pct_raw - intercept) / slope) / 100
    return max(0., min(1., cpu_conc_pct_corrected))


def quantify_peak(x: np.array, peak_track: int, peak_position: int, params: dict) -> tuple:
    intermediate_steps = {}

    if "baseline_method" in params and params['baseline_method'] != 'zero':
        if params['baseline_method'] == "ria":
            intermediate_steps["baseline"] = _ria_baseline(x[:, peak_track, :]) * params['baseline_mult_fac']
        else:
            raise ValueError(f"Unsupported baseline method: {params['baseline_method']}")

    if params['numerator'] == 'trapz_over_baseline':
        intermediate_steps["baseline_used_for_loq"] = x[1, peak_track, :] - intermediate_steps["baseline"][1, :]
        intermediate_steps["absolute_loq_threshold"] = np.max(intermediate_steps["baseline_used_for_loq"]) * params['loq_threshold']
        intermediate_steps["peak_center_position"] = peak_position
        
        possible_starts = np.where(intermediate_steps["baseline_used_for_loq"][:peak_position] < intermediate_steps["absolute_loq_threshold"])[0]
        intermediate_steps["peak_start_position"] = possible_starts[-1] if len(possible_starts) > 0 else 0
        
        possible_ends = np.where(intermediate_steps["baseline_used_for_loq"][peak_position:] < intermediate_steps["absolute_loq_threshold"])[0]
        intermediate_steps["peak_end_position"] = (possible_ends[0] + peak_position) if len(possible_ends) > 0 else len(intermediate_steps["baseline_used_for_loq"])
        
        if peak_position - intermediate_steps["peak_start_position"] > params['max_halfwidth']:
            intermediate_steps["peak_start_position"] = peak_position - params['max_halfwidth']
        if intermediate_steps["peak_end_position"] - peak_position > params['max_halfwidth']:
            intermediate_steps["peak_end_position"] = peak_position + params['max_halfwidth']

        intermediate_steps["trace_for_computing_peak_trapz"] = x[1, peak_track, :] - intermediate_steps["baseline"][1, :]
        intermediate_steps["numerator"] = np.trapz(
            intermediate_steps["trace_for_computing_peak_trapz"][intermediate_steps["peak_start_position"]:intermediate_steps["peak_end_position"]],
            x[0, peak_track, intermediate_steps["peak_start_position"]:intermediate_steps["peak_end_position"]]
        )
    else:
        raise ValueError(f"Unsupported numerator strategy")

    intermediate_steps["denominator"] = np.trapz(x[1, peak_track, :], x[0, peak_track, :])

    cpu_conc = (intermediate_steps["numerator"] / intermediate_steps["denominator"]) * params['multiplier']
    if params.get("cap_between_01"):
        cpu_conc = max(min(cpu_conc, 1.0), 0.0)

    return cpu_conc, intermediate_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Peak Caller with Custom MZ Range and Sensitivity Parameters")
    parser.add_argument('--input_npy', type=str, required=True)
    parser.add_argument('--target_mz', type=float, required=True)
    parser.add_argument('--isotype', type=str, required=True)
    parser.add_argument('--total_ig', type=float, required=True)
    
    parser.add_argument('--loq_threshold', type=float, default=0.05)
    parser.add_argument('--max_halfwidth', type=int, default=100)
    
    # Custom m/z window boundaries parameters (from original configurations)
    parser.add_argument('--min_mz', type=float, default=10900.0, help="Minimum m/z boundary for array filter")
    parser.add_argument('--max_mz', type=float, default=12500.0, help="Maximum m/z boundary for array filter")
    
    args = parser.parse_args()

    try:
        x_raw = np.load(args.input_npy)
        # Apply structured boundaries dynamically via input arguments
        mz_filter = (x_raw[0, 0, ...] >= args.min_mz) & (x_raw[0, 0, ...] <= args.max_mz)
        x_raw = x_raw[..., mz_filter]
    except Exception as error:
        print(f"Error reading file or filtering m/z matrix: {error}", file=sys.stderr)
        sys.exit(1)

    track_name = args.isotype.replace("Ig", "")
    track = track_to_int(track_name)
    position = get_mz_position(x_raw[:, track, :], args.target_mz)

    EXEC_PARAMS = {
        'name': 'standalone_call', 
        "mz_range": [args.min_mz, args.max_mz],
        'numerator': 'trapz_over_baseline', 
        'peak_width_method': 'loq', 
        'loq_threshold': args.loq_threshold, 
        'max_halfwidth': args.max_halfwidth,
        'baseline_method': "ria", 
        'baseline_mult_fac': 1.0, 
        'denominator': 'trapz', 
        'multiplier': 1.0, 
        'cap_between_01': True
    }
    PERCENT_LLMI = {"G": 0.0010, "A": 0.00145, "M": 0.0023}

    raw_pct, _ = quantify_peak(x=x_raw, peak_track=track, peak_position=position, params=EXEC_PARAMS)

    if track_name in "GAM":
        if raw_pct < PERCENT_LLMI[track_name]:
            abs_gL = -1.0
            output_status = "Rejected (Below LLMI)"
        else:
            corrected_pct = correct_cpu_conc(raw_pct)
            abs_gL = corrected_pct * args.total_ig
            output_status = "Success"
    else:
        corrected_pct = correct_cpu_conc(raw_pct)
        abs_gL = corrected_pct * args.total_ig
        output_status = "Unable to filter from noise (light chain channel)"

    print("\n" + "="*40 + "\nPeak Call Metrics\n" + "="*40)
    print(f"Process Status: {output_status}")
    print(f"Raw Intensity Ratio: {raw_pct:.5f}")
    print(f"Calculated Concentration: {abs_gL:.4f} g/L (Target m/z: {args.target_mz})")
