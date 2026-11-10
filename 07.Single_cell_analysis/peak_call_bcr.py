import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

heavy_chain_dict = {"IGHM": "IgM", "IGHA": "IgA", "IGHG": "IgG"}
light_chain_dict = {"IGLC": "Lambda", "IGKC": "Kappa"}

N_GRID   = 8000
HIST_BIN = 2


def parse_ms_results(ms_results):
    """'IGHA_IGLC:11471.47:0.586:...' -> [{'mz': ..., 'label': 'IgA Lambda (0.59 g/L)'}]"""
    out = []
    for item in ms_results:
        parts = item.split(":")
        isotype, mz, conc = parts[0], float(parts[1]), float(parts[2])
        if "_" in isotype:
            hc, lc = isotype.split("_")
            label = f"{heavy_chain_dict[hc]} {light_chain_dict[lc]}"
        else:
            label = heavy_chain_dict[isotype]
        out.append({'mz': mz, 'label': f"{label} ({conc:.2f} g/L)"})
    return out


def plot_spectra_hist_kde(values, ms_list, title_tag, save=None, clip=None,
                          panel_label=None, x_offset=3, figsize=(12, 7),
                          kde_bw=0.02,       # gaussian_kde bandwidth factor
                          prom_frac=0.002,   # prominence as fraction of max density
                          peak_dist=30,      # min peak separation (grid points)
                          min_cells=10,      # peak must be supported by >= this many cells (+-2 m/z)
                          logy=True,
                          base_font=14, label_font=13, annot_font=11, legend_font=12):
    """
    Top: raw histogram (calibration/ground truth). Bottom: KDE with detected peaks.
    Detection and display use the SAME KDE curve.
    Returns detected peak positions (np.array).
    """
    vals_all = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(float)

    if clip is None:   # auto window: trim rare outlier estimates
        lo, hi = np.percentile(vals_all, [0.5, 99.5])
        clip = (lo - 50, hi + 50)
    vals = vals_all[(vals_all >= clip[0]) & (vals_all <= clip[1])]
    n_out = len(vals_all) - len(vals)
    if n_out:
        print(f"[{title_tag}] {n_out} cells outside {clip[0]:.0f}-{clip[1]:.0f} excluded")

    # ---- KDE + peak detection (one curve for both) ----
    kde = gaussian_kde(vals, bw_method=kde_bw)
    x_range = np.linspace(clip[0], clip[1], N_GRID)
    y_kde = kde(x_range)
    pidx, _ = find_peaks(y_kde, prominence=y_kde.max() * prom_frac, distance=peak_dist)
    pidx = np.array([i for i in pidx
                     if np.sum(np.abs(vals - x_range[i]) <= 2) >= min_cells], int)
    peaks = x_range[pidx]
    print(f"[{title_tag}] Gaussian peaks:", np.round(peaks, 2))

    # ---- plot ----
    plt.rcParams.update({'font.size': base_font})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.4]})

    ax0.hist(vals, bins=np.arange(clip[0], clip[1] + HIST_BIN, HIST_BIN), color='grey')
    ax0.set_ylabel("Cells", fontsize=label_font)

    ax1.plot(x_range, y_kde, color='black', lw=1.5)
    ax1.set_ylabel("Density", fontsize=label_font)

    if logy:
        ax0.set_yscale('log'); ax0.set_ylabel("Cells (log)", fontsize=label_font)
        ax1.set_yscale('log'); ax1.set_ylabel("Density (log)", fontsize=label_font)
        ax1.set_ylim(max(y_kde[y_kde > 0].min(), y_kde.max() * 1e-5), y_kde.max() * 2)

    ax0.tick_params(labelsize=label_font)
    ax1.tick_params(labelsize=label_font)

    # total cells: top-right of the histogram panel
    total_label = f"Total cells = {len(vals):,}"
    if panel_label:
        total_label = f"{panel_label}, {total_label}"
    ax0.text(0.985, 0.92, total_label, transform=ax0.transAxes,
             ha='right', va='top', fontsize=label_font,
             bbox=dict(facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.8))

    peak_colors = sns.color_palette("deep", 10)
    handles = []
    for k, px in enumerate(peaks):
        c = peak_colors[k % len(peak_colors)]
        for ax in (ax0, ax1):
            ax.axvline(px, color=c, linestyle='-', alpha=0.85, linewidth=2.5)
        handles.append(Line2D([0], [0], color=c, lw=3,
                              label=f"scBCR peak #{k+1}: est m/z = {px:.1f}"))

    for ms in ms_list:
        for ax in (ax0, ax1):
            ax.axvline(ms['mz'], color='red', linestyle='--', alpha=0.9)
        y_text = ax1.get_ylim()[0] * 1.5 if logy else ax1.get_ylim()[0]
        ax1.text(ms['mz'] - x_offset, y_text, ms['label'],
                 rotation=90, fontsize=annot_font, ha='left', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
    handles.append(Line2D([0], [0], color='red', ls='--', label='MS peaks'))

    ax1.set_xlabel("Estimated m/z ratio", fontsize=label_font + 1)
    ax1.set_xlim(clip)

    fig.align_ylabels([ax0, ax1])          # <- y-axis labels at the same x position
    fig.legend(handles=handles, title="Peaks",
               bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=legend_font)
    plt.tight_layout()
    if save:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    plt.show()
    return peaks
