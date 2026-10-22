# =====================================================================
# Does BCR-predicted m/z matching the patient's own M-protein exceed
# random chance?
#
# Design:
#   - True pair      : (case's expanded-clone est_mz, case's OWN MS m/z)
#   - Negative control: inter-patient decoy pairs = same est_mz vs
#                       M-protein m/z of OTHER individuals, restricted to
#                       the SAME heavy-chain isotype (conservative: m/z is
#                       isotype-dependent, so cross-isotype decoys would
#                       inflate apparent specificity)
#   - Match criterion: |est_mz - MS m/z| <= delta
#   - Tests:
#       (1) Monte-Carlo permutation: each iteration pairs every case with
#           one random same-isotype decoy M-protein -> null distribution
#           of the number of matches -> empirical P(null >= observed)
#       (2) Per-case percentile: rank of the true distance among that
#           case's decoy distances (0% = closer than all decoys)
#       (3) Matched vs decoy |dmz| distributions (Mann-Whitney U)
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ISO_PALETTE = {'IgG': '#66c2a5', 'IgA': '#fc8d62', 'IgM': '#8da0cb'}


def match_specificity_analysis(pairs_df, decoy_pool,
                               id_col='case_id', est_col='est_mz_ratio',
                               ms_col='MS_mz_ratio', iso_col='isotype',
                               pool_id_col='pid', pool_mz_col='mz',
                               pool_iso_col='isotype',
                               delta=10, n_perm=10000, seed=1):
    """
    pairs_df  : one row per case (true pair).
    decoy_pool: one row per candidate decoy M-protein from the cohort
                (other individuals' MS peaks). Deduplicated beforehand.
    Returns (case_table, summary_dict, null_counts).
    """
    rng = np.random.default_rng(seed)
    pairs = pairs_df.copy().reset_index(drop=True)
    pool = decoy_pool.copy()
    pool[pool_mz_col] = pd.to_numeric(pool[pool_mz_col], errors='coerce')
    pool = pool.dropna(subset=[pool_mz_col, pool_iso_col])

    case_rows, decoy_sets = [], []
    for _, r in pairs.iterrows():
        est_vals = np.atleast_1d(np.asarray(r[est_col], dtype=float))  
        ms, iso, cid = float(r[ms_col]), r[iso_col], r[id_col]
        dp = pool[(pool[pool_iso_col] == iso) & (pool[pool_id_col] != cid)]
        dp_mz = dp[pool_mz_col].to_numpy(float)
        d_decoy = np.min(np.abs(dp_mz[:, None] - est_vals[None, :]), axis=1)  
        true_d = float(np.min(np.abs(est_vals - ms)))                          
        decoy_sets.append(d_decoy)
        case_rows.append({
            'case_id': cid, 'isotype': iso,
            'n_clones_tested': len(est_vals),                                  
            'est_mz_ratio': float(est_vals[np.argmin(np.abs(est_vals - ms))]), 
            'MS_mz_ratio': ms,
            'true_abs_diff': true_d,
            'matched': bool(true_d <= delta),
            'n_decoys': len(d_decoy),
            'decoy_median_abs_diff': float(np.median(d_decoy)) if len(d_decoy) else np.nan,
            'decoy_match_prob': float(np.mean(d_decoy <= delta)) if len(d_decoy) else np.nan,
            'percentile_vs_decoys': float(np.mean(d_decoy < true_d) * 100) if len(d_decoy) else np.nan,
        })
    case_table = pd.DataFrame(case_rows)

    observed = int(case_table['matched'].sum())
    n_cases = len(case_table)

    # Monte-Carlo null: one random same-isotype decoy per case per iteration
    null_counts = np.zeros(n_perm, dtype=int)
    for b in range(n_perm):
        cnt = 0
        for d_decoy in decoy_sets:
            if len(d_decoy) == 0:
                continue
            cnt += d_decoy[rng.integers(len(d_decoy))] <= delta
        null_counts[b] = cnt
    p_perm = (1 + np.sum(null_counts >= observed)) / (n_perm + 1)

    # matched vs decoy distance distributions
    all_decoy_d = np.concatenate([d for d in decoy_sets if len(d)]) if decoy_sets else np.array([])
    mw_p = np.nan
    if len(all_decoy_d) and len(case_table):
        mw_p = mannwhitneyu(case_table['true_abs_diff'], all_decoy_d,
                            alternative='less').pvalue

    summary = {
        'n_cases': n_cases, 'observed_matches': observed,
        'observed_match_rate': observed / n_cases if n_cases else np.nan,
        'delta': delta,
        'null_mean_matches': float(null_counts.mean()),
        'null_expected_rate': float(null_counts.mean() / n_cases) if n_cases else np.nan,
        'p_permutation': float(p_perm), 'n_perm': n_perm,
        'median_true_diff': float(case_table['true_abs_diff'].median()),
        'median_decoy_diff': float(np.median(all_decoy_d)) if len(all_decoy_d) else np.nan,
        'p_mannwhitney_true_lt_decoy': float(mw_p),
        'median_percentile_vs_decoys': float(case_table['percentile_vs_decoys'].median()),
    }
    return case_table, summary, null_counts


def plot_match_specificity(case_table, summary, null_counts, cohort_label,
                           save_fig=None, figsize=(11, 4.2)):
    """Supplementary figure: (A) true vs decoy-null match count, (B) per-case
    percentile of true distance among same-isotype decoys."""
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (A) permutation null vs observed
    bins = np.arange(-0.5, summary['n_cases'] + 1.5, 1)
    ax1.hist(null_counts, bins=bins, color='lightgrey', edgecolor='grey', density=True)
    ax1.axvline(summary['observed_matches'], color='#d7301f', lw=2.5)
    ax1.text(summary['observed_matches'], ax1.get_ylim()[1]*0.95,
             f"  observed = {summary['observed_matches']}/{summary['n_cases']}\n"
             f"  P = {summary['p_permutation']:.2g}",
             color='#d7301f', va='top', ha='left', fontsize=11)
    ax1.set_xlabel(f"Matches expected by chance\n(random same-isotype pairing, "
                   f"|Δm/z| ≤ {summary['delta']})")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{cohort_label}: match count vs permutation null", fontsize=12)

    # (B) per-case percentile of true distance among decoys
    ct = case_table.sort_values('percentile_vs_decoys').reset_index(drop=True)
    colors = [ISO_PALETTE.get(i, 'grey') for i in ct['isotype']]
    y = np.arange(len(ct))
    ax2.hlines(y, 0, ct['percentile_vs_decoys'], color=colors, lw=2, alpha=0.8)
    ax2.scatter(ct['percentile_vs_decoys'], y, c=colors, s=45,
                edgecolor='black', linewidth=0.5, zorder=3)
    ax2.axvline(50, ls='--', color='grey', alpha=0.5)
    if len(ct) <= 20:                       # per-case labels only for small n
        ax2.set_yticks(y)
        ax2.set_yticklabels(ct['case_id'], fontsize=9)
    else:
        ax2.set_yticks([])
        ax2.set_ylabel(f"Cases (n={len(ct)}, sorted)")
    ax2.set_xlabel("Percentile of true |Δm/z| among\nsame-isotype inter-patient decoys (%)")
    ax2.set_xlim(-2, 102)
    ax2.set_title("Per-case specificity (0% = closest)", fontsize=12)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markeredgecolor='black', markersize=8, label=i)
               for i, c in ISO_PALETTE.items() if i in set(ct['isotype'])]
    ax2.legend(handles=handles, fontsize=10, loc='lower right')

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=300)
    plt.show()


def sensitivity_over_delta(pairs_df, decoy_pool, deltas=(5, 10, 20),
                           n_perm=10000, **kwargs):
    rows = []
    for d in deltas:
        _, s, _ = match_specificity_analysis(pairs_df, decoy_pool,
                                             delta=d, n_perm=n_perm, **kwargs)
        rows.append({'delta': d,
                     'observed_matches': s['observed_matches'],
                     'n_cases': s['n_cases'],
                     'null_mean_matches': round(s['null_mean_matches'], 2),
                     'p_permutation': s['p_permutation']})
    return pd.DataFrame(rows)

