"""
Replication of Table 1 and Figure 3 from
"Bond Market Views of the Fed's Monetary Policy Rule"

Key methodological choices (discovered through iteration):
1. Daily changes restricted to ONE-CALENDAR-DAY gaps only (drops weekends/holidays)
2. Winsorize raw yield/IC changes at 1st/99th percentiles BEFORE constructing variables
3. Forward-window lag uses (window_length - 1) / window_length scaling
4. HAC standard errors with maxlags=5 for Figure 3
5. HC1 robust standard errors for Table 1
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os
import io
import requests

OUTPUT_DIR = "/home/pblit/Music/Projects/luigi/claude"

SAMPLE_START = "2000-01-03"
SAMPLE_END = "2022-02-28"
SHORT_SAMPLE_START = "2004-01-02"
POST_BREAK = "2020-08-01"
RHO = 0.8
HAC_MAXLAGS = 5


# ============================================================
# 1. Download and parse data
# ============================================================

def download_data():
    """Download nominal and TIPS yield curve data from the Fed."""

    url_nom = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    print("Downloading nominal yields...")
    resp = requests.get(url_nom)
    lines = resp.text.split('\n')
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df_nom = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), na_values=['NA'])
    df_nom['Date'] = pd.to_datetime(df_nom['Date'], errors='coerce')
    df_nom = df_nom.dropna(subset=['Date'])
    print(f"  Nominal yields: {len(df_nom)} rows, {df_nom['Date'].min()} to {df_nom['Date'].max()}")

    url_tips = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
    print("Downloading TIPS yields...")
    resp = requests.get(url_tips)
    lines = resp.text.split('\n')
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df_tips = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), na_values=['NA'])
    df_tips['Date'] = pd.to_datetime(df_tips['Date'], errors='coerce')
    df_tips = df_tips.dropna(subset=['Date'])
    print(f"  TIPS yields: {len(df_tips)} rows, {df_tips['Date'].min()} to {df_tips['Date'].max()}")

    return df_nom, df_tips


def prepare_data(df_nom, df_tips):
    """Merge datasets, apply sample restrictions, construct changes."""

    needed_nom = ['Date', 'SVENY01', 'SVENY02', 'SVENY04', 'SVENY05', 'SVENY09', 'SVENY10']
    needed_tips = ['Date', 'TIPSY02', 'TIPSY05', 'TIPSY10']

    df = df_nom[needed_nom].merge(df_tips[needed_tips], on='Date', how='inner')
    df = df[(df['Date'] >= SAMPLE_START) & (df['Date'] <= SAMPLE_END)].copy()
    print(f"  Merged sample: {len(df)} rows")

    # Inflation compensation
    df['IC2'] = df['SVENY02'] - df['TIPSY02']
    df['IC5'] = df['SVENY05'] - df['TIPSY05']
    df['IC10'] = df['SVENY10'] - df['TIPSY10']

    # KEY: Only use changes where dates are exactly 1 calendar day apart
    one_day_gap = df['Date'].diff().dt.days.eq(1)

    def winsorize(s, lo=0.01, hi=0.99):
        q = s.quantile([lo, hi])
        return s.clip(q.iloc[0], q.iloc[1])

    # Compute raw daily changes, restrict to 1-day gaps, then winsorize
    raw_cols = ['SVENY01', 'SVENY02', 'SVENY04', 'SVENY05', 'SVENY09', 'SVENY10',
                'IC2', 'IC5', 'IC10']
    for col in raw_cols:
        df[f'd_{col}'] = winsorize(df[col].diff().where(one_day_gap))

    n_one_day = one_day_gap.sum()
    print(f"  One-calendar-day gaps: {n_one_day} (dropped {len(df) - 1 - n_one_day} weekend/holiday gaps)")

    return df


# ============================================================
# 2. Construct regression variables and run regressions
# ============================================================

def build_lhs_rhs(df, spec):
    """Construct LHS and RHS for a given specification."""
    lag_factor = (spec['window_length'] - 1) / spec['window_length']
    lhs = df.eval(spec['left_num']) - RHO * lag_factor * df.eval(spec['lag_num'])
    rhs = (1 - RHO) * df.eval(spec['right_num'])
    return lhs, rhs


SPECS = {
    '10y_avg': {
        'label': '10yr avg', 'sample_start': SAMPLE_START,
        'left_num': 'd_SVENY10', 'lag_num': 'd_SVENY09', 'right_num': 'd_IC10',
        'window_length': 10,
    },
    '0_2': {
        'label': '[0-2] yr', 'sample_start': SHORT_SAMPLE_START,
        'left_num': 'd_SVENY02', 'lag_num': 'd_SVENY01', 'right_num': 'd_IC2',
        'window_length': 2,
    },
    '3_5': {
        'label': '[3-5] yr', 'sample_start': SHORT_SAMPLE_START,
        'left_num': '(5 * d_SVENY05 - 2 * d_SVENY02) / 3',
        'lag_num': '(4 * d_SVENY04 - d_SVENY01) / 3',
        'right_num': '(5 * d_IC5 - 2 * d_IC2) / 3',
        'window_length': 3,
    },
    '6_10': {
        'label': '[6-10] yr', 'sample_start': SAMPLE_START,
        'left_num': '(10 * d_SVENY10 - 5 * d_SVENY05) / 5',
        'lag_num': '(9 * d_SVENY09 - 4 * d_SVENY04) / 5',
        'right_num': '(10 * d_IC10 - 5 * d_IC5) / 5',
        'window_length': 5,
    },
    '0_5': {
        'label': '[0-5] yr', 'sample_start': SAMPLE_START,
        'left_num': 'd_SVENY05', 'lag_num': 'd_SVENY04', 'right_num': 'd_IC5',
        'window_length': 5,
    },
}

TABLE_PERIODS = [
    ('2000-01-03', '2003-12-31', '2000-2003', 1),
    ('2004-01-01', '2007-12-31', '2004-2007', 2),
    ('2008-01-01', '2011-12-31', '2008-2011', 3),
    ('2012-01-01', '2015-12-31', '2012-2015', 4),
    ('2016-01-01', '2019-12-31', '2016-2019', 5),
    ('2020-01-01', None,         '2020-2022', 6),
]


def excluded_mask(dates):
    """Return boolean mask for excluded periods (2008, Jan-Jul 2020)."""
    return (dates.dt.year == 2008) | ((dates >= '2020-01-01') & (dates <= '2020-07-31'))


def run_table1_regression(df, spec_key):
    """Run equation (8) regression for a given column."""
    spec = SPECS[spec_key]
    lhs, rhs = build_lhs_rhs(df, spec)

    d = df.copy()
    d['lhs'] = lhs
    d['rhs'] = rhs
    d = d[d['Date'] >= spec['sample_start']].copy()
    d = d[~excluded_mask(d['Date'])].copy()

    # Create period interaction terms
    x_cols = []
    for start, end, label, _ in TABLE_PERIODS:
        final_end = SAMPLE_END if end is None else end
        col = f'x_{label}'
        d[col] = d['rhs'] * ((d['Date'] >= start) & (d['Date'] <= final_end)).astype(float)
        x_cols.append(col)

    d = d.dropna(subset=['lhs', 'rhs'])

    X = sm.add_constant(d[x_cols])
    model = OLS(d['lhs'], X).fit(cov_type='HC1')

    return model, len(d)


def run_figure3_regression(df, spec_key):
    """Run equation (9) regression with HAC standard errors."""
    spec = SPECS[spec_key]
    lhs, rhs = build_lhs_rhs(df, spec)

    d = df.copy()
    d['lhs'] = lhs
    d['rhs'] = rhs
    d = d[d['Date'] >= spec['sample_start']].copy()
    d = d[~excluded_mask(d['Date'])].copy()
    d['post'] = (d['Date'] >= POST_BREAK).astype(float)
    d['post_rhs'] = d['post'] * d['rhs']
    d = d.dropna(subset=['lhs', 'rhs'])

    X = sm.add_constant(d[['rhs', 'post_rhs']])
    model = OLS(d['lhs'], X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS})

    return model, len(d)


# ============================================================
# 3. Output formatting
# ============================================================

def format_table1(results):
    """Format Table 1 output."""
    period_labels = [label for _, _, label, _ in TABLE_PERIODS]
    cw = 16

    output = []
    output.append("\nTABLE 1: Estimates of equation (8)")
    output.append("=" * (22 + 4 * cw))
    output.append(f"{'':22s}{'(1)':>{cw}s}{'(2)':>{cw}s}{'(3)':>{cw}s}{'(4)':>{cw}s}")
    output.append(f"{'':22s}{'10yr avg':>{cw}s}{'[0-2] yr':>{cw}s}{'[3-5] yr':>{cw}s}{'[6-10] yr':>{cw}s}")
    output.append("-" * (22 + 4 * cw))

    col_keys = ['10y_avg', '0_2', '3_5', '6_10']

    for _, _, plabel, pidx in TABLE_PERIODS:
        coef_row = f"psi_{plabel:17s}"
        se_row = f"{'':22s}"

        for ck in col_keys:
            model, n = results[ck]
            x_col = f'x_{plabel}'
            if x_col in model.params.index:
                coef = model.params[x_col]
                pval = model.pvalues[x_col]
                se = model.bse[x_col]
                stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else '   '))
                coef_row += f"{coef:8.2f}{stars:4s}    "
                se_row += f"{'(' + f'{se:.2f}' + ')':>{cw}s}"
            else:
                coef_row += f"{'':>{cw}s}"
                se_row += f"{'':>{cw}s}"

        output.append(coef_row)
        output.append(se_row)
        output.append("")

    output.append("-" * (22 + 4 * cw))
    r2_row = f"{'R-squared':22s}"
    n_row = f"{'N':22s}"
    for ck in col_keys:
        model, n = results[ck]
        r2_row += f"{model.rsquared:>{cw}.2f}"
        n_row += f"{n:>{cw}d}"
    output.append(r2_row)
    output.append(n_row)
    output.append("=" * (22 + 4 * cw))
    output.append("\nNotes: Robust (HC1) standard errors in parentheses.")
    output.append("*** p<0.01, ** p<0.05, * p<0.10\n")

    return '\n'.join(output)


def create_figure3(fig_results):
    """Create Figure 3."""
    specs = list(fig_results.keys())
    z_99 = norm.ppf(0.995)

    d_vals, ci_lo, ci_hi = [], [], []
    for spec in specs:
        model, _ = fig_results[spec]
        d = model.params['post_rhs']
        se = model.bse['post_rhs']
        d_vals.append(d)
        ci_lo.append(d - z_99 * se)
        ci_hi.append(d + z_99 * se)

    fig, ax = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(len(specs))

    for i in range(len(specs)):
        ax.plot([i, i], [ci_lo[i], ci_hi[i]], color='#4472C4', linewidth=2.5)
        cap_w = 0.12
        ax.plot([i-cap_w, i+cap_w], [ci_lo[i]]*2, color='#C0504D', linewidth=2.5)
        ax.plot([i-cap_w, i+cap_w], [ci_hi[i]]*2, color='#C0504D', linewidth=2.5)
        ax.plot(i, d_vals[i], 'o', color='#4472C4', markersize=7, zorder=5)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(['Baseline\n[0-10]', 'Year 0-5', 'Year 6-10'], fontsize=11)
    ax.set_yticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_ylim(-1.6, 0.5)
    ax.set_xlim(-0.5, len(specs) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure3.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 60)
    print("REPLICATION: Bond Market Views of the Fed")
    print("=" * 60)

    df_nom, df_tips = download_data()
    df = prepare_data(df_nom, df_tips)

    # ---- Table 1 ----
    print("\n" + "=" * 60)
    print("TABLE 1 REGRESSIONS (Equation 8)")
    print("=" * 60)

    table_results = {}
    for key in ['10y_avg', '0_2', '3_5', '6_10']:
        model, n = run_table1_regression(df, key)
        table_results[key] = (model, n)
        print(f"\n{SPECS[key]['label']}: N={n}, R2={model.rsquared:.4f}")

    table_str = format_table1(table_results)
    print(table_str)
    with open(os.path.join(OUTPUT_DIR, 'table1.txt'), 'w') as f:
        f.write(table_str)

    # ---- Figure 3 ----
    print("=" * 60)
    print("FIGURE 3 REGRESSIONS (Equation 9)")
    print("=" * 60)

    fig_results = {}
    for key in ['10y_avg', '0_5', '6_10']:
        model, n = run_figure3_regression(df, key)
        fig_results[key] = (model, n)
        d = model.params['post_rhs']
        se = model.bse['post_rhs']
        print(f"  {SPECS[key]['label']}: d={d:.4f} (SE={se:.4f}), N={n}")

    create_figure3(fig_results)

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("COMPARISON WITH TARGETS")
    print("=" * 60)

    targets = {
        '10y_avg': {1:1.63, 2:1.41, 3:1.55, 4:1.48, 5:1.55, 6:1.02},
        '0_2':     {2:1.21, 3:1.03, 4:-0.06, 5:1.06, 6:0.06},
        '3_5':     {2:2.11, 3:2.41, 4:1.94, 5:1.87, 6:1.07},
        '6_10':    {1:1.61, 2:1.67, 3:2.02, 4:2.11, 5:1.87, 6:1.50},
    }
    target_n = {'10y_avg': 4019, '0_2': 3249, '3_5': 3249, '6_10': 4019}
    target_r2 = {'10y_avg': 0.41, '0_2': 0.07, '3_5': 0.26, '6_10': 0.34}

    for key in ['10y_avg', '0_2', '3_5', '6_10']:
        model, n = table_results[key]
        print(f"\n{SPECS[key]['label']} (N={n} vs {target_n[key]}, R2={model.rsquared:.2f} vs {target_r2[key]}):")
        for _, _, plabel, pidx in TABLE_PERIODS:
            x_col = f'x_{plabel}'
            if x_col in model.params.index and pidx in targets[key]:
                mc = model.params[x_col]
                tc = targets[key][pidx]
                print(f"  {plabel}: {mc:.2f} vs {tc:.2f} (diff={mc-tc:+.2f})")

    print(f"\nFigure 3:")
    print(f"  Baseline: d={fig_results['10y_avg'][0].params['post_rhs']:.3f} (target: -0.54)")
    print(f"  [0-5]:    d={fig_results['0_5'][0].params['post_rhs']:.3f} (target: ~-1.0)")
    print(f"  [6-10]:   d={fig_results['6_10'][0].params['post_rhs']:.3f} (target: -0.30)")


if __name__ == '__main__':
    main()
