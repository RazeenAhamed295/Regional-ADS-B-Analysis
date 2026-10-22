"""
Bootstrap resampling to determine minimum sample size for stable turn statistics.

Input: detected_turns.csv with columns icao24, turn_id, heading_change, severity, is_tight

Steps:
1. Aggregate by airframe (icao24) - total_turns and tight_turns per aircraft
2. Bootstrap resample AIRFRAMES (not individual turns) for N = [50, 100, 200, 400]
3. Calculate 90% CI (5th/95th percentiles) for mean total_turns and mean tight_turns
4. Plot CI width vs sample size N
5. Find minimum N where CI width < 10% of mean
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Load detected_turns.csv and aggregate by icao24."""
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    required_cols = ['icao24', 'turn_id', 'heading_change', 'severity', 'is_tight']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Aggregate by airframe
    agg = df.groupby('icao24').agg(
        total_turns=('turn_id', 'count'),
        tight_turns=('is_tight', 'sum')
    ).reset_index()
    
    return agg


def bootstrap_resample(airframes: pd.DataFrame, sample_sizes: list[int], n_iterations: int = 100, 
                       random_seed: int = 42) -> dict:
    """
    Bootstrap resample AIRFRAMES (with replacement) for each sample size.
    Returns dict with sample_size -> dict of percentiles for total_turns and tight_turns.
    """
    np.random.seed(random_seed)
    n_airframes = len(airframes)
    
    results = {}
    
    for N in sample_sizes:
        total_turns_means = []
        tight_turns_means = []
        
        for _ in range(n_iterations):
            # Sample N airframes WITH REPLACEMENT
            sampled_indices = np.random.choice(n_airframes, size=N, replace=True)
            sample = airframes.iloc[sampled_indices]
            
            total_turns_means.append(sample['total_turns'].mean())
            tight_turns_means.append(sample['tight_turns'].mean())
        
        total_turns_means = np.array(total_turns_means)
        tight_turns_means = np.array(tight_turns_means)
        
        # 90% CI: 5th and 95th percentiles
        results[N] = {
            'total_turns': {
                'mean': total_turns_means.mean(),
                'ci_low': np.percentile(total_turns_means, 5),
                'ci_high': np.percentile(total_turns_means, 95),
                'ci_width': np.percentile(total_turns_means, 95) - np.percentile(total_turns_means, 5)
            },
            'tight_turns': {
                'mean': tight_turns_means.mean(),
                'ci_low': np.percentile(tight_turns_means, 5),
                'ci_high': np.percentile(tight_turns_means, 95),
                'ci_width': np.percentile(tight_turns_means, 95) - np.percentile(tight_turns_means, 5)
            }
        }
    
    return results


def find_minimum_n(results: dict, threshold_pct: float = 0.10) -> dict:
    """Find minimum N where CI width < threshold_pct of mean for both metrics."""
    min_n = {}
    
    for metric in ['total_turns', 'tight_turns']:
        for N in sorted(results.keys()):
            ci_width = results[N][metric]['ci_width']
            mean_val = results[N][metric]['mean']
            
            if mean_val > 0 and (ci_width / mean_val) < threshold_pct:
                min_n[metric] = N
                break
        else:
            min_n[metric] = None  # Not found
    
    return min_n


def create_results_dataframe(results: dict) -> pd.DataFrame:
    """Create a clean DataFrame summarizing CI widths for each N."""
    rows = []
    for N in sorted(results.keys()):
        for metric in ['total_turns', 'tight_turns']:
            r = results[N][metric]
            rows.append({
                'sample_size_N': N,
                'metric': metric,
                'mean': r['mean'],
                'ci_5th': r['ci_low'],
                'ci_95th': r['ci_high'],
                'ci_width': r['ci_width'],
                'ci_width_pct_of_mean': (r['ci_width'] / r['mean'] * 100) if r['mean'] > 0 else np.nan
            })
    
    return pd.DataFrame(rows)


def plot_ci_width(results: dict, output_path: str = 'ci_plot.png'):
    """Plot CI Width vs Sample Size N for both metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sample_sizes = sorted(results.keys())
    
    # Plot for total_turns
    ci_widths_total = [results[N]['total_turns']['ci_width'] for N in sample_sizes]
    ax1.plot(sample_sizes, ci_widths_total, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Sample Size N (Number of Airframes)')
    ax1.set_ylabel('90% CI Width (mean total_turns)')
    ax1.set_title('CI Width vs Sample Size: Total Turns per Airframe')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot for tight_turns
    ci_widths_tight = [results[N]['tight_turns']['ci_width'] for N in sample_sizes]
    ax2.plot(sample_sizes, ci_widths_tight, 'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Sample Size N (Number of Airframes)')
    ax2.set_ylabel('90% CI Width (mean tight_turns)')
    ax2.set_title('CI Width vs Sample Size: Tight Turns per Airframe')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    csv_path = 'detected_turns.csv'
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found in current directory.")
        print("Please ensure detected_turns.csv exists with columns:")
        print("  icao24, turn_id, heading_change, severity, is_tight")
        return
    
    print("Step 1: Loading and aggregating by airframe...")
    airframes = load_and_aggregate(csv_path)
    print(f"  Loaded {len(airframes)} unique airframes")
    print(f"  Total turns across all airframes: {airframes['total_turns'].sum()}")
    print(f"  Total tight turns across all airframes: {airframes['tight_turns'].sum()}")
    
    sample_sizes = [50, 100, 200, 400]
    n_iterations = 100
    
    print(f"\nStep 2: Bootstrap resampling (N={sample_sizes}, iterations={n_iterations})...")
    results = bootstrap_resample(airframes, sample_sizes, n_iterations)
    
    print("\nStep 3: Creating results DataFrame...")
    results_df = create_results_dataframe(results)
    print(results_df.to_string(index=False))
    
    print("\nStep 4: Finding minimum N where CI width < 10% of mean...")
    min_n = find_minimum_n(results, threshold_pct=0.10)
    for metric, n in min_n.items():
        if n:
            ci_width = results[n][metric]['ci_width']
            mean_val = results[n][metric]['mean']
            pct = (ci_width / mean_val * 100)
            print(f"  {metric}: Minimum N = {n} (CI width = {ci_width:.4f}, {pct:.1f}% of mean)")
        else:
            print(f"  {metric}: No N found where CI width < 10% of mean (try larger sample sizes)")
    
    print("\nStep 5: Generating plot...")
    plot_ci_width(results, 'ci_plot.png')
    
    print("\nDone!")


if __name__ == '__main__':
    main()