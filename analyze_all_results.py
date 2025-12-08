"""
Comprehensive analysis script for comparing Direct Eval, Prolog, and ProbLog methods.

This script performs:
1. Overall metrics comparison across all methods
2. Protected variable analysis (defendant_state, applicant_gender, applicant_age)
3. Metric variation analysis across protected attributes
4. Generates all visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from collections import defaultdict

# Article indices in the ECtHR dataset labels array
# labels array: [2, 3, 5, 6, 8, 9, 11, 14, P1-1]
ARTICLE_INDEX = {
    "article2": 0,  # Article 2 is at index 0
    "article3": 1,  # Article 3 is at index 1
    "article6": 3,  # Article 6 is at index 3
}

# ProbLog threshold for violation detection
PROBLOG_THRESHOLD = 0.75


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_direct_eval_results(jsonl_path: str) -> Dict[str, Dict[str, int]]:
    """Load direct evaluation results from JSONL file."""
    results = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            case_id = str(entry["id"])
            labels = entry.get("labels", {})
            if isinstance(labels, dict):
                results[case_id] = {
                    "article2": labels.get("article2", 0),
                    "article3": labels.get("article3", 0),
                    "article6": labels.get("article6", 0),
                }
            else:
                # If it's a list, assume it's the 9-dim vector
                results[case_id] = {
                    "article2": labels[0] if len(labels) > 0 else 0,
                    "article3": labels[1] if len(labels) > 1 else 0,
                    "article6": labels[3] if len(labels) > 3 else 0,
                }
    return results


def load_prolog_problog_results(json_path: str, direct_eval_ids: set = None) -> Tuple[Dict, Dict]:
    """Load Prolog and ProbLog results from comparison_results.json.
    
    Args:
        json_path: Path to comparison_results.json
        direct_eval_ids: Optional set of case IDs to filter by (only include these cases)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prolog_results = {}
    problog_results = {}
    
    for entry in data.get("prolog_results", []):
        case_id = str(entry["case_id"])
        if direct_eval_ids is not None and case_id not in direct_eval_ids:
            continue
        prolog_results[case_id] = {
            "article2": entry.get("article2", False),
            "article3": entry.get("article3", False),
            "article6": entry.get("article6", False),
        }
    
    for entry in data.get("problog_results", []):
        case_id = str(entry["case_id"])
        if direct_eval_ids is not None and case_id not in direct_eval_ids:
            continue
        problog_results[case_id] = {
            "article2": entry.get("article2", 0.0) >= PROBLOG_THRESHOLD,
            "article3": entry.get("article3", 0.0) >= PROBLOG_THRESHOLD,
            "article6": entry.get("article6", 0.0) >= PROBLOG_THRESHOLD,
        }
    
    return prolog_results, problog_results


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict:
    """Compute precision, recall, F1, accuracy."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


# ============================================================================
# Overall Comparison Analysis
# ============================================================================

def align_results_by_position(
    direct_eval: Dict[str, Dict],
    prolog: Dict[str, Dict],
    problog: Dict[str, Dict],
    ground_truth: List[Dict[str, bool]]
) -> Tuple[Dict[str, List[bool]], Dict[str, List[bool]]]:
    """Align all results by position/order."""
    def try_int(s):
        try:
            return int(s)
        except:
            return float('inf')
    
    direct_ids = sorted(direct_eval.keys(), key=try_int)
    prolog_ids = sorted(prolog.keys(), key=try_int)
    problog_ids = sorted(problog.keys(), key=try_int)
    
    n_cases = min(len(direct_ids), len(prolog_ids), len(problog_ids), len(ground_truth))
    
    aligned_gt = {"article2": [], "article3": [], "article6": []}
    aligned_direct = {"article2": [], "article3": [], "article6": []}
    aligned_prolog = {"article2": [], "article3": [], "article6": []}
    aligned_problog = {"article2": [], "article3": [], "article6": []}
    
    for i in range(n_cases):
        for article in ["article2", "article3", "article6"]:
            aligned_gt[article].append(ground_truth[i][article])
            aligned_direct[article].append(bool(direct_eval[direct_ids[i]][article]))
            aligned_prolog[article].append(bool(prolog[prolog_ids[i]][article]))
            aligned_problog[article].append(bool(problog[problog_ids[i]][article]))
    
    return aligned_gt, {
        "direct_eval": aligned_direct,
        "prolog": aligned_prolog,
        "problog": aligned_problog
    }


def create_overall_visualizations(metrics_data: Dict, output_dir: Path):
    """Create overall comparison visualizations."""
    output_dir.mkdir(exist_ok=True)
    articles = ["article2", "article3", "article6"]
    methods = ["direct_eval", "prolog", "problog"]
    metric_names = ["precision", "recall", "f1", "accuracy"]
    
    # Plot 1: Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Comparison Across Methods', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for metric_idx, metric in enumerate(metric_names):
        ax = axes[metric_idx]
        x = np.arange(len(articles))
        width = 0.25
        
        for method_idx, method in enumerate(methods):
            values = [metrics_data[article][method][metric] for article in articles]
            offset = (method_idx - 1) * width
            ax.bar(x + offset, values, width, label=method.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Article')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} by Article')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('article', 'Article ').upper() for a in articles])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Confusion matrices
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Confusion Matrices by Method and Article', fontsize=16, fontweight='bold')
    
    for article_idx, article in enumerate(articles):
        for method_idx, method in enumerate(methods):
            ax = axes[method_idx, article_idx]
            m = metrics_data[article][method]
            
            cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['No Violation', 'Violation'])
            ax.set_yticklabels(['No Violation', 'Violation'])
            
            if article_idx == 0:
                ax.set_ylabel(method.replace('_', ' ').title(), fontweight='bold')
            if method_idx == 0:
                ax.set_title(article.replace('article', 'Article ').upper(), fontweight='bold')
            
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Violation counts
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(articles))
    width = 0.25
    
    for method_idx, method in enumerate(methods):
        counts = [sum(metrics_data[article][method]['tp'] + metrics_data[article][method]['fp'] 
                     for article in [a]) for a in articles]
        offset = (method_idx - 1) * width
        ax.bar(x + offset, counts, width, label=method.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Article')
    ax.set_ylabel('Predicted Violations')
    ax.set_title('Predicted Violation Counts by Method and Article')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('article', 'Article ').upper() for a in articles])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "violation_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created overall comparison visualizations")


# ============================================================================
# Protected Variable Analysis
# ============================================================================

def analyze_by_protected_variable(
    direct_eval: Dict[str, Dict],
    prolog: Dict[str, Dict],
    problog: Dict[str, Dict],
    test_data,
    protected_var: str
) -> Dict:
    """Analyze metrics broken down by a protected variable."""
    groups = defaultdict(lambda: {
        "case_ids": [],
        "ground_truth": {"article2": [], "article3": [], "article6": []},
        "direct_eval": {"article2": [], "article3": [], "article6": []},
        "prolog": {"article2": [], "article3": [], "article6": []},
        "problog": {"article2": [], "article3": [], "article6": []},
    })
    
    def try_int(s):
        try:
            return int(s)
        except:
            return float('inf')
    
    direct_ids = sorted(direct_eval.keys(), key=try_int)
    prolog_ids = sorted(prolog.keys(), key=try_int)
    problog_ids = sorted(problog.keys(), key=try_int)
    
    n_cases = min(len(direct_ids), len(prolog_ids), len(problog_ids), len(test_data))
    
    for i in range(n_cases):
        if i >= len(test_data):
            break
            
        entry = test_data[i]
        case_id = direct_ids[i]
        
        pv_value = entry.get(protected_var, None)
        if pv_value is None:
            continue
            
        labels = entry.get("labels", [])
        gt = {
            "article2": ARTICLE_INDEX["article2"] in labels,
            "article3": ARTICLE_INDEX["article3"] in labels,
            "article6": ARTICLE_INDEX["article6"] in labels,
        }
        
        de_pred = direct_eval.get(case_id, {"article2": 0, "article3": 0, "article6": 0})
        pl_pred = prolog.get(case_id, {"article2": False, "article3": False, "article6": False})
        pb_pred = problog.get(case_id, {"article2": False, "article3": False, "article6": False})
        
        de_pred_bool = {k: bool(v) for k, v in de_pred.items()}
        pl_pred_bool = {k: bool(v) for k, v in pl_pred.items()}
        pb_pred_bool = {k: bool(v) for k, v in pb_pred.items()}
        
        groups[pv_value]["case_ids"].append(case_id)
        for article in ["article2", "article3", "article6"]:
            groups[pv_value]["ground_truth"][article].append(gt[article])
            groups[pv_value]["direct_eval"][article].append(de_pred_bool[article])
            groups[pv_value]["prolog"][article].append(pl_pred_bool[article])
            groups[pv_value]["problog"][article].append(pb_pred_bool[article])
    
    results = {}
    for pv_value, group_data in groups.items():
        if len(group_data["case_ids"]) == 0:
            continue
            
        results[pv_value] = {}
        for article in ["article2", "article3", "article6"]:
            gt = group_data["ground_truth"][article]
            results[pv_value][article] = {
                "direct_eval": compute_metrics(group_data["direct_eval"][article], gt),
                "prolog": compute_metrics(group_data["prolog"][article], gt),
                "problog": compute_metrics(group_data["problog"][article], gt),
                "n": len(gt),
                "gt_violations": sum(gt),
            }
    
    return results


def create_protected_variable_plots(results: Dict, protected_var: str, output_dir: Path):
    """Create plots showing metrics by protected variable."""
    output_dir.mkdir(exist_ok=True)
    
    articles = ["article2", "article3", "article6"]
    methods = ["direct_eval", "prolog", "problog"]
    metrics = ["precision", "recall", "f1", "accuracy"]
    
    pv_values = sorted([k for k in results.keys() if results[k]])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Metrics by {protected_var.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        x = np.arange(len(pv_values))
        width = 0.25
        
        for method_idx, method in enumerate(methods):
            values = []
            for pv_val in pv_values:
                method_vals = []
                for article in articles:
                    if pv_val in results and article in results[pv_val]:
                        val = results[pv_val][article][method].get(metric, 0.0)
                        method_vals.append(val)
                avg_val = np.mean(method_vals) if method_vals else 0.0
                values.append(avg_val)
            
            offset = (method_idx - 1) * width
            ax.bar(x + offset, values, width, label=method.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel(protected_var.replace("_", " ").title())
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} by {protected_var.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pv_values], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = output_dir / f"metrics_by_{protected_var}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created plot: {output_path}")


# ============================================================================
# Variation Analysis
# ============================================================================

def compute_variation_metrics(results: Dict, protected_var: str) -> Dict:
    """Compute variation metrics (range, std dev) for each metric across groups."""
    articles = ["article2", "article3", "article6"]
    methods = ["direct_eval", "prolog", "problog"]
    metrics = ["precision", "recall", "f1", "accuracy"]
    
    variation = {}
    
    for article in articles:
        variation[article] = {}
        for method in methods:
            variation[article][method] = {}
            
            metric_values = {m: [] for m in metrics}
            
            for pv_value, group_data in results.items():
                if pv_value not in results or article not in results[pv_value]:
                    continue
                if method not in results[pv_value][article]:
                    continue
                
                method_data = results[pv_value][article][method]
                for metric in metrics:
                    if metric in method_data:
                        metric_values[metric].append(method_data[metric])
            
            for metric in metrics:
                if len(metric_values[metric]) == 0:
                    continue
                
                values = np.array(metric_values[metric])
                variation[article][method][metric] = {
                    "range": float(np.max(values) - np.min(values)),
                    "std_dev": float(np.std(values)),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "mean": float(np.mean(values)),
                }
    
    return variation


def create_variation_plots(all_variations: Dict, method_aggregates: Dict, output_dir: Path):
    """Create plots showing metric variation across protected attributes."""
    output_dir.mkdir(exist_ok=True)
    
    methods = ["direct_eval", "prolog", "problog"]
    metrics = ["precision", "recall", "f1", "accuracy"]
    protected_vars = ["defendant_state", "applicant_gender", "applicant_age"]
    
    # Plot 1: Average variation by method and metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Average Metric Variation Across Protected Attributes', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    method_colors = {
        "direct_eval": "#1f77b4",
        "prolog": "#ff7f0e", 
        "problog": "#2ca02c"
    }
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        method_names = []
        avg_ranges = []
        colors = []
        
        for method in methods:
            if len(method_aggregates[method][metric]) > 0:
                avg_range = np.mean(method_aggregates[method][metric])
                method_names.append(method.replace('_', ' ').title())
                avg_ranges.append(avg_range)
                colors.append(method_colors[method])
        
        bars = ax.bar(method_names, avg_ranges, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Average Range (Max - Min)', fontsize=11)
        ax.set_title(f'{metric.title()} Variation', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(avg_ranges) * 1.2 if avg_ranges else 1])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metric_variation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created variation summary plot")


# ============================================================================
# Main Analysis Function
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS: Direct Eval vs Prolog vs ProbLog")
    print("=" * 80)
    print()
    
    # Load all results
    print("Loading results...")
    direct_eval = load_direct_eval_results("results_strict_direct_eval.jsonl")
    direct_eval_ids = set(direct_eval.keys())
    prolog, problog = load_prolog_problog_results("comparison_results.json", direct_eval_ids=direct_eval_ids)
    
    print("Loading ECtHR dataset from Hugging Face...")
    ecthr = load_dataset("coastalcph/fairlex", "ecthr")
    test_data = ecthr["test"]
    
    print(f"Loaded {len(direct_eval)} direct eval results")
    print(f"Loaded {len(prolog)} Prolog results")
    print(f"Loaded {len(problog)} ProbLog results")
    print(f"Loaded {len(test_data)} test cases")
    print()
    
    # Extract ground truth
    ground_truth = []
    for entry in test_data:
        labels = entry.get("labels", [])
        ground_truth.append({
            "article2": ARTICLE_INDEX["article2"] in labels,
            "article3": ARTICLE_INDEX["article3"] in labels,
            "article6": ARTICLE_INDEX["article6"] in labels,
        })
    
    # ========================================================================
    # Part 1: Overall Comparison
    # ========================================================================
    print("=" * 80)
    print("PART 1: OVERALL METRICS COMPARISON")
    print("=" * 80)
    print()
    
    print("Aligning results by position...")
    aligned_gt, aligned_preds = align_results_by_position(
        direct_eval, prolog, problog, ground_truth
    )
    print(f"Aligned {len(aligned_gt['article2'])} cases")
    print()
    
    print("Computing metrics...")
    articles = ["article2", "article3", "article6"]
    methods = ["direct_eval", "prolog", "problog"]
    metrics_data = {}
    
    for article in articles:
        metrics_data[article] = {}
        gt = aligned_gt[article]
        
        for method in methods:
            pred = aligned_preds[method][article]
            metrics = compute_metrics(pred, gt)
            metrics_data[article][method] = metrics
    
    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print()
    
    for article in articles:
        print(f"{article.upper()}:")
        print(f"  Ground Truth Violations: {sum(aligned_gt[article])}")
        print()
        
        for method in methods:
            m = metrics_data[article][method]
            print(f"  {method.replace('_', ' ').title()}:")
            print(f"    Precision: {m['precision']:.3f}")
            print(f"    Recall:    {m['recall']:.3f}")
            print(f"    F1:        {m['f1']:.3f}")
            print(f"    Accuracy:  {m['accuracy']:.3f}")
            print(f"    TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}")
            print()
        print()
    
    # Create overall visualizations
    print("Creating overall comparison visualizations...")
    create_overall_visualizations(metrics_data, Path("plots"))
    
    # Save overall metrics
    with open("all_methods_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print("Saved overall metrics to: all_methods_comparison.json")
    print()
    
    # ========================================================================
    # Part 2: Protected Variable Analysis
    # ========================================================================
    print("=" * 80)
    print("PART 2: PROTECTED VARIABLE ANALYSIS")
    print("=" * 80)
    print()
    
    protected_vars = ["defendant_state", "applicant_gender", "applicant_age"]
    all_protected_results = {}
    
    for pv in protected_vars:
        print(f"Analyzing by {pv}...")
        results = analyze_by_protected_variable(
            direct_eval, prolog, problog, test_data, pv
        )
        all_protected_results[pv] = results
        
        # Create plots
        create_protected_variable_plots(results, pv, Path("plots"))
    
    # Save protected variable results
    with open("protected_variables_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(all_protected_results, f, indent=2, ensure_ascii=False)
    print("Saved protected variable analysis to: protected_variables_analysis.json")
    print()
    
    # ========================================================================
    # Part 3: Variation Analysis
    # ========================================================================
    print("=" * 80)
    print("PART 3: METRIC VARIATION ANALYSIS")
    print("=" * 80)
    print()
    
    all_variations = {}
    method_aggregates = {method: {metric: [] for metric in ["precision", "recall", "f1", "accuracy"]} 
                        for method in methods}
    
    for pv in protected_vars:
        if pv not in all_protected_results:
            continue
        
        print(f"Computing variation for {pv}...")
        variation = compute_variation_metrics(all_protected_results[pv], pv)
        all_variations[pv] = variation
        
        # Aggregate for overall summary
        for method in methods:
            for metric in ["precision", "recall", "f1", "accuracy"]:
                metric_ranges = []
                for article in articles:
                    if article in variation and method in variation[article]:
                        if metric in variation[article][method]:
                            metric_ranges.append(variation[article][method][metric]['range'])
                if metric_ranges:
                    method_aggregates[method][metric].extend(metric_ranges)
    
    # Create variation plots
    print("Creating variation visualizations...")
    create_variation_plots(all_variations, method_aggregates, Path("plots"))
    
    # Print variation summary
    print("\n" + "=" * 80)
    print("VARIATION SUMMARY (Average Range Across All Protected Variables)")
    print("=" * 80)
    print()
    
    print(f"{'Method':<15} {'Metric':<12} {'Avg Variation':<15}")
    print("-" * 50)
    
    for method in methods:
        for metric in ["precision", "recall", "f1", "accuracy"]:
            if len(method_aggregates[method][metric]) > 0:
                avg_var = np.mean(method_aggregates[method][metric])
                print(f"{method.replace('_', ' ').title():<15} {metric:<12} {avg_var:<15.4f}")
    
    print()
    
    # Save variation results
    with open("metric_variation_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(all_variations, f, indent=2, ensure_ascii=False)
    print("Saved variation analysis to: metric_variation_analysis.json")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll results and visualizations saved to:")
    print("  - all_methods_comparison.json")
    print("  - protected_variables_analysis.json")
    print("  - metric_variation_analysis.json")
    print("  - plots/ directory")


if __name__ == "__main__":
    main()

