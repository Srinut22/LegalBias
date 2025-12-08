"""
Compare standard Prolog vs ProbLog evaluation for articles 2, 3, and 6.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try to import ProbLog
try:
    from problog.program import PrologString
    from problog import get_evaluatable
    PROBLOG_AVAILABLE = True
except ImportError:
    PROBLOG_AVAILABLE = False
    print("Warning: ProbLog not available. Install with: pip install problog")
    print("Will only evaluate with standard Prolog.")


# ============================================================
# SWI-Prolog evaluation
# ============================================================

def query_prolog(query: str, prolog_files: List[str], extract_var: Optional[str] = None, debug: bool = False) -> Tuple[bool, List[str]]:
    """Query Prolog using SWI-Prolog."""
    
    # All predicates that might appear in case files (from generateProlog.py)
    # Declare them as dynamic so Prolog doesn't error when they're missing
    DYNAMIC_PREDICATES = [
        # Article 2 predicates
        "state_responsible_for_death/1", "life_taken_intentionally/1", "death_resulted_from_force/1",
        "force_used_against/1", "force_absolutely_necessary/1", "defence_from_unlawful_violence/1",
        "effect_lawful_arrest/1", "prevent_escape_of_lawfully_detained_individual/1", "quell_riot_or_insurrection/1",
        "convicted_of/2", "death_penalty_provided_by_law/1",
        # Article 3 predicates
        "state_responsible_for_mistreatment/1", "torture/1", "inhuman_treatment/1", "degrading_treatment/1",
        # Article 6 predicates
        "trial/1", "independent_tribunal/1", "impartial_tribunal/1", "tribunal_established_by_law/1",
        "hearing_within_reasonable_time/1", "fair_hearing/1", "public_judgment_or_justified_exclusion/1",
        "charged/2", "presumed_innocent/1",
        "informed_promptly_in_language_understood_of_nature_and_cause/1",
        "adequate_time_and_facilities_for_defence/1", "effective_legal_assistance/1",
        "relevant_witness_evidence/1", "opportunity_to_examine_witnesses_on_equal_terms/1",
        "needs_interpreter/1", "interpreter_provided_free_of_charge/1",
        # Helper predicates from article rules (these are defined in the rule files)
        "deprivation_of_life/1", "justified_force/1", "judicial_execution/1",
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8') as f:
        # Declare all predicates as dynamic so missing facts don't cause errors
        for pred in DYNAMIC_PREDICATES:
            f.write(f":- dynamic {pred}.\n")
        
        # Consult all given Prolog files
        for pl_file in prolog_files:
            pl_file_escaped = pl_file.replace('\\', '/')
            f.write(f":- consult('{pl_file_escaped}').\n")

        # Build query code
        if extract_var:
            # print all solutions for extract_var
            prolog_code = (
                f":- catch((findall({extract_var}, ({query}), Solutions), "
                f"forall(member(P, Solutions), writeln(P))), Error, "
                f"(print_message(error, Error), fail)), halt.\n"
            )
        else:
            # just say true/false
            prolog_code = (
                f":- catch((({query}) -> writeln('true') ; writeln('false')), "
                f"Error, (print_message(error, Error), writeln('false'))), halt.\n"
            )

        f.write(prolog_code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ['swipl', '-q', '-t', 'halt', '-s', temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        if debug and result.stderr:
            print(f"  [DEBUG] Prolog stderr: {result.stderr[:500]}")
        if debug:
            print(f"  [DEBUG] Prolog stdout: {result.stdout[:500]}")
            print(f"  [DEBUG] Return code: {result.returncode}")

        if result.returncode == 0:
            output = result.stdout.strip()
            if extract_var:
                solutions = [line.strip() for line in output.split('\n') if line.strip()]
                # Filter out error messages
                solutions = [s for s in solutions if not s.startswith('ERROR:') and not s.startswith('Warning:')]
                return (len(solutions) > 0, solutions)
            else:
                has_solution = output.strip().lower() == "true"
                return (has_solution, [])
        
        # If return code is non-zero, check stderr
        if result.stderr and not debug:
            # Only print first error to avoid spam
            pass
        return (False, [])
    except Exception as e:
        if debug:
            print(f"  [DEBUG] Exception: {e}")
        return (False, [])
    finally:
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


# ============================================================
# ProbLog evaluation
# ============================================================

def query_problog(query: str, prolog_files: List[str]) -> float:
    """
    Query ProbLog and return the probability of a **ground** query.

    Assumes:
      - `query` is ground, e.g. "violation(article6, the_applicant)".
      - `prolog_files` includes:
          * one ProbLog case file (with `p::fact.` facts), and
          * one or more rule files (pure Prolog, e.g. article2.pl).
    """
    if not PROBLOG_AVAILABLE:
        return 0.0

    try:
        # 1) Read all Prolog / ProbLog code as-is
        prolog_code = ""
        for pl_file in prolog_files:
            with open(pl_file, 'r', encoding='utf-8') as f:
                prolog_code += f.read() + "\n"

        # 2) Add a query for the given ground predicate
        prolog_code += f"\nquery({query}).\n"

        # 3) Parse and evaluate
        program = PrologString(prolog_code)
        evaluatable = get_evaluatable().create_from(program)
        result = evaluatable.evaluate()

        # 4) We asked a single ground query, so result should be a single probability
        if isinstance(result, dict) and result:
            return float(next(iter(result.values())))
        if isinstance(result, (int, float)):
            return float(result)
        return 0.0
    except Exception:
        # ProbLog errors are expected for some cases – treat as 0 probability
        return 0.0


# ============================================================
# Person extraction from case files
# ============================================================

def get_all_persons_in_case(case_file: Path) -> List[str]:
    """
    Extract all Person arguments from a case file by parsing facts.
    Simple approach: find '(' then extract until ',' or ')'.
    """
    persons = set()
    try:
        with open(case_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        # Match all facts: pred(arg1, arg2, ...).
        # Handle both regular facts and ProbLog facts: 0.5::pred(arg1, arg2).
        pattern = r'(?:\d+\.?\d*::\s*)?\w+\s*\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        for arglist in matches:
            # Split by comma, but be careful with nested parens
            args = []
            current = ""
            depth = 0
            for char in arglist:
                if char == '(':
                    depth += 1
                    current += char
                elif char == ')':
                    depth -= 1
                    current += char
                elif char == ',' and depth == 0:
                    args.append(current.strip())
                    current = ""
                else:
                    current += char
            if current.strip():
                args.append(current.strip())
            
            # Add all arguments as potential persons
            for arg in args:
                arg = arg.strip()
                if arg and not arg.startswith('_') and not arg.isdigit():
                    arg_clean = arg.strip('"').strip("'")
                    persons.add(arg_clean)
    except Exception:
        pass
    return sorted(list(persons))


# ============================================================
# Per-case evaluation
# ============================================================

def evaluate_case_prolog(case_file: Path, article2_rules: Path, article3_rules: Path, article6_rules: Path) -> Dict:
    """Evaluate a case using standard Prolog (crisp semantics)."""
    case_id = case_file.stem.replace('case_', '')
    case_file_abs = str(case_file.resolve()).replace('\\', '/')

    results = {
        "case_id": case_id,
        "article2": False,
        "article3": False,
        "article6": False,
    }

    # Extract all persons from the case file
    persons = get_all_persons_in_case(case_file)
    
    # For each article, query for violations with each person
    # Article 2
    prolog_files = [str(article2_rules.resolve().as_posix()), case_file_abs]
    for person in persons:
        has_violation, _ = query_prolog(f"violation(article2, {person})", prolog_files)
        if has_violation:
            results["article2"] = True
            break

    # Article 3
    prolog_files = [str(article3_rules.resolve().as_posix()), case_file_abs]
    for person in persons:
        has_violation, _ = query_prolog(f"violation(article3, {person})", prolog_files)
        if has_violation:
            results["article3"] = True
            break

    # Article 6
    prolog_files = [str(article6_rules.resolve().as_posix()), case_file_abs]
    for person in persons:
        has_violation, _ = query_prolog(f"violation(article6, {person})", prolog_files)
        if has_violation:
            results["article6"] = True
            break

    return results


def evaluate_case_problog(case_file: Path, article2_rules: Path, article3_rules: Path, article6_rules: Path) -> Dict:
    """
    Evaluate a case using ProbLog (probabilistic semantics).

    For each article:
      1. Extract all candidate symbols from the case file (Persons, Crimes, etc.).
      2. For each symbol S, query ProbLog:
            violation(articleX, S)
      3. Take the **max** probability over S as the case-level violation probability.
    """
    case_id = case_file.stem.replace('case_', '')
    case_file_abs = str(case_file.resolve()).replace('\\', '/')

    results = {
        "case_id": case_id,
        "article2": 0.0,
        "article3": 0.0,
        "article6": 0.0,
    }

    persons = get_all_persons_in_case(case_file)
    if not persons:
        return results

    # Article 2
    prolog_files = [str(article2_rules.resolve().as_posix()), case_file_abs]
    probs_2 = []
    for p in persons:
        q = f"violation(article2, {p})"
        prob = query_problog(q, prolog_files)
        probs_2.append(prob)
    results["article2"] = max(probs_2) if probs_2 else 0.0

    # Article 3
    prolog_files = [str(article3_rules.resolve().as_posix()), case_file_abs]
    probs_3 = []
    for p in persons:
        q = f"violation(article3, {p})"
        prob = query_problog(q, prolog_files)
        probs_3.append(prob)
    results["article3"] = max(probs_3) if probs_3 else 0.0

    # Article 6
    prolog_files = [str(article6_rules.resolve().as_posix()), case_file_abs]
    probs_6 = []
    for p in persons:
        q = f"violation(article6, {p})"
        prob = query_problog(q, prolog_files)
        probs_6.append(prob)
    results["article6"] = max(probs_6) if probs_6 else 0.0

    return results


# ============================================================
# Metrics
# ============================================================

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


# Article indices in the ECtHR dataset labels array
# labels array: [2, 3, 5, 6, 8, 9, 11, 14, P1-1]
ARTICLE_INDEX = {
    "article2": 0,  # Article 2 is at index 0
    "article3": 1,  # Article 3 is at index 1
    "article6": 3,  # Article 6 is at index 3
}




# ============================================================
# CLI
# ============================================================

def main():
    # Simple pipeline: Load ECtHR dataset and evaluate against it
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: Please install datasets: pip install datasets")
        return

    # Load ECtHR dataset
    print("Loading ECtHR dataset from Hugging Face...")
    ecthr = load_dataset("coastalcph/fairlex", "ecthr")
    test_data = ecthr["test"]  # Use test split
    
    # Setup paths
    prolog_dir = Path("Prolog").resolve()
    problog_dir = Path("ProbLog").resolve()
    article2_rules = Path("ECHR/echr_prolog/article2.pl").resolve()
    article3_rules = Path("ECHR/echr_prolog/article3.pl").resolve()
    article6_rules = Path("ECHR/echr_prolog/article6.pl").resolve()

    if not prolog_dir.exists():
        print(f"Error: Prolog directory not found: {prolog_dir}")
        return

    if not problog_dir.exists():
        problog_dir = prolog_dir

    prolog_case_files = sorted(list(prolog_dir.glob("case_*.pl")))
    problog_case_files = sorted(list(problog_dir.glob("case_*.pl")))
    
    n_cases = min(len(test_data), len(prolog_case_files))
    print(f"Evaluating {n_cases} cases...")
    
    # Extract ground truth from dataset
    # NOTE: We match by position - assumes dataset[i] corresponds to prolog_case_files[i]
    # This assumes case files were generated in the same order as the dataset
    print("Extracting ground truth from ECtHR dataset labels...")
    print("WARNING: Matching by position - assumes dataset order matches case file order")
    ground_truth = {}
    for i in range(n_cases):
        entry = test_data[i]
        labels = entry.get("labels", [])
        case_file = prolog_case_files[i]
        case_id = case_file.stem.replace("case_", "")
        
        ground_truth[case_id] = {
            "article2": ARTICLE_INDEX["article2"] in labels,
            "article3": ARTICLE_INDEX["article3"] in labels,
            "article6": ARTICLE_INDEX["article6"] in labels,
        }
    
    article2_gt = sum(1 for gt in ground_truth.values() if gt["article2"])
    article3_gt = sum(1 for gt in ground_truth.values() if gt["article3"])
    article6_gt = sum(1 for gt in ground_truth.values() if gt["article6"])
    print(f"Ground truth violations: Article 2: {article2_gt}, Article 3: {article3_gt}, Article 6: {article6_gt}")
    print()


    # ============================================================
    # Evaluate Prolog cases
    # ============================================================
    print("Evaluating Prolog cases...")
    prolog_results = []
    for i, case_file in enumerate(prolog_case_files[:n_cases], 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n_cases}")
        prolog_result = evaluate_case_prolog(case_file, article2_rules, article3_rules, article6_rules)
        prolog_results.append(prolog_result)
    print()

    # ============================================================
    # Evaluate ProbLog cases
    # ============================================================
    problog_results = []
    if PROBLOG_AVAILABLE:
        print("Evaluating ProbLog cases...")
        for i, case_file in enumerate(problog_case_files[:n_cases], 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{n_cases}")
            problog_result = evaluate_case_problog(case_file, article2_rules, article3_rules, article6_rules)
            problog_results.append(problog_result)
        print()

    # ============================================================
    # Debug: Check case ID alignment
    # ============================================================
    print("Checking case ID alignment...")
    prolog_article2_cases = [r["case_id"] for r in prolog_results if r["article2"]]
    gt_article2_cases = [case_id for case_id, gt in ground_truth.items() if gt.get("article2")]
    print(f"Prolog found Article 2 violations in cases: {prolog_article2_cases[:10]}...")
    print(f"Ground truth Article 2 violations in cases: {gt_article2_cases[:10]}...")
    overlap = set(prolog_article2_cases) & set(gt_article2_cases)
    print(f"Overlap: {len(overlap)} cases")
    if len(overlap) == 0 and len(prolog_article2_cases) > 0:
        print("WARNING: No overlap! Case IDs might be mismatched.")
    print()

    # ============================================================
    # Compute statistics against ECtHR labels
    # ============================================================
    print("Computing statistics...")
    articles = ["article2", "article3", "article6"]
    comparison = {}

    for article in articles:
        # Get ground truth from ECtHR dataset
        gt = [ground_truth.get(str(r["case_id"]), {}).get(article, False) for r in prolog_results]

        # Prolog predictions
        prolog_pred = [r[article] for r in prolog_results]
        prolog_metrics = compute_metrics(prolog_pred, gt)

        # ProbLog predictions (threshold 0.75)
        if PROBLOG_AVAILABLE:
            problog_probs = [r[article] for r in problog_results]
            problog_pred = [p >= 0.75 for p in problog_probs]
            problog_metrics = compute_metrics(problog_pred, gt)
        else:
            problog_probs = []
            problog_pred = []
            problog_metrics = None

        comparison[article] = {
            "prolog": prolog_metrics,
            "problog": problog_metrics,
            "ground_truth_count": sum(gt),
            "prolog_predicted_count": sum(prolog_pred),
            "problog_predicted_count": sum(problog_pred) if PROBLOG_AVAILABLE else None,
            "prolog_predictions": prolog_pred,
            "problog_predictions": problog_pred,
            "problog_probabilities": problog_probs,
        }

    # Save results
    output_data = {
        "prolog_results": prolog_results,
        "problog_results": problog_results if PROBLOG_AVAILABLE else [],
        "comparison": comparison,
        "ground_truth_source": "ecthr_dataset",
    }

    output_file = "comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    for article in articles:
        print(f"\n{article.upper()}:")
        comp = comparison[article]
        print(f"  Ground truth violations: {comp['ground_truth_count']}")
        print(f"  Prolog predicted violations: {comp['prolog_predicted_count']}")
        if PROBLOG_AVAILABLE:
            print(f"  ProbLog predicted violations: {comp['problog_predicted_count']}")

        print(f"\n  Prolog Metrics (vs Ground Truth):")
        pm = comp['prolog']
        print(f"    Precision: {pm['precision']:.3f}")
        print(f"    Recall:    {pm['recall']:.3f}")
        print(f"    F1:        {pm['f1']:.3f}")
        print(f"    Accuracy:  {pm['accuracy']:.3f}")
        print(f"    TP: {pm['tp']}, FP: {pm['fp']}, TN: {pm['tn']}, FN: {pm['fn']}")

        if comp['problog']:
            print(f"\n  ProbLog Metrics (vs Ground Truth):")
            pbm = comp['problog']
            print(f"    Precision: {pbm['precision']:.3f}")
            print(f"    Recall:    {pbm['recall']:.3f}")
            print(f"    F1:        {pbm['f1']:.3f}")
            print(f"    Accuracy:  {pbm['accuracy']:.3f}")
            print(f"    TP: {pbm['tp']}, FP: {pbm['fp']}, TN: {pbm['tn']}, FN: {pbm['fn']}")

    print(f"\nResults saved to: comparison_results.json")


if __name__ == "__main__":
    main()
