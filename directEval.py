from pathlib import Path
import json
import time
import re

from utils import (
    get_ecthr_dataset,
    load_model_and_tokenizer,
    query_local_model,
    batch_query_local_model,
)

MODEL_NAME = str(Path(__file__).parent / "Qwen1.5-7B")

# Global index order for labels (length = 9)
ARTICLES = ["2", "3", "5", "6", "8", "9", "11", "14", "P1-1"]

# Subset of articles we actually query Qwen about
TARGET_ARTICLES = ["2", "3", "6"]

SYSTEM_PROMPT = (
    "You are a legal expert specializing in European human rights law. "
    "Your task is BINARY classification. "
    "Given the article definition and case facts, decide whether the proceedings violated the article.\n"
    "CRITICAL: You must be EXTREMELY STRICT. \n"
    "Violations are the EXCEPTION, not the rule.\n"
    "Guidelines:\n"
    "- Only respond 'Yes' if there is EXPLICIT, CLEAR, and UNEQUIVOCAL evidence of a violation.\n"
    "- The violation must directly match the specific requirements stated in the article definition.\n"
    "- If the evidence is ambiguous, unclear, or only partially matches the article requirements, respond 'No'.\n"
    "- When in doubt, respond 'No'. Default to 'No' unless violation is certain.\n"
    "- Not every case violates every article. Most cases violate zero or very few articles.\n\n"
    "Output format:\n"
    "- Respond with exactly ONE WORD: Yes or No.\n"
    "- Do NOT include explanations, punctuation, or extra words."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a legal expert specializing in European human rights law. "
    "Your task is to summarize case facts clearly and concisely.\n"
    "Given the full case facts, write a summary that preserves all legally relevant information "
    "needed to determine whether there was a violation of ECHR articles, but avoids unnecessary detail.\n\n"
    "The summary should be a few paragraphs at most."
)


def load_article_texts():
    """
    Load article definitions from ECHR/articles_text/article{article}.txt
    for all articles in the global ARTICLES list.
    """
    base_dir = Path(__file__).resolve().parent
    articles_dir = base_dir / "ECHR" / "echr_articles"

    article_texts = {}
    for art in ARTICLES:
        filename = f"article{art}.txt"
        path = articles_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            article_texts[art] = f.read().strip()

    return article_texts


def normalize_yes_no(text):
    clean_text = text.strip()
    if not clean_text:
        return 0

    first_word = re.split(r"\s+", clean_text)[0]
    first_word = first_word.strip("!.,'\"").upper()

    if first_word in {"YES", "Y", "YEAH", "YA", "TRUE"}:
        return 1
    if first_word in {"NO", "N", "NOPE", "NAH", "FALSE"}:
        return 0

    return 0


def build_messages(article_definition, case_facts):
    full_user_prompt = f"""Article Definition:
{article_definition}

Case Facts:
{case_facts}

Question: Did the proceedings violate the defined Article above?

IMPORTANT: Be very strict. Only answer 'Yes' if there is explicit, clear evidence of a violation that directly matches the article requirements. If uncertain or if the evidence is ambiguous, answer 'No'. Most cases do not violate most articles.

Answer (Yes or No only):"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_user_prompt},
    ]


def build_summary_messages(case_facts):
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": case_facts},
    ]


def summarize_case(
    model,
    tokenizer,
    case_id,
    case_facts,
    max_new_tokens=256,
    temperature=0.0,
    seed=123,
    summary_dir=None,
):
    messages = build_summary_messages(case_facts)

    start = time.time()
    summary_text = query_local_model(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    summary_time = time.time() - start

    if summary_dir is None:
        summary_dir = Path("summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"case_{case_id}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text, summary_time


def evaluate_case(
    model,
    tokenizer,
    case_id,
    case_facts,
    article_texts,
    temperature=0.1,  # Slight temperature to reduce bias
    seed=42,
    max_new_tokens=10,
):
    """
    Query Qwen for Articles 2, 3, and 6 only (TARGET_ARTICLES),
    then build a 9-dimensional label vector aligned with ARTICLES.

    label_vector[i] = 1 if the article at ARTICLES[i] is predicted violated, else 0.
    """
    start = time.time()

    # Build prompts only for TARGET_ARTICLES (2, 3, 6)
    batch_messages = [
        build_messages(
            article_definition=article_texts[article_id],
            case_facts=case_facts,
        )
        for article_id in TARGET_ARTICLES
    ]

    raw_outputs = batch_query_local_model(
        model=model,
        tokenizer=tokenizer,
        batch_messages=batch_messages,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )

    # Binary yes/no for each of TARGET_ARTICLES
    target_outputs = [normalize_yes_no(text) for text in raw_outputs]

    # Build full 9-dim label vector aligned with ARTICLES
    label_vector = [0] * len(ARTICLES)
    for pred, art in zip(target_outputs, TARGET_ARTICLES):
        if pred == 1:
            idx = ARTICLES.index(art)
            label_vector[idx] = 1

    elapsed = time.time() - start

    return {
        "id": case_id,
        "time": elapsed,
        "labels": label_vector,          # 9-dim vector aligned with ARTICLES
        "target_outputs": target_outputs,  # predictions just for 2,3,6
        "raw_outputs": raw_outputs,        # raw model outputs for 2,3,6
    }


def extract_case_id_from_filename(path: Path) -> str:
    """
    Extract the first number in the filename stem as case_id.
    If no digits are found, return the whole stem.
    Aligned with generateProlog.py.
    """
    stem = path.stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else stem


# --- main execution ---

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
articles = load_article_texts()

base_dir = Path(__file__).resolve().parent
summary_dir = base_dir / "Summary"      # Aligned with generateProlog.py: uses ./Summary

# Auto-discover ALL summaries in ./Summary (aligned with generateProlog.py):
summaries = {}
if summary_dir.exists():
    summary_files = sorted(summary_dir.glob("*.txt"))
    print(f"[INFO] Found {len(summary_files)} summary files in {summary_dir}")
    for summary_path in summary_files:
        case_id_str = extract_case_id_from_filename(summary_path)
        try:
            case_id = int(case_id_str)
        except ValueError:
            # If case_id is not a number, use string as-is
            case_id = case_id_str
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries[case_id] = f.read().strip()
    print(f"[INFO] Loaded {len(summaries)} summaries")
else:
    print(f"Warning: summaries directory not found: {summary_dir}")

results_path = "results_strict_direct_eval.jsonl"

print(f"Loaded {len(summaries)} summaries from {summary_dir}. Evaluating violations...")

# Debug: Test one case first to see what the model outputs
if summaries:
    test_case_id = sorted(summaries.keys())[0]
    test_summary = summaries[test_case_id]
    print(f"\n=== DEBUG: Testing case {test_case_id} ===")
    print(f"Summary length: {len(test_summary)} chars")
    print(f"Summary preview: {test_summary[:200]}...")

    # Test with Article 2 (first in ARTICLES)
    test_messages = build_messages(
        article_definition=articles[ARTICLES[0]],
        case_facts=test_summary,
    )
    test_output = query_local_model(
        model=model,
        tokenizer=tokenizer,
        messages=test_messages,
        temperature=0.0,
        seed=42,
        max_new_tokens=5,
    )
    print(f"Raw model output for Article {ARTICLES[0]}: '{test_output}'")
    print(f"Normalized: {normalize_yes_no(test_output)}")
    print("=" * 50 + "\n")

with open(results_path, "w", encoding="utf-8") as f:
    for case_id in sorted(summaries.keys()):
        summary_text = summaries[case_id]

        print(f"\nEvaluating case {case_id}...")

        # Evaluate the summary against TARGET_ARTICLES (2,3,6),
        # then get a 9-dim label vector.
        summary_eval = evaluate_case(
            model=model,
            tokenizer=tokenizer,
            case_id=case_id,
            case_facts=summary_text,
            article_texts=articles,
        )

        record = {
            "id": case_id,
            "time_summary_violation": summary_eval["time"],
            "labels": summary_eval["labels"],            # 9-dim [0/1,...]
            "target_outputs": summary_eval["target_outputs"],
            "raw_outputs": summary_eval["raw_outputs"],  # raw for 2,3,6
        }

        f.write(json.dumps(record) + "\n")

        # Show which articles were violated (in the 9-article index space)
        violated_articles = [
            ARTICLES[i]
            for i, val in enumerate(summary_eval["labels"])
            if val == 1
        ]
        if violated_articles:
            print(f"  Case {case_id}: Violations detected for Articles {violated_articles}")
        else:
            print(f"  Case {case_id}: No violations detected")

print(f"\nResults saved to {results_path}")
