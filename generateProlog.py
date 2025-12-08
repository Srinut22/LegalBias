#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate two kinds of KBs from case summaries in ./Summary:

1. Deterministic Prolog KB  (./Prolog/case_<id>.pl)
   - Contains only "verified" facts, i.e. predicates whose P(Yes) >= THRESHOLD.

2. ProbLog KB (./ProbLog/case_<id>.pl)
   - Contains ALL grounded predicates with their confidence:
       p_yes::predicate(args).

Where p_yes is computed from the model's token logits for "Yes" vs "No".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROLOG_DIR = "./Prolog"     # deterministic facts
PROBLOG_DIR = "./Problog"   # probabilistic facts
THRESHOLD = 0.6             # baseline minimum P(Yes) to treat fact as "verified"

# For "institutional" / Article 6-style predicates where evidence is often
# more implicit / described via fairness reasoning, not explicit phrases.
INSTITUTIONAL_PREDICATES = {
    "independent_tribunal",
    "impartial_tribunal",
    "tribunal_established_by_law",
    "hearing_within_reasonable_time",
    "fair_hearing",
    "public_judgment_or_justified_exclusion",
}
INSTITUTIONAL_THRESHOLD = 0.5  # can tune separately from THRESHOLD

_model = None
_tokenizer = None


# ============================================================
# 1. Model loading + helpers
# ============================================================

def load_qwen_model():
    """Load the Qwen model and tokenizer from Hugging Face (downloads if needed)."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        print(f"[INFO] Loading Qwen model from Hugging Face: {MODEL_NAME}")

        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("[INFO] Using CUDA with bfloat16")
        elif torch.cuda.is_available():
            dtype = torch.float16
            print("[INFO] Using CUDA with float16")
        else:
            dtype = torch.float32
            print("[INFO] Using CPU with float32")

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
        )
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
        if getattr(_model.config, "pad_token_id", None) is None:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        _tokenizer.padding_side = "left"

        print("[INFO] Model and tokenizer loaded.")
    else:
        print("[INFO] Using already loaded Qwen model")

    return _model, _tokenizer


def query_qwen(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    seed: int = 42,
    max_new_tokens: int = 512,
) -> str:
    """
    Query Qwen with system + user prompts and return the generated text.
    Used for entity extraction (not for yes/no scoring).
    """
    model, tokenizer = load_qwen_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = ""
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        text += "ASSISTANT:"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    do_sample = temperature > 0.0

    generate_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**generate_kwargs)

    input_length = model_inputs.input_ids.shape[1]
    generated_text = tokenizer.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]

    return generated_text.strip()


def yes_no_probs(system_prompt: str, user_prompt: str) -> Tuple[float, float]:
    """
    Compute P(Yes), P(No) for the question encoded in system+user prompt,
    by inspecting the logits for the first assistant token.

    We restrict to two candidate tokens: " Yes" and " No" (with leading space).
    """
    model, tokenizer = load_qwen_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = ""
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        text += "ASSISTANT:"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**model_inputs)
        # logits shape: [batch, seq_len, vocab_size]
        logits = outputs.logits[0, -1, :]  # last position (assistant's first token)

    # Encode " Yes" and " No" as candidate tokens
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)

    if not yes_ids or not no_ids:
        raise RuntimeError("Could not encode ' Yes' or ' No' tokens with tokenizer")

    yes_id = yes_ids[-1]
    no_id = no_ids[-1]

    selected_logits = torch.stack([logits[yes_id], logits[no_id]])  # [2]
    probs = torch.softmax(selected_logits, dim=-1)                  # [2]

    p_yes = float(probs[0].item())
    p_no = float(probs[1].item())
    return p_yes, p_no


# ============================================================
# 2. Predicates + Prolog helpers
# ============================================================

@dataclass
class Predicate:
    name: str
    arg_types: Tuple[str, ...]   # e.g. ("Person",) or ("Person", "Crime")
    description: str             # with {Person}, {Crime}, etc. placeholders

    def instantiate(self, **arg_values: str) -> str:
        return self.description.format(**arg_values)


predicates = [
    Predicate(
        name="state_responsible_for_death",
        arg_types=("Person",),
        description=(
            "The State (through its agents, authorities, or omissions) is in whole "
            "or in substantial part responsible for {Person}'s death. This includes "
            "deaths caused directly by state agents (e.g., police, military) or by "
            "the State's failure to protect {Person} from a real and immediate risk "
            "to life that it knew or ought to have known about."
        ),
    ),

    Predicate(
        name="life_taken_intentionally",
        arg_types=("Person",),
        description=(
            "{Person}'s life was taken intentionally (deliberately and not by accident). "
            "This requires that {Person} died and that someone deliberately used lethal "
            "or clearly life-threatening force, or knowingly created a very high risk "
            "of {Person}'s death."
        ),
    ),

    Predicate(
        name="death_resulted_from_force",
        arg_types=("Person",),
        description=(
            "{Person} died as a result of force used against them, such as a police or "
            "military operation, use of firearms, physical restraint, or other "
            "coercive measures. The focus is on whether the force caused {Person}'s "
            "death, regardless of whether the death was intended."
        ),
    ),

    Predicate(
        name="force_used_against",
        arg_types=("Person",),
        description=(
            "Force (e.g., physical force, firearms, explosives, or other coercive "
            "measures) was used against {Person} by state agents or with the State's "
            "authority or acquiescence."
        ),
    ),

    Predicate(
        name="force_absolutely_necessary",
        arg_types=("Person",),
        description=(
            "The degree and manner of force used against {Person} were no more than "
            "absolutely necessary in the circumstances, taking into account the aim "
            "pursued, the risks involved, and whether less harmful alternatives were "
            "reasonably available."
        ),
    ),

    Predicate(
        name="defence_from_unlawful_violence",
        arg_types=("Person",),
        description=(
            "Force was used for the purpose of defending {Person} or another individual "
            "from unlawful violence, within the meaning of Article 2(2)(a)."
        ),
    ),

    Predicate(
        name="effect_lawful_arrest",
        arg_types=("Person",),
        description=(
            "Force was used for the purpose of effecting a lawful arrest of {Person} "
            "or preventing {Person}'s escape at the moment of arrest, within the "
            "meaning of Article 2(2)(b)."
        ),
    ),

    Predicate(
        name="prevent_escape_of_lawfully_detained_individual",
        arg_types=("Person",),
        description=(
            "Force was used to prevent {Person}'s escape from lawful detention "
            "(for example, prison, police custody, or other lawful deprivation of "
            "liberty), within the meaning of Article 2(2)(b)."
        ),
    ),

    Predicate(
        name="quell_riot_or_insurrection",
        arg_types=("Person",),
        description=(
            "Force used against {Person} formed part of action taken for the purpose "
            "of lawfully quelling a riot or insurrection, within the meaning of "
            "Article 2(2)(c)."
        ),
    ),

    # === Death penalty / conviction predicates used in judicial_execution/1 ===

    Predicate(
        name="convicted_of",
        arg_types=("Person", "Crime"),
        description=(
            "{Person} was found guilty of {Crime} by a court or other competent judicial "
            "body after formal proceedings resulting in a conviction."
        ),
    ),

    Predicate(
        name="death_penalty_provided_by_law",
        arg_types=("Crime",),
        description=(
            "At the relevant time and place, the applicable law provided that {Crime} "
            "is punishable by the death penalty."
        ),
    ),

    Predicate(
        name="state_responsible_for_mistreatment",
        arg_types=("Person",),
        description=(
            "The State (through its agents, authorities, or with its knowledge and "
            "acquiescence) is responsible for the treatment of {Person}. This includes "
            "acts committed directly by state officials or by private individuals where "
            "the State knew or ought to have known and failed to prevent or stop them."
        ),
    ),

    Predicate(
        name="torture",
        arg_types=("Person",),
        description=(
            "{Person} was subjected to torture. This means the intentional infliction "
            "of very severe physical or mental pain or suffering on {Person} by or "
            "with the consent or acquiescence of public officials, for purposes such "
            "as obtaining information or a confession, punishment, intimidation, or "
            "discrimination."
        ),
    ),

    Predicate(
        name="inhuman_treatment",
        arg_types=("Person",),
        description=(
            "{Person} was subjected to inhuman treatment or punishment. This involves "
            "serious physical or mental suffering, humiliation, or distress inflicted "
            "on {Person} by or attributable to the State, going beyond the level of "
            "hardship or distress inevitably connected with lawful measures."
        ),
    ),

    Predicate(
        name="degrading_treatment",
        arg_types=("Person",),
        description=(
            "{Person} was subjected to degrading treatment or punishment. This means "
            "that {Person} was humiliated or debased in a way that grossly offended "
            "their human dignity or aroused feelings of fear, anguish, or inferiority, "
            "and that this treatment is attributable to the State."
        ),
    ),

    Predicate(
        name="trial",
        arg_types=("Person",),
        description=(
            "{Person} was the subject of criminal or civil proceedings before a tribunal "
            "or court determining their rights, obligations, or a criminal charge."
        ),
    ),

    Predicate(
        name="independent_tribunal",
        arg_types=("Person",),
        description=(
            "The tribunal that heard {Person}'s case was independent of the executive "
            "and the parties, both in law and in practice."
        ),
    ),

    Predicate(
        name="impartial_tribunal",
        arg_types=("Person",),
        description=(
            "The tribunal that heard {Person}'s case was impartial, meaning there were "
            "no legitimate doubts as to the judges' neutrality and lack of bias."
        ),
    ),

    Predicate(
        name="tribunal_established_by_law",
        arg_types=("Person",),
        description=(
            "The tribunal that heard {Person}'s case was established by law, with its "
            "composition, jurisdiction, and procedure based on legal rules rather than "
            "ad hoc or arbitrary appointment."
        ),
    ),

    Predicate(
        name="hearing_within_reasonable_time",
        arg_types=("Person",),
        description=(
            "{Person}'s case was heard and determined within a reasonable time, taking "
            "into account factors such as complexity of the case, conduct of the "
            "authorities, and conduct of {Person}."
        ),
    ),

    Predicate(
        name="fair_hearing",
        arg_types=("Person",),
        description=(
            "{Person} had a fair hearing before the tribunal, including equality of arms, "
            "the ability to present arguments and evidence, and an overall procedure that "
            "respected the adversarial principle."
        ),
    ),

    Predicate(
        name="public_judgment_or_justified_exclusion",
        arg_types=("Person",),
        description=(
            "The judgment in {Person}'s case was pronounced publicly, or any exclusion "
            "of the public from the proceedings or from the pronouncement of the "
            "judgment was justified under the exceptions allowed by Article 6(1)."
        ),
    ),

    # === Article 6(2): Presumption of innocence ===

    Predicate(
        name="charged",
        arg_types=("Person", "Crime"),  # Crime here stands in for "Offence"
        description=(
            "{Person} was charged with a criminal offence {Crime} within the meaning of "
            "Article 6(2), i.e. there was an official notification or equivalent measure "
            "informing {Person} of an allegation of criminal wrongdoing."
        ),
    ),

    Predicate(
        name="presumed_innocent",
        arg_types=("Person",),
        description=(
            "{Person} was treated as innocent until proved guilty according to law, in "
            "the sense that public authorities and courts did not declare or imply that "
            "{Person} was guilty before a final conviction."
        ),
    ),

    # === Article 6(3): Minimum rights in criminal proceedings ===

    Predicate(
        name="informed_promptly_in_language_understood_of_nature_and_cause",
        arg_types=("Person",),
        description=(
            "{Person} was informed promptly, and in a language which they understood, of "
            "the nature and cause of the accusation against them."
        ),
    ),

    Predicate(
        name="adequate_time_and_facilities_for_defence",
        arg_types=("Person",),
        description=(
            "{Person} had adequate time and facilities to prepare their defence, "
            "including access to the case file and the ability to communicate with "
            "their lawyer in confidence."
        ),
    ),

    Predicate(
        name="effective_legal_assistance",
        arg_types=("Person",),
        description=(
            "{Person} had effective legal assistance for their defence, either by "
            "choosing a lawyer or having one assigned, and the assistance was practical "
            "and effective rather than purely formal."
        ),
    ),

    Predicate(
        name="relevant_witness_evidence",
        arg_types=("Person",),
        description=(
            "Evidence from witnesses whose testimony was important or potentially "
            "decisive for the case against {Person} was relied upon or at issue in the "
            "proceedings."
        ),
    ),

    Predicate(
        name="opportunity_to_examine_witnesses_on_equal_terms",
        arg_types=("Person",),
        description=(
            "{Person} had an adequate and proper opportunity to examine or have examined "
            "witnesses against them, and to obtain the attendance and examination of "
            "witnesses on their behalf under the same conditions as witnesses against them."
        ),
    ),

    Predicate(
        name="needs_interpreter",
        arg_types=("Person",),
        description=(
            "{Person} did not sufficiently understand or speak the language of the court "
            "and therefore required the assistance of an interpreter in the proceedings."
        ),
    ),

    Predicate(
        name="interpreter_provided_free_of_charge",
        arg_types=("Person",),
        description=(
            "{Person} was provided with the assistance of an interpreter, free of charge, "
            "for the purposes of the criminal proceedings, including understanding the "
            "accusation and participating effectively in the hearing."
        ),
    ),
]


def generate_prolog_fact(name: str, args: List[str]) -> str:
    """Return a Prolog fact `name(args).`."""
    cleaned_args = [a.replace(" ", "_") for a in args]
    return f"{name}({', '.join(cleaned_args)})."


def prolog_term_str(name: str, args: List[str]) -> str:
    """Return a Prolog term string without trailing '.', e.g. name(args)."""
    cleaned_args = [a.replace(" ", "_") for a in args]
    return f"{name}({', '.join(cleaned_args)})"


# ============================================================
# 3. Entity extraction (Person, Crime)
# ============================================================

def extract_entities(case_text: str) -> Dict[str, List[str]]:
    """
    Use Qwen to extract Person and Crime entities from a case summary.
    """
    system_prompt = """You are a legal expert extracting entities from European Court of Human Rights case summaries.

Extract relevant entities from the case summary. Return ONLY a JSON dictionary with the following structure:
{
    "Person": [...],
    "Crime": [...]
}
"""

    user_prompt = f"""Extract Person and Crime entities from this case SUMMARY:

{case_text}

Return ONLY a JSON dictionary with keys "Person" and "Crime" as described."""

    try:
        print("[INFO] Querying Qwen for entity extraction...")
        response = query_qwen(system_prompt, user_prompt, max_new_tokens=512)
        response = response.strip()

        # Strip markdown fences if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Extract first JSON object
        json_start = response.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in response")

        brace_count = 0
        json_end = json_start
        for i in range(json_start, len(response)):
            if response[i] == "{":
                brace_count += 1
            elif response[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        json_str = response[json_start:json_end]
        print(f"[INFO] Extracted JSON snippet: {json_str[:200]}...")

        data = json.loads(json_str)

        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                return [x]
            if x is None:
                return []
            return list(x)

        persons = ensure_list(data.get("Person", []))
        crimes = ensure_list(data.get("Crime", []))

        if not persons:
            persons = ["applicant"]

        result = {
            "Person": persons,
            "Crime": crimes,
        }
        print(f"[INFO] Persons: {result['Person']}")
        print(f"[INFO] Crimes:  {result['Crime']}")
        return result

    except Exception as e:
        print(f"[ERROR] Entity extraction failed: {e}")
        try:
            print(f"[INFO] Raw response: {response[:200]}...")
        except NameError:
            pass
        return {
            "Person": ["applicant"],
            "Crime": [],
        }


# ============================================================
# 4. Groundings + probabilistic verification
# ============================================================

def generate_groundings(predicate: Predicate, entities: Dict[str, List[str]]) -> List[Tuple[str, ...]]:
    """
    Generate all valid groundings for a predicate from extracted entities:
      - "Person" -> all persons
      - "Crime"  -> all crimes
    """
    entity_lists: List[List[str]] = []

    for arg_type in predicate.arg_types:
        if arg_type == "Person":
            persons = entities.get("Person", [])
            if not persons:
                return []
            entity_lists.append(persons)
        elif arg_type == "Crime":
            crimes = entities.get("Crime", [])
            if not crimes:
                return []
            entity_lists.append(crimes)
        else:
            # Unknown arg type → skip predicate
            return []

    if not entity_lists:
        return []

    return list(product(*entity_lists))


def build_verification_prompts(
    predicate: Predicate,
    fact_term: str,
    description_instantiated: str,
    case_text: str,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for fact verification, with a more
    permissive reading for institutional predicates (Article 6-style).
    """
    if predicate.name in INSTITUTIONAL_PREDICATES:
        system_prompt = """You are a legal expert analyzing a SUMMARY of a European Court of Human Rights case.

You will be given:
- A factual summary of a case (not the full judgment).
- A proposed Prolog-style fact about that case.
- A natural language description of what that fact means.

Your task is to decide whether the fact is supported by the summary,
taking into account BOTH explicit statements and clear implications
from the way the fairness of the proceedings is described.

RULES:
1. Treat the fact as supported if the summary clearly indicates it
   either explicitly OR through a clear description of the applicant's
   complaint and/or the courts' reasoning about the fairness of the
   proceedings, independence/impartiality of the tribunal, timing, or
   publicity of the judgment.
2. Do NOT rely on speculation that goes beyond what the summary reasonably implies.
3. Ignore information about other people or other proceedings."""
    else:
        system_prompt = """You are a legal expert analyzing a SUMMARY of a European Court of Human Rights case.

You will be given:
- A factual summary of a case (not the full judgment).
- A proposed Prolog-style fact about that case.
- A natural language description of what that fact means.

Your task is to decide STRICTLY whether the fact is explicitly supported by the summary.

RULES:
1. Only consider the fact supported if it is clearly and unambiguously stated in the summary.
2. Do NOT infer extra facts by speculation or legal argument - stay at the factual level.
3. If the summary says something similar but about a different person or a different crime, treat it as NOT supported."""

    user_prompt = f"""CASE SUMMARY:
{case_text}

PROLOG FACT:
{fact_term}

FACT DESCRIPTION:
{description_instantiated}

QUESTION:
Is this fact supported by the summary, under the rules above?

You MUST begin your answer with exactly one word: "Yes" or "No".
Your very first token should be "Yes" or "No". The rest of your answer (if any) does not matter."""
    return system_prompt, user_prompt


def verify_fact_probabilistic(
    predicate: Predicate,
    grounding: Tuple[str, ...],
    case_text: str,
    threshold: float = THRESHOLD,
) -> Tuple[bool, float]:
    """
    Compute P(Yes) and P(No) from model logits and return (is_true, p_yes),
    where is_true = (p_yes >= threshold).

    For predicates in INSTITUTIONAL_PREDICATES, we:
      - use a more permissive prompt (institutional fairness reasoning),
      - and a potentially different threshold (INSTITUTIONAL_THRESHOLD).
    """
    args_list = list(grounding)
    fact_term = prolog_term_str(predicate.name, args_list)

    arg_values = {arg_type: value for arg_type, value in zip(predicate.arg_types, grounding)}
    description_instantiated = predicate.instantiate(**arg_values)

    system_prompt, user_prompt = build_verification_prompts(
        predicate,
        fact_term,
        description_instantiated,
        case_text,
    )

    p_yes, p_no = yes_no_probs(system_prompt, user_prompt)

    # Choose effective threshold
    effective_threshold = INSTITUTIONAL_THRESHOLD if predicate.name in INSTITUTIONAL_PREDICATES else threshold
    is_true = p_yes >= effective_threshold

    print(f"    [PROB] {fact_term} → p_yes={p_yes:.3f}, p_no={p_no:.3f}, "
          f"thr={effective_threshold:.2f}, is_true={is_true}")
    return is_true, p_yes


# ============================================================
# 5. Per-summary KB generation
# ============================================================

def generate_kbs_for_summary(case_text: str) -> Tuple[List[str], List[str]]:
    """
    Given a case summary, extract entities and verify all predicate groundings.

    Returns:
        (prolog_facts, problog_facts)

        - prolog_facts: only verified facts (p_yes >= threshold), as `pred(args).`
        - problog_facts: all facts with probabilities, as `p::pred(args).`
    """
    print("=" * 60)
    print("Step 1: Entity extraction")
    print("=" * 60)
    entities = extract_entities(case_text)

    print("\n" + "=" * 60)
    print("Step 2: Groundings + probabilistic verification")
    print("=" * 60)

    prolog_facts: List[str] = []
    problog_facts: List[str] = []
    total_groundings = 0
    total_verified = 0

    for i, predicate in enumerate(predicates, start=1):
        print(f"\n[{i}/{len(predicates)}] Predicate: {predicate.name}")
        print(f"    Arg types: {predicate.arg_types}")

        groundings = generate_groundings(predicate, entities)
        print(f"    Groundings: {len(groundings)}")
        total_groundings += len(groundings)

        for j, grounding in enumerate(groundings, start=1):
            args_list = list(grounding)
            term = prolog_term_str(predicate.name, args_list)
            print(f"    ({j}/{len(groundings)}) Checking {term}")
            is_true, p_yes = verify_fact_probabilistic(predicate, grounding, case_text)

            # ProbLog fact: always include
            # Clamp probabilities a bit away from 0 and 1 if you like
            p_clamped = max(0.001, min(0.999, p_yes))
            problog_facts.append(f"{p_clamped:.3f}::{term}.")

            # Prolog fact: only if verified
            if is_true:
                fact = term + "."
                prolog_facts.append(fact)
                total_verified += 1
                print(f"Verified {fact}")
            else:
                print("Not verified")

    # deduplicate and sort
    prolog_facts = sorted(set(prolog_facts))
    problog_facts = sorted(set(problog_facts))

    return prolog_facts, problog_facts


# ============================================================
# 6. Batch over ./Summary → ./Prolog + ./ProbLog
# ============================================================

def extract_case_id_from_filename(path: Path) -> str:
    """
    Extract the first number in the filename stem as case_id.
    If no digits are found, return the whole stem.
    """
    stem = path.stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else stem


def process_all_summaries(
    summaries_dir: str = "./Summary",
    prolog_dir: str = PROLOG_DIR,
    problog_dir: str = PROBLOG_DIR,
) -> None:
    summaries_path = Path(summaries_dir)
    prolog_path = Path(prolog_dir)
    problog_path = Path(problog_dir)
    prolog_path.mkdir(parents=True, exist_ok=True)
    problog_path.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(summaries_path.glob("*.txt"))
    print(f"[INFO] Found {len(summary_files)} summary files in {summaries_dir}")
    for summary_file in summary_files:
        case_id = extract_case_id_from_filename(summary_file)
        print("\n" + "#" * 60)
        print(f"[INFO] Processing {summary_file} (case_id={case_id})")
        print("#" * 60 + "\n")

        with open(summary_file, "r", encoding="utf-8") as f:
            case_text = f.read()

        prolog_facts, problog_facts = generate_kbs_for_summary(case_text)

        prolog_out = prolog_path / f"case_{case_id}.pl"
        print(f"[INFO] Writing {len(prolog_facts)} Prolog facts to {prolog_out}")
        with open(prolog_out, "w", encoding="utf-8") as f:
            for fact in prolog_facts:
                f.write(fact + "\n")

        problog_out = problog_path / f"case_{case_id}.pl"
        print(f"[INFO] Writing {len(problog_facts)} ProbLog facts to {problog_out}")
        with open(problog_out, "w", encoding="utf-8") as f:
            for fact in problog_facts:
                f.write(fact + "\n")

        print(f"[INFO] Done case_id={case_id}")


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":
    process_all_summaries()
