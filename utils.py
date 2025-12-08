# utils.py
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from huggingface_hub.utils import validate_repo_id, HFValidationError


def get_ecthr_dataset(config_name="ecthr"):
    """
    Load a subset of the Fairlex dataset.
    """
    return load_dataset("coastalcph/fairlex", config_name)


def load_model_and_tokenizer(model_name):
    """
    Load an HF causal LM for inference.
    """
    # Normalize the path to ensure it's recognized as a local path
    model_path = Path(model_name).resolve()
    is_local = model_path.exists() and model_path.is_dir()
    
    # Convert to absolute path string with forward slashes (works on Windows too)
    # This helps HuggingFace recognize it as a local path
    if is_local:
        model_name = str(model_path).replace("\\", "/")
    
    # Dtype selection: BF16 > FP16 > FP32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Use local_files_only for local paths to avoid repo ID validation
    load_kwargs = {
    "torch_dtype": dtype,
    "device_map": "auto",
    }

    if is_local:
        load_kwargs["local_files_only"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()

    tokenizer_kwargs = {}
    if is_local:
        tokenizer_kwargs["local_files_only"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Simple pad-token handling
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"

    return model, tokenizer


def query_local_model(
    model,
    tokenizer,
    messages,
    temperature=0.0,
    seed=42,
    max_new_tokens=5,
):
    """
    Generic chat-style generation.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    # Use chat template
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
    do_sample = temperature > 0.0

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = model_inputs.input_ids.shape[1]
    generated_text = tokenizer.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]

    return generated_text.strip()


def batch_query_local_model(
    model,
    tokenizer,
    batch_messages,
    temperature=0.0,
    seed=42,
    max_new_tokens=5,
):
    """
    Batched chat-style generation.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    # Convert messages → prompt text
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in batch_messages
        ]
    else:
        texts = []
        for msgs in batch_messages:
            t = ""
            for m in msgs:
                role = m.get("role", "user").upper()
                content = m.get("content", "")
                t += f"{role}: {content}\n"
            t += "ASSISTANT:"
            texts.append(t)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    torch.manual_seed(seed)
    do_sample = temperature > 0.0

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    completions = tokenizer.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
    )

    return [c.strip() for c in completions]
