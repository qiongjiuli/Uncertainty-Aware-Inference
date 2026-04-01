"""
Model loading utilities for all quantization configurations.

Supported configs:
    FP16        : standard half-precision baseline
    GPTQ_INT8   : 8-bit GPTQ via GPTQModel (quantized on-the-fly)
    GPTQ_INT4   : 4-bit GPTQ via GPTQModel (pre-quantized checkpoint)
    AWQ_INT4    : 4-bit AWQ via AutoAWQ
    NF4         : 4-bit NormalFloat via bitsandbytes (QLoRA-style)

NOTE: Uses GPTQModel (not AutoGPTQ) — the newer maintained package.
    pip install gptqmodel

Usage:
    from src.quantization.loaders import load_model
    model, tokenizer = load_model("meta-llama/Llama-2-7b-hf", "GPTQ_INT4")
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hub IDs for pre-quantized GPTQ/AWQ checkpoints
# ---------------------------------------------------------------------------

PREQUANTIZED_IDS: dict[str, dict[str, str]] = {
    "meta-llama/Llama-2-7b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-7B-GPTQ",
        "GPTQ_INT8": "TheBloke/Llama-2-7B-GPTQ",   # same repo, different bits param
        "AWQ_INT4" : "TheBloke/Llama-2-7B-AWQ",
    },
    "mistralai/Mistral-7B-v0.1": {
        "GPTQ_INT4": "TheBloke/Mistral-7B-v0.1-GPTQ",
        "GPTQ_INT8": "TheBloke/Mistral-7B-v0.1-GPTQ",
        "AWQ_INT4" : "TheBloke/Mistral-7B-v0.1-AWQ",
    },
    "meta-llama/Llama-2-13b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-13B-GPTQ",
        "GPTQ_INT8": "TheBloke/Llama-2-13B-GPTQ",
        "AWQ_INT4" : "TheBloke/Llama-2-13B-AWQ",
    },
}


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_fp16(
    model_id: str,
    device_map: str = "auto",
) -> tuple:
    """Standard FP16 baseline."""
    logger.info(f"Loading FP16: {model_id}")
    tokenizer = _load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_gptq_int4(
    model_id: str,
    device_map: str = "auto",
    quantized_model_id: Optional[str] = None,
) -> tuple:
    """
    4-bit GPTQ via GPTQModel.
    Note: GPTQModel uses device="cuda:0" not device="auto"
    """
    try:
        from gptqmodel import GPTQModel
    except ImportError:
        raise ImportError(
            "GPTQModel not installed. Run: pip install gptqmodel"
        )

    quant_id = quantized_model_id or PREQUANTIZED_IDS.get(model_id, {}).get("GPTQ_INT4")
    if quant_id is None:
        raise ValueError(
            f"No pre-quantized GPTQ_INT4 checkpoint found for {model_id}. "
            "Provide quantized_model_id explicitly."
        )

    logger.info(f"Loading GPTQ_INT4 via GPTQModel: {quant_id}")
    tokenizer = _load_tokenizer(model_id)

    # GPTQModel requires explicit device string, not "auto"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(
        quant_id,
        device=device,
    )
    model.eval()
    return model, tokenizer


def load_gptq_int8(
    model_id: str,
    device_map: str = "auto",
    quantized_model_id: Optional[str] = None,
) -> tuple:
    """
    8-bit GPTQ via GPTQModel — quantizes on-the-fly.
    Uses WikiText-2 for calibration (C4 dataset scripts no longer supported).
    Takes 10-20 minutes on A100.
    """
    try:
        from gptqmodel import GPTQModel
        from gptqmodel.quantization import QuantizeConfig
    except ImportError:
        raise ImportError(
            "GPTQModel not installed. Run: pip install gptqmodel"
        )

    logger.info(f"Loading GPTQ_INT8 via GPTQModel: {model_id}")
    tokenizer = _load_tokenizer(model_id)

    quant_config = QuantizeConfig(
        bits       = 8,
        group_size = 128,
        desc_act   = False,
    )

    # GPTQModel requires explicit device string
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = GPTQModel.load(
        model_id,
        quantize_config=quant_config,
    )

    # Use WikiText-2 for calibration — C4 dataset scripts no longer supported
    logger.info("Loading WikiText-2 calibration data (128 samples)...")
    from datasets import load_dataset
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    samples = []
    for item in wikitext:
        text = item["text"].strip()
        if len(text) < 50:        # skip short/empty lines
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=False,
        )
        samples.append(enc["input_ids"])
        if len(samples) >= 128:
            break

    logger.info(f"Quantizing to INT8 with {len(samples)} calibration samples...")
    model.quantize(samples)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_awq_int4(
    model_id: str,
    device_map: str = "auto",
    quantized_model_id: Optional[str] = None,
) -> tuple:
    """4-bit AWQ via AutoAWQ."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("Install autoawq: pip install autoawq")

    quant_id = quantized_model_id or PREQUANTIZED_IDS.get(model_id, {}).get("AWQ_INT4")
    if quant_id is None:
        raise ValueError(
            f"No pre-quantized AWQ_INT4 checkpoint found for {model_id}. "
            "Provide quantized_model_id explicitly."
        )

    logger.info(f"Loading AWQ_INT4: {quant_id}")
    tokenizer = _load_tokenizer(model_id)
    model = AutoAWQForCausalLM.from_quantized(
        quant_id,
        fuse_layers=False,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def load_nf4(
    model_id: str,
    device_map: str = "auto",
) -> tuple:
    """4-bit NormalFloat via bitsandbytes (QLoRA-style)."""
    logger.info(f"Loading NF4: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = _load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

LOADERS = {
    "FP16"      : load_fp16,
    "GPTQ_INT8" : load_gptq_int8,
    "GPTQ_INT4" : load_gptq_int4,
    "AWQ_INT4"  : load_awq_int4,
    "NF4"       : load_nf4,
}


def load_model(
    model_id: str,
    precision: str,
    device_map: str = "auto",
    quantized_model_id: Optional[str] = None,
) -> tuple:
    """
    Load a model in the specified precision.

    Args:
        model_id            : HuggingFace base model ID
        precision           : FP16 | GPTQ_INT8 | GPTQ_INT4 | AWQ_INT4 | NF4
        device_map          : device placement (default "auto")
        quantized_model_id  : optional override for pre-quantized Hub checkpoint

    Returns:
        (model, tokenizer)

    Example:
        model, tok = load_model("meta-llama/Llama-2-7b-hf", "GPTQ_INT4")
        model, tok = load_model("meta-llama/Llama-2-7b-hf", "GPTQ_INT8")
        model, tok = load_model("meta-llama/Llama-2-7b-hf", "AWQ_INT4")
    """
    if precision not in LOADERS:
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Choose from: {list(LOADERS.keys())}"
        )

    loader = LOADERS[precision]

    if precision in ("GPTQ_INT4", "GPTQ_INT8", "AWQ_INT4"):
        return loader(model_id, device_map=device_map,
                      quantized_model_id=quantized_model_id)
    return loader(model_id, device_map=device_map)


def free_model(model) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
