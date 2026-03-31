"""
Model loading utilities for all quantization configurations.

Supported configs:
    FP16        : standard half-precision baseline
    GPTQ_INT8   : 8-bit via bitsandbytes load_in_8bit
    GPTQ_INT4   : 4-bit GPTQ via AutoGPTQ
    AWQ_INT4    : 4-bit AWQ via AutoAWQ
    NF4         : 4-bit NormalFloat via bitsandbytes (QLoRA-style)

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
# Hub IDs for pre-quantized checkpoints
# Override in your config if you want different checkpoints.
# ---------------------------------------------------------------------------

PREQUANTIZED_IDS: dict[str, dict[str, str]] = {
    "meta-llama/Llama-2-7b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-7B-GPTQ",
        "AWQ_INT4" : "TheBloke/Llama-2-7B-AWQ",
    },
    "mistralai/Mistral-7B-v0.1": {
        "GPTQ_INT4": "TheBloke/Mistral-7B-v0.1-GPTQ",
        "AWQ_INT4" : "TheBloke/Mistral-7B-v0.1-AWQ",
    },
    "meta-llama/Llama-2-13b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-13B-GPTQ",
        "AWQ_INT4" : "TheBloke/Llama-2-13B-AWQ",
    },
}


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16(
    model_id: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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


def load_gptq_int8(
    model_id: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """8-bit quantization via bitsandbytes."""
    logger.info(f"Loading GPTQ_INT8 (bitsandbytes 8-bit): {model_id}")
    tokenizer = _load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
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
    """4-bit GPTQ via AutoGPTQ."""
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise ImportError("Install auto-gptq: pip install auto-gptq")

    quant_id = quantized_model_id or PREQUANTIZED_IDS.get(model_id, {}).get("GPTQ_INT4")
    if quant_id is None:
        raise ValueError(
            f"No pre-quantized GPTQ_INT4 checkpoint found for {model_id}. "
            "Provide quantized_model_id explicitly."
        )

    logger.info(f"Loading GPTQ_INT4: {quant_id}")
    tokenizer = _load_tokenizer(model_id)
    model = AutoGPTQForCausalLM.from_quantized(
        quant_id,
        use_safetensors=True,
        device_map=device_map,
        inject_fused_attention=False,
        trust_remote_code=False,
    )
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
        fuse_layers=False,        # safer for evaluation
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def load_nf4(
    model_id: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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
        model_id            : HuggingFace model ID (base model)
        precision           : one of FP16 | GPTQ_INT8 | GPTQ_INT4 | AWQ_INT4 | NF4
        device_map          : passed to from_pretrained (default "auto")
        quantized_model_id  : override Hub ID for pre-quantized checkpoints

    Returns:
        (model, tokenizer)
    """
    if precision not in LOADERS:
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Choose from: {list(LOADERS.keys())}"
        )

    loader = LOADERS[precision]

    # GPTQ/AWQ accept an extra kwarg
    if precision in ("GPTQ_INT4", "AWQ_INT4"):
        return loader(model_id, device_map=device_map,
                      quantized_model_id=quantized_model_id)
    return loader(model_id, device_map=device_map)


def free_model(model) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
