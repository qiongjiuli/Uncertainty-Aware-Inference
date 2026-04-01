"""
Dataset loading utilities.

Loads ARC-Challenge, HellaSwag, and TriviaQA into a unified
multiple-choice sample format:
    {
        "question": str,
        "choices":  list[str],
        "answer":   int   (index of correct choice)
    }
"""

from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Loaders for each dataset
# ---------------------------------------------------------------------------

def load_arc_challenge(
    split: str = "validation",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    ARC-Challenge: science MCQ with 4 choices.
    Uncertainty profile: factual, model should be confident on known facts.
    """
    raw = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    samples = []

    for item in raw:
        choices = item["choices"]["text"]
        labels  = item["choices"]["label"]
        answer  = item["answerKey"]

        # Map answer key to index
        if answer in labels:
            answer_idx = labels.index(answer)
        elif answer.isdigit():
            answer_idx = int(answer) - 1
        else:
            continue  # skip malformed samples

        if answer_idx >= len(choices):
            continue

        samples.append({
            "question": item["question"],
            "choices" : choices,
            "answer"  : answer_idx,
            "source"  : "arc_challenge",
        })

    return _subsample(samples, n_samples, seed)


def load_hellaswag(
    split: str = "validation",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    HellaSwag: sentence completion with 4 choices.
    Uncertainty profile: commonsense reasoning with inherent ambiguity.
    """
    raw = load_dataset("hellaswag", split=split)
    samples = []

    for item in raw:
        # Prepend activity label to question for context
        question = f"({item['activity_label']}) {item['ctx']}"
        choices  = item["endings"]
        try:
            answer_idx = int(item["label"])
        except (ValueError, TypeError):
            continue

        if answer_idx >= len(choices):
            continue

        samples.append({
            "question": question,
            "choices" : choices,
            "answer"  : answer_idx,
            "source"  : "hellaswag",
        })

    return _subsample(samples, n_samples, seed)


def load_triviaqa_as_mcq(
    split: str = "validation",
    n_samples: Optional[int] = None,
    n_distractors: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    TriviaQA: converted to MCQ by sampling distractor answers.
    Uncertainty profile: factual QA — model should express high confidence
    on things it knows, high uncertainty on things it doesn't.

    Note: distractors are other answers from the dataset, so they are
    plausible-sounding but incorrect for this question.
    """
    raw = load_dataset("trivia_qa", "rc.nocontext", split=split)
    all_answers = [
        item["answer"]["value"] for item in raw
        if item["answer"]["value"]
    ]
    rng = random.Random(seed)
    samples = []

    for item in raw:
        question    = item["question"]
        correct_ans = item["answer"]["value"]
        if not correct_ans:
            continue

        # Sample distractors (other questions' answers, not this one's)
        pool       = [a for a in all_answers if a != correct_ans]
        distractors= rng.sample(pool, min(n_distractors, len(pool)))

        # Insert correct answer at a random position
        choices     = distractors[:]
        correct_idx = rng.randint(0, len(choices))
        choices.insert(correct_idx, correct_ans)

        samples.append({
            "question": question,
            "choices" : choices,
            "answer"  : correct_idx,
            "source"  : "triviaqa",
        })

    return _subsample(samples, n_samples, seed)


# ---------------------------------------------------------------------------
# KD training dataset (plain text, no labels needed)
# ---------------------------------------------------------------------------

def load_kd_corpus(
    n_samples: int = 2000,
    seed: int = 42,
) -> list[str]:
    """
    Plain text corpus for knowledge distillation training.
    Uses WikiText-2 for generic, clean text.
    """
    raw  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = [
        item["text"].strip()
        for item in raw
        if len(item["text"].strip()) > 50     # skip short/empty lines
    ]
    rng = random.Random(seed)
    rng.shuffle(text)
    return text[:n_samples]


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "arc_challenge": load_arc_challenge,
    "hellaswag"    : load_hellaswag,
    "triviaqa"     : load_triviaqa_as_mcq,
}


def load_dataset_mcq(
    name: str,
    split: str = "validation",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Unified entry point. name must be one of: arc_challenge, hellaswag, triviaqa.
    """
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(DATASET_LOADERS.keys())}"
        )
    loader = DATASET_LOADERS[name]
    return loader(split=split, n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample(
    samples: list[dict],
    n: Optional[int],
    seed: int,
) -> list[dict]:
    if n is None or n >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, n)
