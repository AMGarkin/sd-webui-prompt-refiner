"""Sampling defaults and generated-token penalties for local prompt rewriting.

The default profile follows Qwen3-VL-4B-Instruct's VL model-card recommendations.
Presence penalty is optional and counts only generated tokens, so words in the
input idea or system instructions do not become penalized before they are used.
"""

import math

import torch


SAMPLING_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


class GeneratedTokenPresencePenalty:
    """Transformers-compatible logits processor: subtract once per seen output token.

    The penalty is additive and independent of repetition count. It is applied
    per batch/beam row, excludes the complete input prefix, preserves score dtype,
    and never modifies the caller's scores in place.
    """

    def __init__(self, prompt_length, penalty):
        self.prompt_length = int(prompt_length)
        self.penalty = float(penalty)
        if self.prompt_length < 0:
            raise ValueError("Presence penalty requires a nonnegative prompt length.")
        if not math.isfinite(self.penalty) or not 0 <= self.penalty <= 2:
            raise ValueError("Presence penalty must be between 0 and 2.")

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if self.penalty == 0 or generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return torch.where(seen, scores - self.penalty, scores)


def sampling_kwargs(temperature, top_p, top_k, min_p, repetition_penalty,
                    presence_penalty, prompt_length):
    """Make explicit generation options; greedy decoding disables sampling filters.

    Returning all sampling fields prevents a model's generation configuration
    from silently restoring filters disabled in the UI. A presence processor is
    passed explicitly because Forge's Transformers version has no native setting.
    """
    sampling = float(temperature) > 0
    options = {
        "do_sample": sampling,
        "temperature": float(temperature) if sampling else None,
        "top_p": float(top_p) if sampling else None,
        "top_k": int(top_k) if sampling else None,
        "min_p": float(min_p) if sampling and float(min_p) > 0 else None,
        "repetition_penalty": float(repetition_penalty),
    }
    if float(presence_penalty) != 0:
        options["logits_processor"] = [
            GeneratedTokenPresencePenalty(prompt_length, presence_penalty)
        ]
    return options
