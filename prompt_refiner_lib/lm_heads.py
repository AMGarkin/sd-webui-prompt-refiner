"""Source guidance for optional output heads; suggestions never bind weights automatically."""

# These repositories describe standard original models, not arbitrary fine-tunes.
# A manual repository remains available because identical dimensions are insufficient.
KNOWN_HEAD_SOURCES = {
    "Custom / fine-tuned model (enter its original repository)": "",
    "Qwen3.5 9B / MOSS-Image (standard)": "Qwen/Qwen3.5-9B",
    "Qwen2.5-VL 7B / Qwen-Image (standard)": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL 8B / Qwen-Image 2.1 (standard)": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3 8B / Flux.2 Klein 9B (standard)": "Qwen/Qwen3-8B",
    "Qwen3 14B (standard)": "Qwen/Qwen3-14B",
    "Qwen3 32B (standard)": "Qwen/Qwen3-32B",
    "Qwen3-VL 32B (standard)": "Qwen/Qwen3-VL-32B-Instruct",
}


class MissingLMHeadError(ValueError):
    """An expected setup requirement, reported as UI guidance without a traceback."""

    def __init__(self, family):
        message = (
            f"{family} requires its original lm_head.weight, which this encoder omits. "
            "Open the LM head section, choose the original model source, download "
            "the head, and select the saved file before refining again. "
        )
        if family == "Qwen3.5":
            message += (
                "For standard MOSS-Image / Qwen3.5 9B, choose "
                "'Qwen3.5 9B / MOSS-Image (standard)' (Qwen/Qwen3.5-9B). "
            )
        message += "For a fine-tuned encoder, use its exact original checkpoint; matching dimensions alone are not enough."
        super().__init__(message)
