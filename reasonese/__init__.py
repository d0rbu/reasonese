"""Foundations for controlled prompt-authoring experiments."""

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.planning import PromptSpec, build_prompt_specs

__all__ = [
    "Assistant",
    "Author",
    "Channel",
    "Framing",
    "Instruction",
    "PromptSpec",
    "build_prompt_specs",
]
__version__ = "0.1.0"
