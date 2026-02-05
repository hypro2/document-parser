# -*- coding: utf-8 -*-
"""OpenAI API compatible providers (vLLM, OpenAI, etc.)."""

from docparser.providers.openai_api.ocr import OpenAIOcr
from docparser.providers.openai_api.vlm import OpenAIVLM

__all__ = ["OpenAIOcr", "OpenAIVLM"]
