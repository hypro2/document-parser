# -*- coding: utf-8 -*-
"""
OpenAI API compatible OCR provider.

Supports:
- vLLM server with OpenAI compatible API
- OpenAI API directly
- Any OpenAI-compatible endpoint (LocalAI, Text Generation WebUI, etc.)
"""

from __future__ import annotations

import base64
import io
from typing import Literal

from PIL import Image
from openai import OpenAI

from docparser.providers.base import OCRModel
from docparser.text_clean import clean_deepseek_ocr_text


class OpenAIOcr(OCRModel):
    """
    OCR provider using OpenAI-compatible API.
    
    Works with vLLM, OpenAI, or any OpenAI-compatible endpoint.
    """
    
    def __init__(
        self,
        model: str = "deepseek-ocr",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 600,
    ):
        """
        Initialize OpenAI-compatible OCR provider.
        
        Args:
            model: Model name deployed on the server
            base_url: API endpoint URL (None for default OpenAI)
            api_key: API key (None for default OpenAI env var)
            timeout_sec: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
        )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 encoded string."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def ocr(
        self,
        *,
        image: Image.Image,
        mode: Literal["markdown", "free"] = "markdown",
    ) -> str:
        """
        Perform OCR on an image using OpenAI-compatible API.
        
        Args:
            image: PIL Image to process
            mode: OCR mode - "markdown" for structured output, "free" for plain text
            
        Returns:
            Extracted text from the image
        """
        # Build prompt based on mode
        if mode == "markdown":
            prompt = "<|image|><|grounding|>Convert all text in the image to markdown format."
        else:
            prompt = "<|image|>Extract all text from this image."
        
        # Encode image
        base64_image = self._image_to_base64(image)
        
        # Build message with vision content
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
            )
            
            raw_text = response.choices[0].message.content or ""
            
            # Clean up OCR output
            return clean_deepseek_ocr_text(raw_text)
            
        except Exception as e:
            # Return error message for debugging
            return f"[OCR Error: {e}]"
