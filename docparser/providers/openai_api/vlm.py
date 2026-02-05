# -*- coding: utf-8 -*-
"""
OpenAI API compatible VLM (Vision Language Model) provider.

Supports:
- vLLM server with OpenAI compatible API
- OpenAI API directly (GPT-4V, etc.)
- Any OpenAI-compatible endpoint
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image
from openai import OpenAI

from docparser.providers.base import VisionLanguageModel


class OpenAIVLM(VisionLanguageModel):
    """
    VLM provider using OpenAI-compatible API.
    
    Works with vLLM, OpenAI GPT-4V, or any OpenAI-compatible endpoint.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 600,
        options: dict[str, Any] | None = None,
    ):
        """
        Initialize OpenAI-compatible VLM provider.
        
        Args:
            model: Model name deployed on the server
            base_url: API endpoint URL (None for default OpenAI)
            api_key: API key (None for default OpenAI env var)
            timeout_sec: Request timeout in seconds
            options: Additional options (temperature, max_tokens, etc.)
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.options = options or {}
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
        )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 encoded string."""
        buffer = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def generate(
        self,
        *,
        prompt: str,
        images: list[Image.Image] | None = None,
    ) -> str:
        """
        Generate text response from prompt and optional images.
        
        Args:
            prompt: Text prompt for the model
            images: Optional list of PIL Images to include
            
        Returns:
            Generated text response
        """
        # Build content array
        content = []
        
        # Add images if provided
        if images:
            for img in images:
                base64_image = self._image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Extract options
        temperature = self.options.get("temperature", 0.1)
        max_tokens = self.options.get("max_tokens", self.options.get("num_predict", 512))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            return f"[VLM Error: {e}]"
