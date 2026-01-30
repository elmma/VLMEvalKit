"""Qwen3-VL with linear attention (distilled) and LoRA support for VLMEvalKit."""

import json
import os
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from .base import BaseModel
from .qwen3_vl.prompt import Qwen3VLPromptMixin

# Add project root to path for importing our modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def ensure_file_url(path: str) -> str:
    """Ensure path has file:// prefix for Qwen3-VL processor."""
    prefixes = ['http://', 'https://', 'file://', 'data:']
    if any(path.startswith(p) for p in prefixes):
        return path
    if os.path.exists(path):
        return 'file://' + path
    raise ValueError(f'Invalid path: {path}')


class Qwen3VLLinearAttention(Qwen3VLPromptMixin, BaseModel):
    """Qwen3-VL with linear attention (distilled) and LoRA fine-tuning support."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        linear_attention_checkpoint: str | None = None,
        lora_adapter_path: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        top_p: float = 0.8,
        top_k: int = 20,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.linear_attention_checkpoint = linear_attention_checkpoint
        self.lora_adapter_path = lora_adapter_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)

        # Generation kwargs
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='flash_attention_2',
        )

        # Apply linear attention if checkpoint provided
        if linear_attention_checkpoint:
            self._apply_linear_attention(linear_attention_checkpoint)

        # Load LoRA adapter if provided
        if lora_adapter_path:
            self._load_lora_adapter(lora_adapter_path)

        self.model.eval()
        torch.cuda.empty_cache()

    def _apply_linear_attention(self, checkpoint_path: str):
        """Replace attention layers and load distilled weights."""
        from src.models.qwen3vl_linear import replace_attention_layers

        # Read config from checkpoint
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                ckpt_config = json.load(f)
            layer_indices = ckpt_config.get("linear_attention_layers")
            attention_type = ckpt_config.get("linear_attention_type", "gla")
        else:
            raise ValueError(f"config.json not found in {checkpoint_path}")

        # Replace attention layers
        if self.verbose:
            print(f"Replacing layers {layer_indices} with {attention_type} attention")
        replace_attention_layers(
            self.model,
            layer_indices=layer_indices,
            attention_type=attention_type
        )

        # Load weights
        weights_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            weights = load_file(weights_path)
        else:
            weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            weights = torch.load(weights_path, map_location="cpu")

        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if self.verbose and (missing or unexpected):
            print(f"Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")

    def _load_lora_adapter(self, adapter_path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False
        )
        if self.verbose:
            print(f"Loaded LoRA adapter from {adapter_path}")

    def generate_inner(self, message, dataset=None) -> str:
        """Generate response for VLMEvalKit format messages."""
        # Build content list for Qwen3-VL
        content = []
        for item in message:
            if item['type'] == 'image':
                content.append({
                    'type': 'image',
                    'image': ensure_file_url(item['value'])
                })
            elif item['type'] == 'video':
                content.append({
                    'type': 'video',
                    'video': ensure_file_url(item['value']),
                    'fps': self.fps,
                    'max_pixels': self.max_pixels,
                })
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})

        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': content})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision inputs
        from qwen_vl_utils import process_vision_info
        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )

        # Prepare inputs
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            return_tensors='pt',
            **(video_kwargs or {}),
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.generate_kwargs)

        # Decode (skip input tokens)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return response.strip()
