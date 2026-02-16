"""InternVL with linear attention (distilled) and LoRA support for VLMEvalKit."""

import json
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

from .internvl.internvl_chat import InternVLChat

# Add project root to path for importing our modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_path(path: str | None) -> str | None:
    """Resolve path relative to PROJECT_ROOT if it doesn't exist as-is."""
    if path is None:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # Try relative to PROJECT_ROOT
    project_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(project_path):
        return project_path
    # Return original (will fail later with clear error)
    return path


class InternVL3LinearAttention(InternVLChat):
    """InternVL3 with linear attention (distilled) and LoRA fine-tuning support."""

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3-2B-hf",
        linear_attention_checkpoint: str | None = None,
        lora_adapter_path: str | None = None,
        load_in_8bit: bool = False,
        version: str = "V2.0",
        verbose: bool = False,
        zero_heads: list | None = None,
        **kwargs,
    ):
        """Initialize InternVL3 with linear attention.
        
        Args:
            model_path: Base model path (HuggingFace ID or local path)
            linear_attention_checkpoint: Path to distilled checkpoint with linear attention
            lora_adapter_path: Optional path to LoRA adapter weights
            load_in_8bit: Whether to load in 8-bit
            version: InternVL version string (V2.0 for InternVL3)
            verbose: Print debug info
        """
        # Resolve relative paths
        self.linear_attention_checkpoint = resolve_path(linear_attention_checkpoint)
        self.lora_adapter_path = resolve_path(lora_adapter_path)
        self.verbose = verbose

        # Store kwargs before parent init
        self._linear_kwargs = kwargs.copy()
        
        # Manual initialization (not calling super().__init__)
        self.model_path = model_path
        self.version = version
        self.use_lmdeploy = False
        self.use_mpo_prompt = False
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = False
        self.cot_prompt_version = 'v1'
        self.system_prompt = None
        self.cot_prompt = None
        self.screen_parse = True
        self.best_of_n = 1
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        
        # Patterns for InternVL
        self.pattern = r'Image(\d+)'
        self.replacement = r'Image-\1'
        self.reverse_pattern = r'Image-(\d+)'
        self.reverse_replacement = r'Image\1'
        
        # Load model with trust_remote_code (InternVL models require this)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.device = 'cuda'

        # Apply linear attention if checkpoint provided
        if self.linear_attention_checkpoint:
            self._apply_linear_attention(self.linear_attention_checkpoint)

        # Load LoRA adapter if provided and path exists
        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            self._load_lora_adapter(self.lora_adapter_path)
        elif self.lora_adapter_path:
            print(f"Warning: LoRA adapter path does not exist: {self.lora_adapter_path}")

        # Apply head ablation (zero specific attention heads)
        if zero_heads:
            from src.models.head_ablation import zero_attention_heads
            zero_attention_heads(self.model, zero_heads)
            print(f"[Head Ablation] Zeroed {len(zero_heads)} head(s): {zero_heads}")

        self.model.eval()
        
        # Generation kwargs
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def _apply_linear_attention(self, checkpoint_path: str):
        """Replace attention layers and load distilled weights."""
        from src.models.internvl_linear import replace_internvl_attention_layers

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
        
        replace_internvl_attention_layers(
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
            if os.path.exists(weights_path):
                weights = torch.load(weights_path, map_location="cpu")
            else:
                raise ValueError(f"No weights found in {checkpoint_path}")

        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if self.verbose and (missing or unexpected):
            print(f"Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")

    def _load_lora_adapter(self, adapter_path: str):
        """Load LoRA adapter weights manually without PeftModel.from_pretrained.
        
        This avoids the prepare_inputs_for_generation requirement for custom models.
        """
        from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(adapter_path)
        
        # Get target module - for InternVL it should be the language_model
        if hasattr(self.model, 'language_model'):
            target_model = self.model.language_model
            if self.verbose:
                print(f"Applying LoRA to language_model: {type(target_model).__name__}")
        else:
            target_model = self.model
        
        # Apply PEFT to target model
        target_model = get_peft_model(target_model, peft_config)
        
        # Load adapter weights
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_weights_path):
            adapter_weights = load_file(adapter_weights_path)
        else:
            adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
            adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
        
        # Set the weights
        set_peft_model_state_dict(target_model, adapter_weights)
        
        # Update reference
        if hasattr(self.model, 'language_model'):
            self.model.language_model = target_model
        else:
            self.model = target_model
        
        if self.verbose:
            print(f"Loaded LoRA adapter from {adapter_path}")
