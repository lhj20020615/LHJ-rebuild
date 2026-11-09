from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class LateFusionClassifier(nn.Module):
    """Text encoder plus four-dimensional ERP late fusion classifier."""

    def __init__(self, model_dir: Path, use_lora: bool = True) -> None:
        super().__init__()
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            if use_lora:
                raise RuntimeError("Install peft or run with --no-lora") from exc
            LoraConfig = get_peft_model = None

        config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        backbone = AutoModel.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            backbone = get_peft_model(backbone, lora_config)
        self.backbone = backbone
        hidden = getattr(config, "hidden_size", getattr(config, "d_model", None))
        if hidden is None:
            raise ValueError("Could not determine the text encoder hidden size")
        self.text_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.erp_projection = nn.Linear(4, hidden)
        self.classifier = nn.Linear(hidden * 2, 3)

    def forward(self, input_ids=None, attention_mask=None, erp_feats=None):
        if erp_feats is None:
            raise ValueError("erp_feats is required for ERP late fusion")
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(outputs.last_hidden_state[:, 0, :])
        erp_features = self.erp_projection(erp_feats.to(self.erp_projection.weight.dtype))
        return self.classifier(torch.cat([text_features, erp_features], dim=1))

