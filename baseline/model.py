import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


AUX_MODES = ("cwe", "latent_bottleneck", "latent_proto", "none")
POOLING_MODES = ("cls", "mean")


def build_backbone(model_name):
    if "codet5p" in model_name and model_name.endswith("embedding"):
        return AutoModel.from_pretrained(model_name, trust_remote_code=True).encoder

    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


def freeze_backbone_layers(backbone, n_layers):
    if n_layers <= 0:
        return {"frozen_modules": 0}
    frozen = 0
    embeddings = getattr(backbone, "embeddings", None) or getattr(backbone, "shared", None)
    if embeddings is not None:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False
        frozen += 1

    encoder = getattr(backbone, "encoder", backbone)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        layers = getattr(encoder, "block", None)
    if layers is not None:
        for layer in list(layers)[:n_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
            frozen += 1
    return {"frozen_modules": frozen, "requested_layers": n_layers}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_((self.lora_b @ self.lora_a) * self.scaling)
        for parameter in self.base.parameters():
            parameter.requires_grad = True
        return self.base


LORA_TARGETS = ("query", "value", "q", "v")


def inject_lora(backbone, rank, alpha=16):
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    injected = 0
    for module in backbone.modules():
        for name, child in list(module.named_children()):
            if name in LORA_TARGETS and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
                injected += 1
    return injected


def merge_lora(backbone):
    merged = 0
    for module in backbone.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
                merged += 1
    return merged


def pool_hidden_states(hidden_states, attention_mask, strategy):
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy != "mean":
        raise ValueError(f"unsupported pooling: {strategy!r}, expected one of {POOLING_MODES}")
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


class TransferModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=2,
        num_cwes=4,
        dropout_rate=0.1,
        aux_mode="cwe",
        num_latent=8,
        latent_temperature=0.1,
        pooling="cls",
    ):
        super().__init__()
        if aux_mode not in AUX_MODES:
            raise ValueError(f"unsupported aux_mode: {aux_mode!r}, expected one of {AUX_MODES}")
        if pooling not in POOLING_MODES:
            raise ValueError(f"unsupported pooling: {pooling!r}, expected one of {POOLING_MODES}")
        self.backbone = backbone
        self.pooling = pooling
        self.aux_mode = aux_mode
        self.num_latent = num_latent
        self.latent_temperature = latent_temperature
        hidden_size = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.vul_head = nn.Linear(hidden_size, num_classes)

        if aux_mode == "cwe":
            self.cwe_head = nn.Linear(hidden_size, num_cwes)
        elif aux_mode == "latent_bottleneck":
            self.latent_proj = nn.Linear(hidden_size, num_latent)
            self.cwe_head = nn.Linear(num_latent, num_cwes)
        elif aux_mode == "latent_proto":
            self.latent_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_latent),
            )
            self.prototypes = nn.Parameter(torch.randn(num_latent, num_latent) * 0.02)
            self.freeze_prototypes_steps = 0
            self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def aux_modules(self):
        modules = []
        for name in ("latent_proj", "cwe_head"):
            if hasattr(self, name):
                modules.append(getattr(self, name))
        return modules

    def aux_parameters(self):
        parameters = [p for module in self.aux_modules() for p in module.parameters()]
        if hasattr(self, "prototypes"):
            parameters.append(self.prototypes)
        return parameters

    def forward(self, input_ids, attention_mask, return_cwe=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(outputs.last_hidden_state, attention_mask, self.pooling)
        cls_output = self.dropout(pooled)
        vul_logits = self.vul_head(cls_output)

        cwe_logits = None
        latent = None
        if return_cwe and self.aux_mode != "none":
            if self.aux_mode == "cwe":
                cwe_logits = self.cwe_head(cls_output)
            elif self.aux_mode == "latent_bottleneck":
                latent = self.latent_proj(cls_output)
                cwe_logits = self.cwe_head(latent)
            elif self.aux_mode == "latent_proto":
                latent = F.normalize(self.latent_proj(cls_output), dim=-1)
                prototypes = F.normalize(self.prototypes, dim=-1)
                if self.training:
                    self._step += 1
                    if self._step.item() <= self.freeze_prototypes_steps:
                        prototypes = prototypes.detach()
                cwe_logits = latent @ prototypes.t()

        return {
            "vul_logits": vul_logits,
            "cwe_logits": cwe_logits,
            "latent": latent,
        }


class BaselineModel(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout_rate=0.1, pooling="cls"):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"unsupported pooling: {pooling!r}, expected one of {POOLING_MODES}")
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.vul_head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_cwe=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(outputs.last_hidden_state, attention_mask, self.pooling)
        cls_output = self.dropout(pooled)
        return {
            "vul_logits": self.vul_head(cls_output),
            "cwe_logits": None,
            "latent": None,
        }


@torch.no_grad()
def sinkhorn(scores, epsilon=0.05, n_iters=3):
    scores = scores.float()
    Q = torch.exp((scores - scores.max()) / epsilon).t()
    Q = Q / Q.sum().clamp_min(1e-12)
    n_prototypes, n_samples = Q.shape
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q / n_prototypes
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q = Q / n_samples
    return (Q * n_samples).t()


def auxiliary_loss(outputs, cwe_targets, aux_mode, sinkhorn_epsilon=0.05, temperature=0.1):
    logits = outputs["cwe_logits"]
    if aux_mode == "none" or logits is None:
        return None, None

    if aux_mode in ("cwe", "latent_bottleneck"):
        valid = cwe_targets != -100
        if not valid.any():
            return logits.sum() * 0.0, logits.argmax(dim=-1)
        loss = F.cross_entropy(logits[valid], cwe_targets[valid])
        return loss, logits.argmax(dim=-1)

    with torch.no_grad():
        targets = sinkhorn(logits.detach(), epsilon=sinkhorn_epsilon)
    loss = -torch.mean(
        torch.sum(targets * F.log_softmax(logits / temperature, dim=-1), dim=-1)
    )
    return loss, logits.argmax(dim=-1)
