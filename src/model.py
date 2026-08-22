import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


AUX_MODES = ("cwe", "latent_bottleneck", "latent_proto", "none")
POOLING_MODES = ("cls", "mean")


def build_backbone(model_name):
    """Load an encoder. Encoder-decoder checkpoints contribute their encoder only.

    T5-family checkpoints (CodeT5, CodeT5+) would otherwise drag in decoder
    weights that this classifier never runs.
    """
    # Nhánh RIÊNG cho codet5p-*-embedding, khoá theo TÊN model để mọi backbone
    # khác đi đúng đường cũ, không đổi một byte nào. Checkpoint đó đính kèm
    # modeling file riêng nên AutoConfig thường sẽ treo ở prompt y/N; và
    # AutoModel của nó trả về một vector 256 chiều đã chuẩn hoá chứ không phải
    # chuỗi hidden states, nên pool_hidden_states không áp được.
    #
    # Lấy `.encoder` của nó: cho ra (batch, seq, 768) đúng như các backbone khác,
    # và có ĐÚNG 84,954,240 tham số ngoài embedding — bằng từng tham số với
    # encoder của codet5p-220m. Nhờ vậy kiến trúc và cách đọc giữ nguyên, chỉ
    # PRETRAIN là khác, tức tách được đúng biến cần tách.
    if "codet5p" in model_name and model_name.endswith("embedding"):
        return AutoModel.from_pretrained(model_name, trust_remote_code=True).encoder

    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


def freeze_backbone_layers(backbone, n_layers):
    """Freeze embeddings and the lowest n encoder layers.

    Source pretraining on a small corpus helps a weak encoder and measurably
    damages a strong one. Holding the lower layers fixed bounds how far Phase 1
    can move a backbone that was already good, without giving up the heads and
    upper layers that the auxiliary task needs.
    """
    if n_layers <= 0:
        return {"frozen_modules": 0}
    frozen = 0
    embeddings = getattr(backbone, "embeddings", None) or getattr(backbone, "shared", None)
    if embeddings is not None:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False
        frozen += 1

    # RoBERTa-family: backbone.encoder.layer. T5-family: backbone.encoder.block.
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
    """A frozen Linear plus a trainable rank-r update.

    Full fine-tuning during Phase 1 rewrites the backbone, which helps a weak
    encoder and damages a strong one. Constraining that stage to a low-rank
    update bounds how far the weights can travel while still letting the source
    task teach something.
    """

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
        """Fold the update into the frozen weight and return the plain Linear.

        Merging before the checkpoint is written means Phase 2 loads an ordinary
        backbone and needs to know nothing about LoRA.
        """
        self.base.weight.add_((self.lora_b @ self.lora_a) * self.scaling)
        for parameter in self.base.parameters():
            parameter.requires_grad = True
        return self.base


# Query and value projections, named differently by BERT-family and T5-family.
LORA_TARGETS = ("query", "value", "q", "v")


def inject_lora(backbone, rank, alpha=16):
    """Freeze the backbone and attach LoRA to its attention projections."""
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
    """Position 0, or the attention-masked mean over the window.

    Default by family: `cls` for CodeBERT, `mean` for CodeT5 and CodeT5+.

    Both families do emit <s> at position 0 -- they share RobertaTokenizerFast,
    and an earlier version of this docstring was right to correct the claim that
    T5 has no token there. But having the token is not the same as having a
    trained summary. RoBERTa pretrains <s> through a sentence-level objective, so
    CodeBERT arrives with that slot already meaning something. T5 pretrains with
    span corruption only, which gives position 0 no sequence-level role at all,
    and mean pooling over encoder states is the standard readout for T5 encoders
    used as classifiers.

    The measured case for `cls` on CodeT5+ was +0.0039 against -0.0184 on the
    twin folds at seed 36. That gap sits inside this project's own noise band
    (per-fold sd reaches 0.09 on 152 test rows, and the cross-machine effect
    alone is 0.028), so it was never strong enough to override the convention.
    Treating it as decisive is how pooling ended up moving with the backbone in
    every run, which is the confound the earlier note was written to flag.
    """
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy != "mean":
        raise ValueError(f"unsupported pooling: {strategy!r}, expected one of {POOLING_MODES}")
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


class TransferModel(nn.Module):
    """CodeBERT backbone with a binary vulnerability head and one auxiliary head.

    The auxiliary head shapes the backbone during source pretraining and is
    frozen and unused from Phase 2 onward, so its shape never constrains the
    target. Four modes:

    cwe                explicit C-way CWE classifier (the v1 method)
    latent_bottleneck  C-way classifier factorized through K latent units, so
                       only a K-sized layer depends on the source taxonomy
    latent_proto       K learnable prototypes trained by balanced self-labelling;
                       needs no CWE labels at all
    none               no auxiliary task, for the ablation that isolates whether
                       the auxiliary signal contributes anything
    """

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
            # Two-layer projection head, following SwAV: the clustering objective
            # acts on its own space rather than directly on the pooled feature the
            # vulnerability head reads, so it cannot sharpen that feature's geometry.
            self.latent_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_latent),
            )
            self.prototypes = nn.Parameter(torch.randn(num_latent, num_latent) * 0.02)
            # Prototypes stay fixed for the first steps so the projection can settle
            # before the assignment targets start moving (SwAV freeze_prototypes_niters).
            self.freeze_prototypes_steps = 0
            self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def aux_modules(self):
        """Parameters that serve only the auxiliary task, frozen from Phase 2 on."""
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
                # Raw cosine scores. The temperature is applied in the loss, not
                # here: Sinkhorn needs the unscaled scores or exp() overflows.
                cwe_logits = latent @ prototypes.t()

        return {
            "vul_logits": vul_logits,
            "cwe_logits": cwe_logits,
            "latent": latent,
        }


class BaselineModel(nn.Module):
    """Plain CodeBERT binary classifier with no source task or CWE head."""

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
    """Balanced soft assignment (Sinkhorn-Knopp) used as the self-labelling target.

    The equipartition constraint is what stops the prototype head from collapsing
    onto one slot: a confident *and* balanced assignment is the only way down.
    """
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
    """Auxiliary loss for the active mode, plus diagnostics for logging.

    Returns (loss, assignment) where assignment is the hard latent/CWE slot per
    sample, or None when the mode has no auxiliary task. The assignment feeds the
    NMI diagnostics that tell us whether the latent structure is CWE-like or has
    merely collapsed onto the binary label.
    """
    logits = outputs["cwe_logits"]
    if aux_mode == "none" or logits is None:
        return None, None

    if aux_mode in ("cwe", "latent_bottleneck"):
        valid = cwe_targets != -100
        if not valid.any():
            return logits.sum() * 0.0, None
        loss = F.cross_entropy(logits[valid], cwe_targets[valid])
        return loss, logits.argmax(dim=-1)

    # latent_proto: no labels are used, the target comes from the balanced
    # assignment of this batch's own scores.
    with torch.no_grad():
        targets = sinkhorn(logits.detach(), epsilon=sinkhorn_epsilon)
    loss = -torch.mean(
        torch.sum(targets * F.log_softmax(logits / temperature, dim=-1), dim=-1)
    )
    return loss, logits.argmax(dim=-1)
