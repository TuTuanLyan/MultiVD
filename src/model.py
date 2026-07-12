import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferModel(nn.Module):
    def __init__(self, backbone, num_classes, num_cwes, dropout_rate = 0.1):
        super(TransferModel, self).__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.vul_head = nn.Linear(hidden_size, num_classes)
        self.cwe_head = nn.Linear(hidden_size, num_cwes)

    def forward(self, input_ids, attention_mask, return_cwe = False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        vul_logits = self.vul_head(cls_output)
        
        if return_cwe:
            cwe_logits = self.cwe_head(cls_output)
        else:
            cwe_logits = None

        return {
            "vul_logits": vul_logits,
            "cwe_logits": cwe_logits
        }


class BaselineModel(nn.Module):
    """Plain CodeBERT binary classifier with no source task or CWE head."""

    def __init__(self, backbone, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.vul_head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_cwe=False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        return {
            "vul_logits": self.vul_head(cls_output),
            "cwe_logits": None,
        }
