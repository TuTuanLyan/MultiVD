import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEVDRouter(nn.Module):
    def __init__(self, pretrained_model, num_experts: int):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class MulVulExpert(nn.Module):
    def __init__(self,
                 pretrained_model,
                 num_labels: int = 1,
                 num_langs: int = 2,
                 pool_length: int = 5,
                 lang_map: dict = None,
                 temperature: float = 0.1):
        super().__init__()

        self.backbone = pretrained_model
        self.hidden_size = self.backbone.config.hidden_size
        self.pool_length = pool_length
        self.lang_map = lang_map or {"python": 0, "c": 1}
        self.temperature = temperature

        self.parameter_pool = nn.Parameter(
            torch.randn(num_langs, pool_length, self.hidden_size) * 0.02
        )
        self.keys = nn.Parameter(
            torch.randn(num_langs, self.hidden_size) * 0.02
        )
        
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, language=None):
        batch_size = input_ids.size(0)

        raw_embeds = self.backbone.embeddings(input_ids)
        cls_query = raw_embeds[:, 0, :]

        if language is not None:
            pool = self.parameter_pool[language]
            
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            selected_keys = keys_norm[language]
            cosine_sim = torch.sum(query_norm * selected_keys, dim=1)
            aux_loss = (1.0 - cosine_sim).mean()
        else:
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            scores = torch.matmul(query_norm, keys_norm.transpose(0, 1)) / self.temperature
            lang_ids = torch.argmax(scores, dim=1)
            pool = self.parameter_pool[lang_ids]
            aux_loss = torch.tensor(0.0, device=cls_query.device)

        # Truncate + concat pool tokens
        keep_len = raw_embeds.size(1) - self.pool_length
        raw_embeds = raw_embeds[:, :keep_len, :]
        attention_mask_truncated = attention_mask[:, :keep_len]

        new_embeds = torch.cat([pool, raw_embeds], dim=1)

        pool_mask = torch.ones(batch_size, self.pool_length,
                               dtype=attention_mask.dtype,
                               device=attention_mask.device)
        new_mask = torch.cat([pool_mask, attention_mask_truncated], dim=1)

        outputs = self.backbone(inputs_embeds=new_embeds, attention_mask=new_mask)
        hidden_states = outputs.last_hidden_state

        # Mean pooling trên pool tokens
        pool_hidden = hidden_states[:, 0:self.pool_length, :]
        final_repr = pool_hidden.mean(dim=1)

        logits = self.classifier(final_repr)
        
        return logits, aux_loss