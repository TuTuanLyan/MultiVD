import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
import numpy as np
import torch.nn.functional as F


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if 'num_class' not in config.__dict__: # for old version of transformers
            config.num_class = 2
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.adjustments = None
        self.post_adjustments = False
        # detect if args have focal_loss
        if hasattr(args, 'focal_loss') and args.focal_loss:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = CrossEntropyLoss()
    
    def set_adjust_logits(self, adjustments, post_adjustments=False):
        self.adjustments = adjustments
        self.post_adjustments = post_adjustments
    
    # def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
    #     if output_attentions:
    #         if input_ids is not None:
    #             outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
    #         else:
    #             outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
    #         attentions = outputs.attentions
    #         last_hidden_state = outputs.last_hidden_state
    #         logits = self.classifier(last_hidden_state)
    #         prob = torch.softmax(logits, dim=-1)
    #         if labels is not None:
    #             loss = self.loss_fct(logits, labels)
    #             return loss, prob, attentions
    #         else:
    #             return prob, attentions
    #     else:
    #         if input_ids is not None:
    #             outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
    #         else:
    #             outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
    #         logits = self.classifier(outputs)
    #         if self.adjustments is not None:
    #             if self.post_adjustments and self.training is False:
    #                 logits = logits - self.adjustments
    #             elif not self.post_adjustments:
    #                 logits = logits + self.adjustments
    #         prob = torch.softmax(logits, dim=-1)
    #         if labels is not None:
                
    #             loss = self.loss_fct(logits, labels)
    #             return loss, prob
    #         else:
    #             return prob
    
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None, require_logits=False):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss = self.loss_fct(logits, labels)
                if require_logits:
                    return loss, logits, attentions
                else:
                    return loss, prob, attentions
            else:
                if require_logits:
                    return logits, attentions
                else:
                    return prob, attentions
                
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            if self.adjustments is not None:
                if self.post_adjustments and self.training is False:
                    logits = logits - self.adjustments
                elif not self.post_adjustments:
                    logits = logits + self.adjustments
            prob = torch.softmax(logits, dim=-1)

            if labels is not None:
                loss = self.loss_fct(logits, labels)
                if require_logits:
                    return loss, logits
                else:
                    return loss, prob
            else:
                if require_logits:
                    return logits
                else:
                    return prob


class ModelML(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelML, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        assert isinstance(args.language, list)
        self.languages = args.language
        # print(self.languages)
        # print("ATTENTION!!!", len(self.languages))
        # self.classifier = torch.nn.ModuleList([RobertaClassificationHead(config) for _ in range(len(self.languages))])
        self.classifier = torch.nn.ModuleDict({l: RobertaClassificationHead(config) for l in self.languages})
        self.args = args
    
    
    def forward(self, lan, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        # assert lan in self.languages, f"language {lan} not in {self.languages}"

        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = []
            for i, l in enumerate(lan):
                # index = self.languages.index(l)
                logits.append(self.classifier[l](last_hidden_state[i].unsqueeze(0)))
            logits = torch.concat(logits, dim=0)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss(label_smoothing=0.1)    # add label smoothing
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]

            # logits = self.classifier[0](outputs)
            
            logits = []
            # print(outputs.shape)
            for i, l in enumerate(lan):
                # index = self.languages.index(l)
                logits.append(self.classifier[l](outputs[i].unsqueeze(0)))
            logits = torch.concat(logits, dim=0)
            # print(logits)
            
            prob = torch.softmax(logits, dim=-1)
            # print(prob,labels)
            # exit()
            if labels is not None:
                loss_fct = CrossEntropyLoss(label_smoothing=0.1)   # label_smoothing=0.1
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_logits, student_logits):
        """
        teacher_prob: [N, C]
        student_prob: [N, C]
        labels: [N, ]
        """
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (self.temperature**2)

        return soft_targets_loss


class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        # self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def compute_adjustment(train_loader, tro, args):
    """compute the base probabilities"""
    # lan, inputs_ids, labels
    label_freq = {}
    for i, batch in enumerate(train_loader):
        target = batch[-1]
        target = target.to(args.device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    print("label_freq", label_freq)
    return adjustments
