# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
from linevul_model import Model, compute_adjustment
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
import hashlib
import json
import sys
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline"))
from evaluate import classification_metrics, find_best_threshold, per_cwe_metrics
from train_transfer import CLASS_TO_CWE
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self,
                 func,
                 input_tokens,
                 input_ids,
                 lan,
                 label,
                 commits_url):
        self.func = func
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.lan = lan
        self.label=label
        self.commits_url = commits_url


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, df, phase):
        self.phase = phase
        self.examples = []

        df.dropna(subset=['target'], inplace=True)
        df.dropna(subset=['processed_func'], inplace=True)
        if 'codeLink' in df.columns:
            df['commit_url'] = df['codeLink']

        df = df[df['language'].isin(args.language)]

        df.reset_index(inplace=True)

        funcs = df["processed_func"].tolist()

        languages = df["language"].tolist()
        language_dict = {}
        for i, language in enumerate(args.language):
            language_dict[language] = i+1

        print("language_dict: ", language_dict)

        df["target"] = df["target"].astype(int)
        labels = df["target"].tolist()
        commits_url = df["commit_url"].tolist()

        self.df = df

        for i in range(len(labels)):
            if labels[i] > 0:
                labels[i] = language_dict[languages[i]]
            else:
                labels[i] = 0

        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], commits_url[i], languages[i], tokenizer, args))
        if phase == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.phase != "test":
            return self.examples[i].lan, torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)
        else:
            return self.examples[i].lan, torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), self.examples[i].func, self.examples[i].commits_url


def convert_examples_to_features(func, label, commits_url, lan, tokenizer, args):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(func, source_tokens, source_ids, lan, label, commits_url)
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(func, source_tokens, source_ids, lan, label, commits_url)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore = np.nan_to_num(fscore, nan=0)
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    fscoreOpt = round(fscore[index], ndigits = 4)
    recallOpt = round(recall[index], ndigits = 4)
    precisionOpt = round(precision[index], ndigits = 4)

    return thresholdOpt, fscoreOpt, recallOpt, precisionOpt


def train(args, train_dataset, model, tokenizer, eval_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)
    if args.logits_adjust:
        args.logit_adjustments = compute_adjustment(train_dataloader, 1.0, args)
        model.set_adjust_logits(args.logit_adjustments, args.post_adjust)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 20
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.freeze_encoder:
        for name, module in model.named_modules():
            if ".roberta" in name:
                for param in module.parameters():
                    param.requires_grad = False

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()
    model.train()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            lan, inputs_ids, labels = batch
            inputs_ids, labels = inputs_ids.to(args.device), labels.to(args.device)

            loss, logits = model(input_ids=inputs_ids, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)
                    model.train()

                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:
        lan, inputs_ids, labels = batch
        inputs_ids, labels = inputs_ids.to(args.device), labels.to(args.device)

        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = np.where(np.max(logits,1)>best_threshold, np.argmax(logits,1), 0)

    print(logits.shape, y_trues.shape)
    recall = recall_score(y_trues, y_preds, average=None)
    precision = precision_score(y_trues, y_preds, average=None)
    f1 = f1_score(y_trues, y_preds, average='micro')
    if logits.shape[1] > 2:
        ap = average_precision_score(y_trues, logits, average=None)
    else:
        ap = np.array([average_precision_score((y_trues == c).astype(int), logits[:, c]) for c in range(logits.shape[1])])
    cm = confusion_matrix(y_trues, y_preds)
    result = {
        "eval_ap": ap,
        "eval_recall": recall,
        "eval_precision": precision,
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
        "eval_loss": float(eval_loss) / float(nb_eval_steps),
        "eval_cm":cm
    }

    result["eval_f1_original"] = result["eval_f1"]
    result["eval_pr_auc_binary"] = float(average_precision_score((y_trues != 0).astype(int), 1 - logits[:, 0]))
    if args.selection_metric == "pr_auc":
        result["eval_f1"] = result["eval_pr_auc_binary"]
    log_epoch(args, result)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(np.round(result[key],4)))

    return result

def test(args, model, tokenizer, test_dataset, cls_threshold=0.5):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    funcs = []
    langs = []
    commits_urls = []
    for batch in tqdm(test_dataloader):
        lan, inputs_ids, labels, func, commits_url = batch
        inputs_ids, labels = inputs_ids.to(args.device), labels.to(args.device)
        with torch.no_grad():
            try:
                lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            except:
                print(inputs_ids.dtype, labels.dtype)
                exit()

            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            funcs.append(func)
            commits_urls.append(commits_url)
            langs.append(lan)
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    langs = np.concatenate(langs, 0)
    y_preds = np.where(np.max(logits,1)>cls_threshold, np.argmax(logits,1), 0)

    lan_results = {}
    for i, lan in enumerate(args.language):
        lan_index = np.where(langs==lan)[0]
        if len(lan_index) == 0:
            continue
        y_trues_lan = y_trues[lan_index] != 0
        logits_lan = 1 - logits[lan_index][:, 0]
        y_preds_lan = logits_lan > cls_threshold

        best_threshold, best_f1, best_recall, best_precision = get_best_f1(y_trues_lan, logits_lan)
        print(f"best_threshold for {lan}: ", best_threshold)
        print(f"best_f1 for {lan}: ", best_f1)
        print("best_recall: ", best_recall)
        print("best_precision: ", best_precision)

        acc = accuracy_score(y_trues_lan, y_preds_lan)
        recall = recall_score(y_trues_lan, y_preds_lan, average=None)
        precision = precision_score(y_trues_lan, y_preds_lan, average=None)
        f1 = f1_score(y_trues_lan, y_preds_lan, average=None)
        ap = average_precision_score(y_trues_lan, logits_lan, average=None)
        cm = confusion_matrix(y_trues_lan, y_preds_lan)
        mcc = matthews_corrcoef(y_trues_lan, y_preds_lan)
        result = {
            f"{lan}_accuracy": float(acc),
            f"{lan}_ap": ap,
            f"{lan}_recall": recall,
            f"{lan}_precision": precision,
            f"{lan}_f1": f1,
            f"{lan}_threshold":best_threshold,
            f"{lan}_mcc": mcc,
            f"{lan}_cm": cm
        }
        print (f"***** {lan} Test results *****")
        print(result)
        lan_results[lan] = result

    if logits.shape[1] == 2 or len(np.unique(y_trues)) < logits.shape[1]:
        return logits, y_trues, langs, lan_results
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds, average=None)
    precision = precision_score(y_trues, y_preds, average=None)
    f1 = f1_score(y_trues, y_preds, average=None)
    ap = average_precision_score(y_trues, logits, average=None)
    cm = confusion_matrix(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    result = {
        "test_accuracy": float(acc),
        "test_ap": ap,
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
        "test_threshold":best_threshold,
        "test_mcc": mcc,
        "test_cm": cm
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(np.round(result[key],4)))

    return logits, y_trues, langs, lan_results

FUNC_VARIANT = "mc"
UPSTREAM_MD5 = {'linevul_main_mc.py': '1f9e993956f3628d370e36808157531b', 'linevul_model.py': '50129ed7d3eb6cb9af0ff13f12051d83'}
SELECTION_METRICS = ("micro_f1", "pr_auc")
LANG_NAME = {"js": "JavaScript", "python": "Python"}
EPOCH_LOG = {"history": [], "train_loss": [], "t": None}


def md5_file(path, chunk=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def limit_records(records, limit):
    if limit <= 0:
        return records
    want, seen, kept = {0: (limit + 1) // 2, 1: limit // 2}, {0: 0, 1: 0}, []
    for record in records:
        label = int(record["label"])
        if seen[label] < want[label]:
            kept.append(record)
            seen[label] += 1
    return kept


def to_frame(records):
    return pd.DataFrame({
        "processed_func": [r["code"] for r in records],
        "target": [int(r["label"]) for r in records],
        "language": [LANG_NAME[r["lang"]] for r in records],
        "commit_url": [r.get("commit_url") or "" for r in records],
    })


def load_fold_records(args):
    js = {s: limit_records(load_jsonl(os.path.join(args.source_root, f"{s}.jsonl")), args.limit_examples) for s in ("train", "val", "test")}
    py = {s: [] for s in ("train", "val", "test")}
    if args.data_root:
        fold_dir = os.path.join(args.data_root, f"fold{args.fold}")
        py = {s: limit_records(load_jsonl(os.path.join(fold_dir, f"{s}.jsonl")), args.limit_examples) for s in ("train", "val", "test")}
    elif "Python" in args.language:
        sys.exit("--language có Python thì cần --data_root (fold SVEN)")
    target = py if "Python" in args.language else js
    logger.info("[func] dữ liệu | fold %d | train JS %d + Python %d | val JS %d + Python %d | chọn ngưỡng %d | test %d (%s)", args.fold,
                len(js["train"]), len(py["train"]), len(js["val"]), len(py["val"]), len(target["val"]), len(target["test"]),
                "SVEN fold k" if target is py else "JS pool")
    return {"train": js["train"] + py["train"], "val": js["val"] + py["val"], "thr": target["val"], "test": target["test"],
            "python_target": target is py}


def fpe_setup(args):
    parsed = dict(vars(args))
    if args.phase == "test" and not args.result_path:
        sys.exit("--phase test cần --result_path")
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linevul_model.py")
    if md5_file(model_file) != UPSTREAM_MD5["linevul_model.py"]:
        sys.exit("func/linevul_model.py phải trùng md5 bản gốc (chép nguyên văn)")
    args.model_name_or_path = args.tokenizer_name = args.model_name
    args.output_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
    args.model_name = os.path.basename(args.checkpoint_path)
    args.do_train, args.do_test = args.phase == "train", args.phase == "test"
    if args.init_ckpt and not args.zeroshot:
        sys.exit("--init_ckpt chỉ dùng với --zeroshot 1")
    return parsed


def write_meta(ckpt, meta):
    with open(str(ckpt) + ".mvd.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)


def read_meta(ckpt):
    path = str(ckpt) + ".mvd.json"
    if not os.path.exists(path):
        sys.exit(f"thiếu {path}: checkpoint không do bản func ghi, hoặc bị chép thiếu file .mvd.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def training_args(args, parsed):
    out = dict(parsed)
    here = os.path.dirname(os.path.abspath(__file__))
    out.update({"mvd_variant": FUNC_VARIANT, "mvd_role": args.mvd_role, "mvd_language": args.language,
                "mvd_upstream_md5": UPSTREAM_MD5, "mvd_func_md5": {f: md5_file(os.path.join(here, f)) for f in sorted(UPSTREAM_MD5)},
                "mvd_selection": SELECTION_TEXT[args.selection_metric]})
    return out


def best_epoch(history):
    best, best_value = None, 0
    for row in history:
        if row["val_selection"] > best_value:
            best, best_value = row["epoch"], row["val_selection"]
    return best


def hook_train_loss(model):
    def _hook(module, _inputs, output):
        if module.training:
            EPOCH_LOG["train_loss"].append(float(output.detach().float().mean().item()))
    model.loss_fct.register_forward_hook(_hook)
    EPOCH_LOG["t"] = time.perf_counter()


def log_epoch(args, result):
    history = EPOCH_LOG["history"]
    losses, EPOCH_LOG["train_loss"] = EPOCH_LOG["train_loss"], []
    history.append({"epoch": len(history) + 1, "train_loss": float(np.mean(losses)) if losses else None, "val_loss": result["eval_loss"],
                    "val_f1_original": result["eval_f1_original"], "val_pr_auc": result["eval_pr_auc_binary"],
                    "val_selection": result["eval_f1"]})
    now = time.perf_counter()
    seconds, EPOCH_LOG["t"] = now - (EPOCH_LOG["t"] or now), now
    row = history[-1]
    logger.info("MVD epoch %d/%d | train loss %s | val loss %.4f | val micro_f1 %.4f | val pr_auc %.4f | chọn theo %s | best ep %s | %ds",
                row["epoch"], args.epochs, "%.4f" % row["train_loss"] if row["train_loss"] is not None else "-", row["val_loss"],
                row["val_f1_original"], row["val_pr_auc"], args.selection_metric, best_epoch(history), int(seconds))


SELECTION_TEXT = {
    "micro_f1": "val micro-F1 (đa lớp, argmax nếu max prob > 0,5, ngược lại lớp 0) (mã gốc: eval_f1 > best_f1 mỗi epoch)",
    "pr_auc": "val PR-AUC nhị phân trên toàn bộ val (nhãn label != 0, điểm 1 - P(lớp 0)) - bài MVD mục IV-D",
}


def clear_stale_checkpoint(args):
    stale = os.path.join(args.output_dir, "checkpoint-best-f1", args.model_name)
    if os.path.exists(stale):
        logger.warning("[func] xoá checkpoint sót của lượt trước: %s", stale)
        os.remove(stale)


def finish_train(args, parsed, seconds, num_class, extra=None):
    saved = os.path.join(args.output_dir, "checkpoint-best-f1", args.model_name)
    if not os.path.exists(saved):
        sys.exit(f"train() gốc không lưu checkpoint nào ({saved}): không epoch nào có giá trị chọn > 0")
    os.replace(saved, args.checkpoint_path)
    os.rmdir(os.path.dirname(saved))
    history = EPOCH_LOG["history"]
    best = best_epoch(history)
    adjust = getattr(args, "logit_adjustments", None)
    meta = {"role": args.mvd_role, "best_epoch": best, "best_val_f1": history[best - 1]["val_f1_original"],
            "best_val_pr_auc": history[best - 1]["val_pr_auc"], "selection_metric": args.selection_metric, "history": history,
            "train_seconds": seconds, "max_steps": args.max_steps, "warmup_steps": args.warmup_steps,
            "logit_adjustments": None if adjust is None else [float(x) for x in adjust.cpu().numpy()],
            "language": args.language, "num_class": int(num_class), "training_args": training_args(args, parsed)}
    meta.update(extra or {})
    write_meta(args.checkpoint_path, meta)
    logger.info("[func] train xong | %s | best epoch %s (chọn theo %s) | %ds | checkpoint %s", args.mvd_role, best, args.selection_metric,
                int(seconds), args.checkpoint_path)


def test_checkpoint(args):
    meta = read_meta(args.checkpoint_path)
    if meta.get("role") != args.mvd_role:
        sys.exit(f"{args.checkpoint_path}: checkpoint của vai {meta.get('role')!r}, lệnh này là vai {args.mvd_role!r}")
    if meta.get("role") == "zeroshot":
        teacher = meta["teacher_checkpoint"]
        if md5_file(teacher) != meta["teacher_md5"]:
            sys.exit(f"checkpoint teacher {teacher} đã đổi từ lúc ghi con trỏ zeroshot (md5 lệch)")
        return teacher, meta
    return args.checkpoint_path, meta


def attach_adjustments(model, args, meta):
    if meta["logit_adjustments"] is not None:
        model.set_adjust_logits(torch.tensor(meta["logit_adjustments"], dtype=torch.float64, device=args.device), args.post_adjust)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def mvd_score(out):
    probs, y_trues = out[0], out[1]
    return (1.0 - probs[:, 0].astype(np.float32)).tolist(), (y_trues != 0).astype(int).tolist()


def write_result(args, meta, ckpt, thr_out, thr_records, test_out, test_records, thr_seconds, test_seconds):
    thr_p, thr_y = mvd_score(thr_out)
    test_p, test_y = mvd_score(test_out)
    assert thr_y == [int(r["label"]) for r in thr_records], "thứ tự / nhãn tập chọn ngưỡng lệch jsonl"
    assert test_y == [int(r["label"]) for r in test_records], "thứ tự / nhãn test lệch test.jsonl"
    threshold, calibrated_f1 = find_best_threshold(thr_y, thr_p)
    cwe = [r.get("cwe_class") for r in test_records]
    python_target = all(r["lang"] == "python" for r in test_records)
    at05, atval = classification_metrics(test_y, test_p, 0.5), classification_metrics(test_y, test_p, threshold)
    logger.info("Test | ROC %.4f | PR %.4f | macro-F1@0.5 %.4f | macro-F1@val %.4f (ngưỡng %.2f)", at05["roc_auc"], at05["pr_auc"],
                at05["macro_f1"], atval["macro_f1"], threshold)
    result = {
        "experiment_name": f"{args.run_name}/mvd_func_{meta['role']}", "phase": "test", "fold": args.fold, "seed": args.seed,
        "source_checkpoint": meta.get("teacher_checkpoint"), "target_checkpoint": str(ckpt), "best_epoch": meta["best_epoch"],
        "best_val_micro_f1_mvd": meta["best_val_f1"], "best_val_pr_auc_mvd": meta["best_val_pr_auc"],
        "mvd_selection_metric": meta["selection_metric"], "val_calibrated_threshold": threshold,
        "val_macro_f1_at_valcal": calibrated_f1, "validation_and_threshold_seconds": thr_seconds,
        "test_inference_seconds": test_seconds, "train_seconds": meta["train_seconds"],
        "test_macro_f1_at_0.5": at05["macro_f1"], "test_macro_f1_at_valcal": atval["macro_f1"],
        "test_positive_f1_at_0.5": at05["positive_f1"], "test_positive_f1_at_valcal": atval["positive_f1"],
        "test_precision_at_0.5": at05["precision"], "test_precision_at_valcal": atval["precision"],
        "test_recall_at_0.5": at05["recall"], "test_recall_at_valcal": atval["recall"],
        "test_accuracy_at_0.5": at05["accuracy"], "test_accuracy_at_valcal": atval["accuracy"],
        "test_roc_auc": at05["roc_auc"], "test_pr_auc": at05["pr_auc"],
        "per_cwe": per_cwe_metrics(test_y, test_p, cwe, threshold, CLASS_TO_CWE) if python_target else None,
        "per_cwe_at_0.5": per_cwe_metrics(test_y, test_p, cwe, 0.5, CLASS_TO_CWE) if python_target else None,
        "per_cwe_at_valcal": per_cwe_metrics(test_y, test_p, cwe, threshold, CLASS_TO_CWE) if python_target else None,
        "mvd_history": meta["history"], "mvd_logit_adjustments": meta["logit_adjustments"],
        "mvd_logit_adjustments_attached_at_test": meta["logit_adjustments"] is not None, "mvd_role": meta["role"],
        "mvd_func_variant": FUNC_VARIANT, "mvd_original_test_metrics": jsonable(test_out[3]),
        "hyperparameters": meta["training_args"],
    }
    out = args.result_path
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(os.path.splitext(out)[0] + ".probs.npz", probabilities=np.asarray(test_p, dtype=np.float32),
                        labels=np.asarray(test_y, dtype=np.int8), cwe_classes=np.asarray(cwe),
                        val_probabilities=np.asarray(thr_p, dtype=np.float32), val_labels=np.asarray(thr_y, dtype=np.int8),
                        threshold=np.float32(threshold))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("MVD result saved: %s", out)


def mvd_role(args):
    if args.zeroshot:
        return "zeroshot"
    return "joint" if "Python" in args.language else "teacher"


def zeroshot_prepare(args, parsed):
    if not args.init_ckpt:
        sys.exit("--zeroshot 1 --phase train cần --init_ckpt <checkpoint teacher>")
    teacher = os.path.abspath(args.init_ckpt)
    teacher_meta = read_meta(teacher)
    if teacher_meta.get("role") != "teacher" or teacher_meta["num_class"] != len(args.language) + 1:
        sys.exit(f"{teacher}: role {teacher_meta.get('role')!r}, num_class {teacher_meta['num_class']} - cần teacher {len(args.language) + 1} lớp")
    meta = dict(teacher_meta)
    meta.update({"role": "zeroshot", "teacher_checkpoint": teacher, "teacher_md5": md5_file(teacher), "train_seconds": 0.0,
                 "teacher_train_seconds": teacher_meta["train_seconds"], "teacher_language": teacher_meta["language"],
                 "language": args.language, "training_args": dict(training_args(args, parsed), teacher_training_args=teacher_meta["training_args"])})
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)
    write_meta(args.checkpoint_path, meta)
    logger.info("[func] zeroshot: không train | con trỏ %s.mvd.json -> teacher %s (best ep %s)", args.checkpoint_path, teacher,
                teacher_meta["best_epoch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--debug", action='store_true',
                        help="Whether to debug.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--language', type=str, default=None,
                        help="language to select")
    parser.add_argument('--resume', type=str, default=None,
                        help="Whether training from specified model.")
    parser.add_argument("--freeze_encoder", action='store_true',
                        help="Whether freeze the encoder.")
    parser.add_argument("--logits-adjust", "--logits_adjust", nargs='?', const=1, default=0, type=int,
                        help="Whether adjust the logits. (Long-Tail Learning via Logit Adjustment)")
    parser.add_argument("--post-adjust", action='store_true',
                        help="Whether post adjust the logits. (Long-Tail Learning via Logit Adjustment)")
    parser.add_argument("--focal-loss", "--focal_loss", nargs='?', const=1, default=0, type=int,
                        help="Whether to use focal loss.")

    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--fold", type=int, required=True, help="fold SVEN Python")
    parser.add_argument("--model_name", required=True, help="thư mục backbone (tokenizer + encoder) = model_name_or_path / tokenizer_name gốc")
    parser.add_argument("--checkpoint_path", required=True, help="checkpoint của ô này (train ghi, test đọc)")
    parser.add_argument("--result_path", help="result JSON (pha test); .probs.npz cạnh nó")
    parser.add_argument("--init_ckpt", help="zeroshot: checkpoint teacher (pha train)")
    parser.add_argument("--data_root", help="fold SVEN Python (data_root/fold<k>/{train,val,test}.jsonl)")
    parser.add_argument("--source_root", required=True, help="pool JS ({train,val,test}.jsonl; test = bản sao val)")
    parser.add_argument("--selection_metric", choices=SELECTION_METRICS, default="micro_f1",
                        help="micro_f1 = như mã gốc (eval_f1); pr_auc = như bài (mục IV-D): PR-AUC nhị phân trên toàn bộ val")
    parser.add_argument("--limit_examples", type=int, default=0, help="CHỈ thử khói: N bản ghi đầu mỗi tập, chia đều hai nhãn; 0 = tắt")
    parser.add_argument("--zeroshot", type=int, default=0, help="1 = chấm checkpoint teacher (--init_ckpt) thẳng trên SVEN, không train")
    args = parser.parse_args()
    parsed = fpe_setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    set_seed(args)
    if args.language is None:
        exit("Please specify language!")
    else:
        if isinstance(args.language, str) and ',' in args.language:
            args.language = args.language.split(',')
        else:
            args.language = [args.language]
    args.mvd_role = mvd_role(args)
    if args.zeroshot and args.do_train:
        zeroshot_prepare(args, parsed)
        return {}

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_class = len(args.language) + 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('./word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))

    logger.info("Training/evaluation parameters %s", args)

    records = load_fold_records(args)
    train_df, val_df = to_frame(records["train"]), to_frame(records["val"])

    if args.do_train:
        clear_stale_checkpoint(args)
        hook_train_loss(model)
        train_dataset = TextDataset(tokenizer, args, train_df, 'train')
        eval_dataset = TextDataset(tokenizer, args, val_df, 'eval')
        started = time.perf_counter()
        train(args, train_dataset, model, tokenizer, eval_dataset)
        finish_train(args, parsed, time.perf_counter() - started, model.classifier.out_proj.out_features)
    results = {}
    if args.do_test:
        ckpt, meta = test_checkpoint(args)
        model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=True)
        attach_adjustments(model, args, meta)
        model.to(args.device)
        thr_dataset = TextDataset(tokenizer, args, to_frame(records["thr"]), 'test')
        test_dataset = TextDataset(tokenizer, args, to_frame(records["test"]), 'test')
        started = time.perf_counter()
        thr_out = test(args, model, tokenizer, thr_dataset, cls_threshold=0.5)
        thr_seconds = time.perf_counter() - started
        started = time.perf_counter()
        test_out = test(args, model, tokenizer, test_dataset, cls_threshold=0.5)
        write_result(args, meta, ckpt, thr_out, records["thr"], test_out, records["test"], thr_seconds,
                     time.perf_counter() - started)
    return results

if __name__ == "__main__":
    main()
