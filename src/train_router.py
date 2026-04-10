import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from ultis import load_data, create_data_loader, focal_loss, CodeDatasetForRouter
from mulvulmoe import MoEVDRouter
from logger import setup_logging, get_logger

import argparse

import os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def train_router(model, device, optimizer, criterion, logger, args_parser, train_loader, val_loader=None):
    """
    Train router (multi-class classification cho CWE groups, chỉ dùng vulnerable data).
    - Nếu có val_loader: đánh giá F1 macro, lưu best model theo F1
    - Nếu không có val: lưu best model theo train loss thấp nhất
    - Optional: xóa các checkpoint cũ, chỉ giữ best
    """
    model.to(device)
    best_f1 = -1.0 if val_loader else float('inf')  # Max F1 nếu có val, min loss nếu không
    best_epoch = 0
    os.makedirs(args_parser.output_dir, exist_ok=True)
    best_model_path = os.path.join(args_parser.output_dir, f"mulvulmoe_router_best.pth")
    patience = 5  # Early stopping nếu F1 không cải thiện sau 5 epoch
    patience_counter = 0

    logger.info(f"Starting training Router | Epochs: {args_parser.epochs}")

    for epoch in range(args_parser.epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['CWE_ID'].to(device).long()  # Multi-class labels (0-11)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [batch, num_groups]
            loss = criterion(logits, labels)  # Focal Loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        log_msg = f"Epoch {epoch+1}/{args_parser.epochs} | Train Loss: {avg_train_loss:.4f}"

        # ------------------- Validation (nếu có) -------------------
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['CWE_ID'].to(device).cpu().numpy()

                    logits = model(input_ids, attention_mask)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

            # Macro metrics để xử lý imbalance
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            log_msg += f" | Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"

            # Update best model dựa trên F1 macro
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved! F1: {f1:.4f} at epoch {best_epoch}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs)")
                    break
        else:
            # Không có val: lưu nếu train loss tốt hơn
            if avg_train_loss < best_f1:  # best_f1 dùng như best_loss
                best_f1 = avg_train_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved! Train Loss: {avg_train_loss:.4f} at epoch {best_epoch}")

        logger.info(log_msg)

    # Optional: Xóa các checkpoint cũ
    if hasattr(args_parser, 'clean_old_checkpoints') and args_parser.clean_old_checkpoints:
        for file in os.listdir(args_parser.output_dir):
            if file.startswith("router_") and file != os.path.basename(best_model_path):
                os.remove(os.path.join(args_parser.output_dir, file))
                logger.info(f"Removed old checkpoint: {file}")

    logger.info(f"Training finished. Best model at epoch {best_epoch} saved to: {best_model_path}")
    if val_loader:
        logger.info(f"Best Val F1: {best_f1:.4f}")
    else:
        logger.info(f"Best Train Loss: {best_f1:.4f}")

def create_router_dataloader(data, tokenizer, batch_size=32, shuffle=True):
    dataset = CodeDatasetForRouter(data, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_argparse_router():
    parser = argparse.ArgumentParser(description="Train MoE Router for Vulnerability Detection")
    parser.add_argument('--train_files', nargs='+', required=True, help='List of training CSV files (vulnerable only)')
    parser.add_argument('--val_files', nargs='+', default=[], help='List of validation CSV files (optional)')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--num_experts', type=int, default=12, help='Number of CWE groups (experts)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='/drive1/cuongtm/ntat/MulVulMoe/model/routers', help='Where to save router model')
    parser.add_argument('--clean_old_checkpoints', action='store_true', help='Delete old checkpoints, keep only best')
    return parser.parse_args()


def main():
    args_parser = get_argparse_router()
    
    # Setup logging
    setup_logging(log_file=f'/drive1/cuongtm/ntat/MulVulMoe/log/train_router_mulvulmoe.log')
    logger = get_logger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args_parser.pretrained_model)
    backbone = AutoModel.from_pretrained(args_parser.pretrained_model)
    backbone.to(device)
    
    model = MoEVDRouter(backbone, num_experts=args_parser.num_experts)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args_parser.learning_rate)
    criterion = focal_loss
    
    logger.info("Loading training data (vulnerable only)...")
    train_data = load_data(args_parser.train_files)
    train_data = train_data[train_data['label'] == 1]  # Chỉ giữ vulnerable
    logger.info(f"Loaded {len(train_data)} training samples")
    
    train_loader = create_router_dataloader(
        train_data,
        tokenizer,
        batch_size=args_parser.batch_size,
        shuffle=True
    )
    
    val_loader = None
    if args_parser.val_files:
        logger.info("Loading validation data...")
        val_data = load_data(args_parser.val_files)
        val_data = val_data[val_data['label'] == 1]  # Chỉ giữ vulnerable
        logger.info(f"Loaded {len(val_data)} validation samples")
        val_loader = create_router_dataloader(
            val_data,
            tokenizer,
            batch_size=args_parser.batch_size,
            shuffle=False
        )
        
    logger.info("Starting training Router...")
    train_router(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion, # lambda logits, targets: focal_loss(logits, targets, alpha=1.0, gamma=2.0),
        logger=logger,
        args_parser=args_parser,
        train_loader=train_loader,
        val_loader=val_loader
    )

    logger.info("Training Router finished.")

if __name__ == '__main__':
    main()