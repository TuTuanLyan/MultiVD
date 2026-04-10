import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import classification_report
import argparse
import os

from ultis import load_data, create_data_loader
from logger import setup_logging, get_logger
from mulvulmoe import MulVulExpert 

def train_expert(model, device, optimizer, criterion, logger, args, train_loader, val_loader=None):
    model.to(device)
    
    best_f1_macro = -1.0
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, f"mulvul_expert{args.cwe_id}.pth")
    
    patience = 5
    patience_counter = 0

    logger.info(f"Start training MulVulExpert - CWE-{args.cwe_id} | Epochs: {args.epochs} | Pool: {args.pool_length} | Aux weight: {args.aux_weight}")

    for epoch in range(args.epochs):
        model.train()
        total_bce = 0.0
        total_aux = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1).float()
            languages = batch['language'].to(device)

            optimizer.zero_grad()
            logits, aux_loss = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   language=languages)
            
            bce_loss = criterion(logits, labels)
            total_loss = bce_loss + args.aux_weight * aux_loss

            total_loss.backward()
            optimizer.step()

            total_bce += bce_loss.item()
            total_aux += aux_loss.item()

        avg_bce = total_bce / len(train_loader)
        avg_aux = total_aux / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | BCE: {avg_bce:.4f} | Aux: {avg_aux:.4f} | Total: {avg_bce + args.aux_weight*avg_aux:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    logits, _ = model(input_ids, attention_mask, language=None)
                    probs = torch.sigmoid(logits).squeeze(1)
                    preds = (probs >= 0.5).cpu().numpy().astype(int)

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy().astype(int))

            report_dict = classification_report(all_labels, all_preds,
                                                target_names=['Safe (0)', 'Vuln (1)'],
                                                output_dict=True, zero_division=0)
            logger.info(f"\n[Validation Report - Epoch {epoch+1}]\n{classification_report(all_labels, all_preds, target_names=['Safe (0)', 'Vuln (1)'], zero_division=0)}")
            
            current_f1 = report_dict['macro avg']['f1-score']
            if current_f1 > best_f1_macro:
                best_f1_macro = current_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model! Macro F1: {current_f1:.4f} (Epoch {best_epoch})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    logger.info(f"Training finished! Best model at epoch {best_epoch}: {best_model_path}")

def get_argparse():
    parser = argparse.ArgumentParser(description="Train MulVulExpert với auxiliary language loss")
    
    parser.add_argument('--train_files', nargs='+', required=True,
                        help='List file train (JSONL)')
    parser.add_argument('--val_files', nargs='+', default=[],
                        help='List file val (JSONL)')
    
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--pool_length', type=int, default=5)
    parser.add_argument('--num_langs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--aux_weight', type=float, default=0.1, help='Trọng số auxiliary language loss (paper dùng 0.1)')
    parser.add_argument('--output_dir', type=str, default='/drive1/cuongtm/ntat/VD/model/experts')
    parser.add_argument('--cwe_id', type=str, required=True, help='VD: 022')

    return parser.parse_args()

def main():
    args = get_argparse()
    
    os.makedirs('/drive1/cuongtm/ntat/MulVulMoe/log/', exist_ok=True)
    log_file = f'/drive1/cuongtm/ntat/MulVulMoe/log/train_mulvul_expert{args.cwe_id}.log'
    setup_logging(log_file=log_file, reset_file=True)
    logger = get_logger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    backbone = AutoModel.from_pretrained(args.pretrained_model)
    
    model = MulVulExpert(
        pretrained_model=backbone,
        num_labels=1,
        num_langs=args.num_langs,
        pool_length=args.pool_length
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    logger.info(f"Loading training data: {args.train_files}")
    train_data = load_data(args.train_files)
    
    num_neg = len(train_data[train_data['label'] == 0])
    num_pos = len(train_data[train_data['label'] == 1])
    weight_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    logger.info(f"Class distribution - Safe: {num_neg} | Vuln: {num_pos} | pos_weight: {weight_ratio:.2f}")
    
    pos_weight = torch.tensor([weight_ratio]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_loader = create_data_loader(train_data, tokenizer, batch_size=args.batch_size, shuffle=True)
    
    val_loader = None
    if args.val_files:
        logger.info(f"Loading validation data: {args.val_files}")
        val_data = load_data(args.val_files)
        val_loader = create_data_loader(val_data, tokenizer, batch_size=args.batch_size, shuffle=False)
    
    train_expert(model, device, optimizer, criterion, logger, args, train_loader, val_loader)

if __name__ == "__main__":
    main()