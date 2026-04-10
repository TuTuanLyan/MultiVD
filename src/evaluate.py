import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import argparse
import os

from ultis import load_data, create_data_loader
from logger import setup_logging, get_logger
from mulvulmoe import MulVulExpert 


def evaluate_model(model, device, test_loader, test_data, id_to_lang, logger, cwe_id):
    """
    Đã sửa: Truyền thêm test_data gốc để lấy CWE_ID
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_langs = []

    logger.info("Bat dau chay Inference tren tap Test...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            langs = batch.get('language', torch.zeros_like(labels))

            # Inference
            logits, _ = model(input_ids, attention_mask, language=None)
            
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5).cpu().numpy().astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_langs.extend(langs.cpu().numpy().astype(int))

    # ==================== TẠO DATAFRAME KẾT QUẢ ====================
    df_res = pd.DataFrame({
        'CWE_ID': test_data['CWE_ID'].values,      # ← THÊM CỘT CWE_ID
        'label': all_labels,
        'pred': all_preds,
        'lang_id': all_langs
    })
    
    df_res['language'] = df_res['lang_id'].map(id_to_lang)
    df_res['is_correct'] = df_res['label'] == df_res['pred']

    # Lưu predictions với CWE_ID
    results_dir = '/drive1/cuongtm/ntat/MulVulMoe/results'
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f'predictions_cwe_{cwe_id}.csv')
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Da luu chi tiet predictions (co CWE_ID) vao: {out_csv}")

    # Hàm in thống kê
    def print_stats(df_subset, name):
        tn = len(df_subset[(df_subset['label'] == 0) & (df_subset['pred'] == 0)])
        fp = len(df_subset[(df_subset['label'] == 0) & (df_subset['pred'] == 1)])
        tp = len(df_subset[(df_subset['label'] == 1) & (df_subset['pred'] == 1)])
        fn = len(df_subset[(df_subset['label'] == 1) & (df_subset['pred'] == 0)])
        
        total = len(df_subset)
        
        logger.info(f"--- THONG KE CHI TIET {name} (Total: {total}) ---")
        logger.info(f" [Safe  0]  Tong: {tn + fp} | Dung: {tn} | Sai (doan la Vuln): {fp}")
        logger.info(f" [Vuln  1]  Tong: {tp + fn} | Dung: {tp} | Sai (doan la Safe): {fn}")
        logger.info(f" Accuracy: {(tn + tp) / total * 100:.2f}%")
        logger.info("-" * 60)

    # 1. Overall Report
    logger.info(f"\n{'='*60}")
    logger.info(f"[OVERALL PERFORMANCE - CWE {cwe_id}]")
    logger.info(f"{'='*60}")
    print_stats(df_res, "OVERALL")
    
    overall_report = classification_report(
        df_res['label'], df_res['pred'],
        target_names=['Safe (0)', 'Vuln (1)'],
        zero_division=0
    )
    logger.info(f"\n{overall_report}")

    # 2. Report theo từng ngôn ngữ
    unique_langs = df_res['language'].dropna().unique()
    for lang in sorted(unique_langs):
        df_lang = df_res[df_res['language'] == lang]
        if len(df_lang) == 0:
            continue
            
        logger.info(f"\n{'*'*60}")
        logger.info(f"[PERFORMANCE FOR LANGUAGE: {lang.upper()}]")
        logger.info(f"{'*'*60}")
        print_stats(df_lang, lang.upper())
        
        lang_report = classification_report(
            df_lang['label'], df_lang['pred'],
            labels=[0, 1],
            target_names=['Safe (0)', 'Vuln (1)'],
            zero_division=0
        )
        logger.info(f"\n{lang_report}")


def get_argparse():
    parser = argparse.ArgumentParser(description="Inference & Evaluate MulVul Expert")
    
    parser.add_argument('--test_files', nargs='+', required=True, 
                        help='List of test files (JSONL)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Duong dan den file .pth da train')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--pool_length', type=int, default=5)
    parser.add_argument('--num_langs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cwe_id', type=str, required=True, help='VD: 022')

    return parser.parse_args()


def main():
    args = get_argparse()
    
    # Setup logging
    os.makedirs('/drive1/cuongtm/ntat/MulVulMoe/log/', exist_ok=True)
    log_file = f'/drive1/cuongtm/ntat/MulVulMoe/log/evaluate_mulvul_fe_cwe_{args.cwe_id}.log'
    setup_logging(log_file=log_file, reset_file=True)
    logger = get_logger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Tokenizer & Backbone
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    backbone = AutoModel.from_pretrained(args.pretrained_model)
    
    lang_map = {'python': 0, 'c': 1}
    id_to_lang = {v: k for k, v in lang_map.items()}
    
    # Khởi tạo model
    model = MulVulExpert(
        pretrained_model=backbone,
        num_labels=1,
        num_langs=args.num_langs,
        pool_length=args.pool_length,
        lang_map=lang_map
    )
    
    # Load trọng số
    logger.info(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Load test data gốc (để lấy CWE_ID)
    logger.info(f"Loading test data: {args.test_files}")
    test_data = load_data(args.test_files)
    
    num_neg = len(test_data[test_data['label'] == 0])
    num_pos = len(test_data[test_data['label'] == 1])
    logger.info(f"Test set distribution - Safe(0): {num_neg} | Vuln(1): {num_pos}")
    
    # DataLoader
    test_loader = create_data_loader(test_data, tokenizer, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate (truyền thêm test_data gốc)
    evaluate_model(model, device, test_loader, test_data, id_to_lang, logger, args.cwe_id)
    
    logger.info("Hoan tat Inference va Evaluate!")


if __name__ == "__main__":
    main()