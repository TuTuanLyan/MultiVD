import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import os
from transformers import AutoTokenizer, AutoModel
from mulvulmoe import MulVulExpert, MoEVDRouter
from ultis import cwe_map, load_data
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from logger import setup_logging, get_logger

# Map ngược: group ID → CWE string (để log dễ đọc)
reverse_cwe_map = {v: k.upper() for k, v in cwe_map.items()}


def load_router(pretrained_model_name, num_experts, device, router_pth_path, logger):
    logger.info(f"Loading Router from: {router_pth_path}")
    backbone = AutoModel.from_pretrained(pretrained_model_name)
    model = MoEVDRouter(backbone, num_experts=num_experts)
    model.load_state_dict(torch.load(router_pth_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Router loaded successfully!")
    return model


def load_expert(pretrained_model_name, device, expert_pth_path, expert_id, logger):
    logger.info(f"Loading Expert {expert_id} from: {expert_pth_path}")
    backbone = AutoModel.from_pretrained(pretrained_model_name)
    model = MulVulExpert(backbone, num_labels=1)
    model.load_state_dict(torch.load(expert_pth_path, map_location=device))
    model.to(device)
    model.eval()
    cwe_name = reverse_cwe_map.get(expert_id, f"Group_{expert_id}")
    logger.info(f"Expert {expert_id} ({cwe_name}) loaded successfully!")
    return model


def moe_inference(router, experts, code_text, tokenizer, device, top_k=2, max_len=512):
    inputs = tokenizer(
        code_text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        # Router inference
        cwe_logits = router(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        cwe_probs = F.softmax(cwe_logits, dim=-1)

        # Chọn top-k experts
        top_k_probs, top_k_indices = torch.topk(cwe_probs, top_k, dim=1)
        top_k_weights = F.softmax(top_k_probs, dim=-1)

        # Kết hợp dự đoán từ các expert
        vul_prob = 0.0
        for i in range(top_k):
            expert_idx = top_k_indices[0, i].item()
            weight = top_k_weights[0, i].item()
            expert = experts[expert_idx]

            expert_logits, _ = expert(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                language=None
            )
            expert_prob = torch.sigmoid(expert_logits.squeeze()).item()
            vul_prob += weight * expert_prob

    # Thông tin thêm để debug
    predicted_cwe_group = torch.argmax(cwe_probs, dim=1).item()
    top_k_cwe_list = top_k_indices[0].cpu().tolist()
    top_k_weights_list = top_k_weights[0].cpu().tolist()

    return vul_prob, predicted_cwe_group, top_k_cwe_list, top_k_weights_list


def evaluate_moe(router, experts, test_data, tokenizer, device, logger,
                 top_k=2, threshold=0.5, result_csv='moe_evaluation_results.csv'):
    logger.info(f"Bắt đầu đánh giá MoE trên {len(test_data)} mẫu | top_k={top_k} | threshold={threshold}")
    
    # Chuẩn bị nhãn true CWE group
    test_data = test_data.copy()
    test_data['true_cwe_group'] = test_data['CWE_ID'].map(cwe_map).fillna(-1).astype(int)

    # Thống kê dataset
    num_vul = (test_data['label'] == 1).sum()
    num_non_vul = (test_data['label'] == 0).sum()
    num_unmapped = (test_data['true_cwe_group'] == -1).sum()
    logger.info(f"Dataset stats → Vul (1): {num_vul} | Non-Vul (0): {num_non_vul} | Unmapped CWE: {num_unmapped}")

    results = []
    all_preds = []
    all_labels = []

    logger.info("Đang inference từng mẫu...")

    for idx, row in enumerate(test_data.itertuples(), 1):
        code = row.code
        true_label = row.label
        true_cwe_group = row.true_cwe_group

        vul_prob, predicted_cwe_group, top_k_cwe, top_k_w = moe_inference(
            router, experts, code, tokenizer, device, top_k
        )

        pred_label = 1 if vul_prob >= threshold else 0
        correct = 1 if pred_label == true_label else 0

        results.append({
            'index': idx,
            'code_preview': code[:150] + '...' if len(code) > 150 else code,
            'true_label': true_label,
            'pred_label': pred_label,
            'vul_prob': round(vul_prob, 4),
            'correct': correct,
            'true_cwe_group': true_cwe_group,
            'predicted_cwe_group': predicted_cwe_group,
            'top_k_cwe_groups': top_k_cwe,
            'top_k_weights': [round(w, 4) for w in top_k_w]
        })

        all_preds.append(pred_label)
        all_labels.append(true_label)


    # ==================== BÁO CÁO CHI TIẾT (Classification Report) ====================
    logger.info("\n" + "="*80)
    logger.info("FINAL CLASSIFICATION REPORT (Binary: Safe vs Vulnerable)")
    logger.info("="*80)

    target_names = ['Safe (0) - TrueNon', 'Vuln (1) - TrueVul']
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    logger.info("\n" + report)

    # Confusion Matrix chi tiết
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    logger.info("\n" + "="*80)
    logger.info("CONFUSION MATRIX & DETAILED METRICS")
    logger.info("="*80)
    logger.info(f"True Negative  (TrueNon - Safe đúng)     : {tn:6d}")
    logger.info(f"False Positive (dự đoán Vul nhưng Safe)  : {fp:6d}")
    logger.info(f"False Negative (dự đoán Safe nhưng Vul)  : {fn:6d}")
    logger.info(f"True Positive  (TrueVul - Vul đúng)      : {tp:6d}")
    logger.info("-" * 80)
    logger.info(f"Accuracy  : {(tn + tp) / len(all_labels):.4f}")
    logger.info(f"Precision (Vuln): {tp / (tp + fp):.4f}" if (tp + fp) > 0 else "Precision (Vuln): 0.0000")
    logger.info(f"Recall    (Vuln): {tp / (tp + fn):.4f}" if (tp + fn) > 0 else "Recall    (Vuln): 0.0000")
    logger.info(f"F1-score  (Vuln): {f1_score(all_labels, all_preds, zero_division=0):.4f}")
    logger.info("="*80)

    # Lưu kết quả chi tiết
    result_df = pd.DataFrame(results)
    result_df.to_csv(result_csv, index=False, encoding='utf-8')
    logger.info(f"Đã lưu đầy đủ kết quả chi tiết vào: {os.path.abspath(result_csv)}")
    logger.info(f"   Tổng mẫu: {len(result_df)} | Correct: {result_df['correct'].sum()} | Wrong: {len(result_df) - result_df['correct'].sum()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Full MoEVD (Router + Experts) - MulVulMoE")
    
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--router_pth', type=str, required=True, help='Path to router_best.pth')
    parser.add_argument('--expert_pths', nargs='+', required=True, help='Paths to all expert .pth files (theo thứ tự group)')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test file (JSONL/CSV)')
    parser.add_argument('--CWE_ID', type=str, required=True, help='CWE ID dùng để log (ví dụ: 022)')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--result_csv', type=str, default=None,
                        help='Path to save detailed CSV (nếu không truyền sẽ tự tạo theo CWE_ID)')

    args = parser.parse_args()

    # Tự động tạo tên file kết quả nếu chưa có
    if args.result_csv is None:
        args.result_csv = f'/drive1/cuongtm/ntat/MulVulMoe/results/mulvulmoe_evaluation_cwe_{args.CWE_ID}.csv'

    # Logging
    log_file = f'/drive1/cuongtm/ntat/MulVulMoe/log/test_mulvulmoe_{args.CWE_ID}.log'
    setup_logging(log_file=log_file, reset_file=True)
    logger = get_logger(__name__)

    logger.info("=" * 90)
    logger.info(f"START MoEVD EVALUATION - CWE-{args.CWE_ID}")
    logger.info("=" * 90)
    logger.info(f"Arguments: {vars(args)}")
    logger.info("-" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Load Router
    router = load_router(args.pretrained_model, args.num_experts, device, args.router_pth, logger)

    # Load tất cả Experts
    experts = []
    for i, pth in enumerate(args.expert_pths):
        expert = load_expert(args.pretrained_model, device, pth, i, logger)
        experts.append(expert)

    # Load test data
    logger.info(f"Loading test data from: {args.test_file}")
    test_data = load_data([args.test_file])
    logger.info(f"Loaded {len(test_data)} test samples.")

    # Chạy evaluation
    evaluate_moe(
        router=router,
        experts=experts,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        logger=logger,
        top_k=args.top_k,
        threshold=args.threshold,
        result_csv=args.result_csv
    )

    logger.info("\nEvaluation completed successfully!")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()