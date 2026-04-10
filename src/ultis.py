import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json

import torch
import torch.nn.functional as F

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len: int = 512, language_map=None):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.language_map = language_map or {'python': 0, 'c': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = row['code']
        label = row['label']
        
        # Hỗ trợ cả cột 'lang' và 'language'
        lang_str = str(row.get('lang') or row.get('language', 'python')).lower().strip()
        lang_id = self.language_map.get(lang_str, 0)

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'language': torch.tensor(lang_id, dtype=torch.long)
        }


class CodeDatasetForRouter(Dataset):
    def __init__(self, data, tokenizer, max_len: int = 512):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = row['code']
        label = row['CWE_ID']
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'CWE_ID': torch.tensor(label, dtype=torch.long)
        }


def load_data(input_data):
    """
    Load data một cách tối ưu bộ nhớ:
      - Nếu input_data là pandas.DataFrame → trả về luôn
      - Nếu là list/tuple các đường dẫn file:
          + File .jsonl → đọc streaming từng dòng
          + File .csv    → dùng pd.read_csv()
    """
    if isinstance(input_data, pd.DataFrame):
        print(f"Received DataFrame directly ({len(input_data)} samples)")
        return input_data

    if not isinstance(input_data, (list, tuple)):
        raise TypeError("load_data expects a list of file paths or a pandas.DataFrame")

    data_list = []
    for file_path in input_data:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            if file_path.lower().endswith('.jsonl') or file_path.lower().endswith('.json'):
                # Đọc JSONL theo kiểu streaming để tiết kiệm bộ nhớ
                records = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}")
                
                df = pd.DataFrame(records)
                print(f"Loaded JSONL: {file_path} ({len(df)} samples)")
            
            else:
                # CSV thì vẫn dùng pandas (thường file CSV không quá lớn)
                df = pd.read_csv(file_path)
                print(f"Loaded CSV: {file_path} ({len(df)} samples)")
            
            data_list.append(df)
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    if not data_list:
        raise ValueError("No valid data files could be loaded!")

    combined_df = pd.concat(data_list, ignore_index=True)
    print(f"Total samples after combining: {len(combined_df)}")
    return combined_df


def create_data_loader(data, tokenizer, batch_size=16, shuffle=True):
    """Tạo DataLoader từ DataFrame"""
    dataset = CodeDataset(data, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


# Map CWE nếu bạn vẫn cần dùng sau này
cwe_map = {
    'cwe-022': 0,
    'cwe-078': 1,
    'cwe-079': 2,
    'cwe-089': 3,
}

def focal_loss(logits, targets, gamma=2.0, reduction='mean'):
    """
    Focal Loss cho multi-class classification (dùng cho Router).
    Giúp xử lý imbalance giữa các nhóm CWE rất tốt.
    
    Args:
        logits:   [batch_size, num_experts]
        targets:  [batch_size] (long tensor, chứa index của CWE group)
        gamma:    focusing parameter (mặc định = 2.0)
        reduction: 'mean' hoặc 'sum'
    """
    # Tính Cross Entropy loss cho từng sample
    ce_loss = F.cross_entropy(logits, targets, reduction='none')   # [batch_size]
    
    # pt = p^t (probability của class đúng)
    pt = torch.exp(-ce_loss)
    
    # Focal Loss
    focal_loss_val = (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val