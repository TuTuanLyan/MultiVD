#!/usr/bin/env python3
"""Thông số thật của từng backbone, đọc từ checkpoint chứ không lấy theo trí nhớ.

Cần trước khi đưa một backbone mới vào hàng đợi, vì hai điều quyết định cách chạy
nó đều không đoán được: `build_backbone` có lấy đúng phần encoder không, và tầng
biểu diễn nào là chuẩn để đọc ra (`cls` hay `mean`).

Số tham số in ra tách riêng phần embedding. Với các model cùng cỡ, phần khác nhau
thường nằm gần hết ở embedding do vocab khác nhau, nên "125M" của model này không
so trực tiếp được với "125M" của model kia nếu không tách.
"""

import argparse
import sys

sys.path.insert(0, "src")

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from model import build_backbone  # noqa: E402

DEFAULT = [
    "microsoft/unixcoder-base",
    "microsoft/codebert-base",
    "Salesforce/codet5p-220m",
    "Salesforce/codet5-base",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT)
    args = parser.parse_args()

    for name in args.models:
        config = AutoConfig.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        backbone = build_backbone(name)

        total = sum(p.numel() for p in backbone.parameters())
        embedding = sum(p.numel() for p in backbone.get_input_embeddings().parameters())
        ids = tokenizer("int main(){ return 0; }")["input_ids"]
        first = tokenizer.convert_ids_to_tokens(ids[:1])

        print(f"\n### {name}")
        print(f"  model_type              {config.model_type}")
        print(f"  is_encoder_decoder      {getattr(config, 'is_encoder_decoder', False)}")
        print(f"  lớp backbone nạp về     {type(backbone).__name__}")
        print(f"  tokenizer               {type(tokenizer).__name__}")
        print(f"  token ở vị trí 0        {first}")
        print(f"  hidden size             {getattr(config, 'hidden_size', getattr(config, 'd_model', '?'))}")
        print(f"  số lớp encoder          {getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', '?'))}")
        print(f"  vocab                   {getattr(config, 'vocab_size', '?'):,}")
        print(f"  max position            {getattr(config, 'max_position_embeddings', 'n/a')}")
        print(f"  THAM SỐ (encoder)       {total:,}")
        print(f"    trong đó embedding    {embedding:,}  ({embedding / total:.1%})")
        print(f"    ngoài embedding       {total - embedding:,}")


if __name__ == "__main__":
    main()
