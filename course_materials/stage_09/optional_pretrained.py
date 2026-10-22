"""可选实践：加载真实预训练情感分类器。首次运行需要联网下载权重。"""

import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def main():
    parser = argparse.ArgumentParser(description="使用已有预训练模型完成英文情感分类")
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--local-only", action="store_true", help="仅使用已缓存模型，不联网")
    parser.add_argument("--text", nargs="+", default=["I enjoyed learning how attention works.", "The instructions were confusing."])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=args.local_only, trust_remote_code=False, use_safetensors=True)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    for text, result in zip(args.text, classifier(args.text)):
        print(text)
        print(result)


if __name__ == "__main__":
    main()
