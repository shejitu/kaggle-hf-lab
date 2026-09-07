#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 Kaggle GPU 上快速测试任意 Hugging Face 模型。
用法：python hf_model_test.py --model <model_id> --text <输入文本>
"""
import argparse
import platform
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    ap.add_argument("--text", default="I love this movie, it is fantastic!")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[环境] Python {platform.python_version()}")

    import torch
    print(f"[环境] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")
        device = "cuda"
    else:
        print("[GPU] 未检测到 GPU，回退到 CPU")
        device = "cpu"

    print(f"[模型] {args.model}")
    print(f"[输入] {args.text}")

    from transformers import pipeline

    t0 = time.time()
    pipe = pipeline(model=args.model, device=device)
    load_s = time.time() - t0
    print(f"[加载] 完成，耗时 {load_s:.1f}s")

    t1 = time.time()
    result = pipe(args.text)
    infer_s = time.time() - t1
    print(f"[推理] 完成，耗时 {infer_s:.2f}s")
    print(f"[结果] {result}")
    print("=" * 60)
    print("TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
