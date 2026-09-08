#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IndexTTS-2.5 on Kaggle T4 — 双模式运行脚本（由 GitHub Actions 调度）

模式 A (synth): 给参考音色 URL + 文本 -> 合成 -> /kaggle/working/output.wav
模式 B (serve): 启动 Gradio share=True 公网页面，地址写入 /kaggle/working/gradio_url.txt

设计要点：
- 所有重文件（模型缓存/venv/克隆仓库）都放 /kaggle/tmp，避免污染 kernel output
- 只有成品文件（output.wav / gradio_url.txt）进 /kaggle/working
- T4 (sm_75) 无原生 bf16，直接用 fp32 加载，保证确定性
"""
import argparse
import os
import shutil
import sys
import threading
import time
import urllib.request

TMP = "/kaggle/tmp"
WORK = "/kaggle/working"
SAMPLE_VOICE_URL = ("https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/"
                    "resolve/main/voice_01.wav")

# 模型缓存一律放 /kaggle/tmp，不进 working
os.makedirs(TMP, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(TMP, "hf"))


def log(msg: str) -> None:
    print(msg, flush=True)


def get_ref(ref: str) -> str:
    """参考音色：'sample' 用官方示例人声；http(s) URL 下载到 TMP；否则按本地路径处理。"""
    if ref == "sample":
        dst = os.path.join(TMP, "voice_01.wav")
        if not os.path.exists(dst):
            log("[REF] downloading official sample voice ...")
            urllib.request.urlretrieve(SAMPLE_VOICE_URL, dst)
        return dst
    if ref.startswith("http://") or ref.startswith("https://"):
        ext = os.path.splitext(ref.split("?")[0])[1] or ".wav"
        dst = os.path.join(TMP, "ref_voice" + ext)
        log("[REF] downloading reference voice ...")
        urllib.request.urlretrieve(ref, dst)
        return dst
    if not os.path.exists(ref):
        raise FileNotFoundError(f"reference voice not found: {ref}")
    return ref


def load_tts():
    """下载 IndexTeam/IndexTTS-2.5 并构建推理器（fp32，适配 T4）。"""
    from huggingface_hub import snapshot_download
    from indextts.infer_v2_5 import IndexTTS2

    mdir = snapshot_download("IndexTeam/IndexTTS-2.5")
    cfg = None
    for root, _dirs, files in os.walk(mdir):
        if "config.yaml" in files:
            cfg = os.path.join(root, "config.yaml")
            break
    if cfg is None:
        raise FileNotFoundError("config.yaml not found in model snapshot")
    log("[MODEL] snapshot ready, loading (fp32 for T4) ...")
    t0 = time.time()
    tts = IndexTTS2(cfg_path=cfg, model_dir=mdir,
                    use_cuda_kernel=False, use_bf16=False)
    log(f"[MODEL] loaded in {time.time() - t0:.1f}s")
    return tts


def do_synth(args) -> None:
    ref = get_ref(args.ref)
    log(f"[1/3] ref voice ready: {ref}")
    tts = load_tts()
    log("[2/3] synthesizing ...")
    out = os.path.join(WORK, "output.wav")
    t0 = time.time()
    tts.infer(spk_audio_prompt=ref, text=args.text,
              output_path=out, lang=args.lang)
    log(f"[3/3] SYNTH_DONE size={os.path.getsize(out)} "
        f"elapsed={time.time() - t0:.1f}s out={out}")


def do_serve(args) -> None:
    import gradio as gr

    tts = load_tts()

    def gen(ref_audio, text, lang_choice):
        if ref_audio is None or not text.strip():
            raise gr.Error("请上传参考音频并输入文本")
        out = os.path.join(TMP, f"gr_out_{int(time.time())}.wav")
        tts.infer(spk_audio_prompt=ref_audio, text=text.strip(),
                  output_path=out, lang=lang_choice)
        return out

    demo = gr.Interface(
        fn=gen,
        inputs=[
            gr.Audio(type="filepath", label="参考音色（上传 5~15 秒清晰人声）"),
            gr.Textbox(label="合成文本", value="你好，这是 IndexTTS 语音合成测试。"),
            gr.Radio(["zh", "en"], value="zh", label="语言"),
        ],
        outputs=[gr.Audio(label="合成结果")],
        title="IndexTTS-2.5 · Kaggle T4 语音合成",
        description="上传参考音色 + 输入文本 → 点击提交合成。会话关闭后地址失效。",
    )

    # 看门狗：到时自动退出，让 Kaggle 保存版本并释放 GPU 配额
    if args.hours > 0:
        def _watchdog():
            time.sleep(args.hours * 3600)
            log(f"[SERVE] {args.hours}h watchdog reached, shutting down.")
            os._exit(0)
        threading.Thread(target=_watchdog, daemon=True).start()

    url_file = os.path.join(WORK, "gradio_url.txt")
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=True, show_error=True)
    # launch 返回后 share_url 已生成
    share = getattr(demo, "share_url", None) or ""
    if share:
        with open(url_file, "w") as f:
            f.write(share + "\n")
    log("SERVE_READY")
    demo.block_thread()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synth", "serve"], required=True)
    p.add_argument("--ref", default="sample",
                   help="'sample' | http(s) URL | local path")
    p.add_argument("--text", default="你好，这是通过 GitHub Actions 在 Kaggle T4 上运行的 IndexTTS 2.5 语音合成。")
    p.add_argument("--lang", default="zh", choices=["zh", "en"])
    p.add_argument("--hours", type=float, default=3.0,
                   help="serve 模式存活小时数（看门狗）")
    args = p.parse_args()
    log(f"[START] mode={args.mode} lang={args.lang} hours={args.hours}")
    if args.mode == "synth":
        do_synth(args)
    else:
        do_serve(args)


if __name__ == "__main__":
    main()
