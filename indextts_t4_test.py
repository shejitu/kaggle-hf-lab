# IndexTTS-2.5 on Kaggle T4 GPU smoke test (GitHub Actions -> Kaggle kernel)
# Runs inside /kaggle/working/kaggle-hf-lab; checkpoints + output.wav stay in kernel output.
# T4 (Turing) has no native bf16: try bf16 first, fall back to fp32 on failure.
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--lang', default='zh')
    a = ap.parse_args()

    import torch
    print('[ENV] torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print('[ENV] GPU:', torch.cuda.get_device_name(0),
              '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB', flush=True)

    from huggingface_hub import snapshot_download
    t0 = time.time()
    print('[1/3] Downloading IndexTeam/IndexTTS-2.5 checkpoints...', flush=True)
    snapshot_download(repo_id='IndexTeam/IndexTTS-2.5', local_dir='checkpoints_25')
    print(f'[1/3] checkpoints ready in {time.time() - t0:.0f}s', flush=True)

    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.examples_downloader import ensure_test_sample_available
    prompt = ensure_test_sample_available()
    print('[2/3] prompt wav:', prompt, flush=True)

    cfg = 'checkpoints_25/config.yaml'
    mdir = 'checkpoints_25'
    t1 = time.time()
    tts = None
    use_bf16 = True
    try:
        tts = IndexTTS2(cfg_path=cfg, model_dir=mdir, use_bf16=True, use_cuda_kernel=False)
        print(f'[2/3] model loaded (bf16) in {time.time() - t1:.0f}s', flush=True)
    except Exception as e:
        print(f'[2/3] bf16 load failed: {e!r} -> retrying fp32', flush=True)
        torch.cuda.empty_cache()
        use_bf16 = False

    t2 = time.time()
    print('[3/3] Synthesizing:', a.text, flush=True)
    try:
        tts.infer(spk_audio_prompt=str(prompt), text=a.text, output_path='output.wav', lang=a.lang)
    except Exception as e:
        if use_bf16:
            print(f'[3/3] bf16 inference failed: {e!r} -> reloading fp32 and retrying', flush=True)
            del tts
            torch.cuda.empty_cache()
            tts = IndexTTS2(cfg_path=cfg, model_dir=mdir, use_bf16=False, use_cuda_kernel=False)
            tts.infer(spk_audio_prompt=str(prompt), text=a.text, output_path='output.wav', lang=a.lang)
        else:
            raise

    size = os.path.getsize('output.wav')
    print(f'SYNTH_DONE in {time.time() - t2:.0f}s -> output.wav ({size} bytes)', flush=True)
    if size < 1000:
        raise SystemExit('output.wav too small, synthesis likely failed')


if __name__ == '__main__':
    main()
