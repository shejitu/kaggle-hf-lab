# kaggle-hf-lab — GitHub → Kaggle GPU 调度实验场

push 代码或手动触发 workflow，即可在你的 Kaggle 账号（xwdfyx）的**免费 GPU** 上自动测试 Hugging Face 模型，日志自动回传 GitHub。

## 工作原理

1. GitHub Actions 使用 [KevKibe/kaggle-script-action](https://github.com/KevKibe/kaggle-script-action) 生成一个 ipynb 推到 Kaggle kernel（`xwdfyx/hf-model-gpu-test`）。
2. Notebook 里依次执行：`git clone 本仓库` → `pip install -r requirements.txt` → `python hf_model_test.py --model <模型> --text <文本>`。
3. Action 轮询 kernel 状态，跑完后用 `kaggle kernels output` 拉取完整日志显示在 GitHub Actions 里。

## 用法

- **手动触发**：仓库 → Actions → "Kaggle GPU 模型测试" → Run workflow → 填入 HF 模型 ID 和测试文本。
- **自动触发**：往 main push 即用默认模型跑一次。

## 前提

- GitHub Secrets：`KAGGLE_USERNAME` / `KAGGLE_KEY`（来自 Kaggle 账号设置页的 API Token）。
- 仓库必须**公开**（Kaggle 服务器要能 `git clone`）。
- Kaggle 免费 GPU 每周约 30 小时配额。

## 换模型测试

把 Actions 手动触发时的 `model_id` 换成任意 HF 模型即可，例如：
- `Qwen/Qwen2.5-0.5B`
- `sentence-transformers/all-MiniLM-L6-v2`
- `distilbert-base-uncased-finetuned-sst-2-english`

## 想引用已有的 Kaggle 数据集（模型离线挂载）？

在 workflow 的 action 配置里加：

```yaml
dataset_sources: |
  ["xwdfyx/你的数据集slug"]
```

模型就会出现在 `/kaggle/input/` 下，脚本里直接从该路径加载，不占运行时带宽。
