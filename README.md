# Wav2Vec2-based Speaker Identification on AISHELL-4

This project implements Mandarin speaker identification for AISHELL-4 meeting segments using transfer learning on `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`.  

# Speaker Identification on AISHELL-4 using Wav2Vec2

This repository implements Mandarin speaker identification on AISHELL-4 meeting speech data using Wav2Vec2 models. We provide two approaches:

- **Baseline model**: Trains a classifier on top of frozen Wav2Vec2 encoder outputs.
- **Transfer model**: Applies domain and task transfer. It features session-level training, mean+std pooling, and an enhanced classifier with BatchNorm.


---

## 🚀 Setup Instructions

### 1️⃣ Install Python dependencies

Ensure Python 3.9+ is installed. If you don’t have it, use 'venv' or `conda`.

Create and activate a virtual environment:

```bash
# With venv
$ python3 -m venv venv
$ source venv/bin/activate

# OR with conda
$ conda create -n [w2v2speaker] python=3.9 -y
$ conda activate [w2v2speaker]
```

```bash
pip install --upgrade pip
pip install -r model/requirements.txt


Example requirements.txt content:

```bash
torch>=1.13.0
torchaudio>=0.13.0
transformers
pandas
scikit-learn
pytorch_lightning>=2.0.0
```
If you need specific CUDA versions for PyTorch, follow PyTorch official installation instructions(https://pytorch.org/get-started/locally/).

## Preparing AISHELL-4 Data

Place original audio and RTTM files:

```bash

data/data_original/wav/*.flac
data/data_original/rttm/*.rttm

```

Segment the audio:

```bash

python data/prepare_segments_from_rttm.py \
  --rttm_dir data/data_original/rttm \
  --pcm_dir data/data_original/wav \
  --output_dir data/segments

```

Rename segments to match data_fixed.csv:

```bash

python data/rename_segments_to_match_csv.py \
  --segments_dir data/segments \
  --csv_path data/data_fixed.csv

```

## Configuration

Edit model/config.yaml to modify:

```bash

batch_size: 2/4
learning_rate: 3e-5
max_epochs: 40
freeze_encoder_layers: 8
pooling_method: "mean+std"
pretrained_model: "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

```

## Downloading Pre-trained Model

The jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn model is used and will be automatically downloaded via Hugging Face Transformers during training.

If you want to manually download:

```bash

transformers-cli download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn

```

## Running the Experiment

### Baseline model

Trains on entire dataset as one classification task.

```bash

python model/train_baseline.py

```

### Transfer model (session-level)

Fine-tunes models per meeting segment, applying transfer learning.

```bash

python model/train_transfer.py

```

Both will:
· Save best checkpoints under lightning_logs/version_X/checkpoints/
· Log training/validation metrics
· Resume from checkpoints if interrupted

## Logs and Results

Check lightning_logs/ for TensorBoard or CSV logs.

Best models saved as:

```bash

lightning_logs/version_X/checkpoints/{meeting_id}-best.ckpt

```

Metrics saved to metrics.csv inside each version_X.

## Notes

Checkpoints will be saved for each session with best validation accuracy.

You can modify train.py parameters (e.g., batch size, learning rate) as needed.

This setup assumes AISHELL-4 is properly licensed for your use.

Ensure sufficient GPU memory (A100 with over 20GB is recommended for batch size >= 2).


## References

· AISHELL-4 dataset: https://www.openslr.org/111/

· Wav2Vec2 paper: https://arxiv.org/abs/2006.11477

· PyTorch Lightning: https://www.pytorchlightning.ai/

· HuggingFace Transformers: https://huggingface.co/docs/transformers/


