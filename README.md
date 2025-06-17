# Wav2Vec2-based Speaker Identification on AISHELL-4

This project implements Mandarin speaker identification for AISHELL-4 meeting segments using transfer learning on `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`.  
It features session-level training, mean+std pooling, and an enhanced classifier with BatchNorm.

---

## Installing Dependencies

We recommend Python ≥3.9. If you don’t have it, use [pyenv](https://github.com/pyenv/pyenv) or `conda`.

Create and activate a virtual environment:

```bash
# With venv
$ python3 -m venv venv
$ source venv/bin/activate

# OR with conda
$ conda create -n w2v2speaker python=3.9 -y
$ conda activate w2v2speaker
```

Example requirements.txt content:

```bash
torch>=1.13.0
torchaudio>=0.13.0
transformers
pandas
scikit-learn
pytorch_lightning>=2.0.0
```
If you need specific CUDA versions for PyTorch, follow PyTorch official installation instructions.

## Setting Up AISHELL-4 Data

```bash
(w2v2speaker) $ python prepare_segments_from_rttm.py --rttm_dir /path/to/rttm \
                                                     --pcm_dir /path/to/audio \
                                                     --output_dir data/segments

(w2v2speaker) $ python rename_segments_to_match_csv.py --segments_dir data/segments \
                                                       --csv_path data/data_fixed.csv

```

## Downloading Pre-trained Model

The jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn model is used and will be automatically downloaded via Hugging Face Transformers during training.

If you want to manually download:

```bash
(w2v2speaker) $ transformers-cli download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn

```

## Running the Experiment

Run per-session training:

```bash
(w2v2speaker) $ python train.py

```
## Example Results

| Session                 | # Speakers | Final Val Acc |
| ----------------------- | ---------- | ------------- |
| 20200706\_L\_R001S01C01 | 7          | 0.708         |
| 20200706\_L\_R001S02C01 | 7          | 0.839         |

## Notes

Checkpoints will be saved for each session with best validation accuracy.

You can modify train.py parameters (e.g., batch size, learning rate) as needed.
