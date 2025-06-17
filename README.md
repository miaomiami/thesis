# Wav2Vec2-based Speaker Identification on AISHELL-4

This project implements Mandarin speaker identification for AISHELL-4 meeting segments using transfer learning on `wav2vec2-large-chinese`.  
It features session-wise training, mean+std+max pooling, and an enhanced classifier with BatchNorm.

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

