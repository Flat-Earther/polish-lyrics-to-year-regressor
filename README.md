# NLP Project — Predicting Song Release Year from Lyrics

This project explores whether the release year of a Polish song can be predicted from its lyrics alone.  
It uses a **HerBERT** backbone and compares three different ways of handling long lyrics:

- **Random chunk**
- **First + last chunk**
- **Hierarchical chunk encoder**

The notebook also includes exploratory analysis, uncertainty estimation, and simple explainability tools.

---

## Model Weights

Available on Hugging Face:
https://huggingface.co/Flat-Earther/polish-lyrics-year-regressor

---

## Project Goal

Given a song lyric, predict the approximate year it was released.

This is formulated as a **regression** problem rather than classification, so the model predicts a continuous year value.

---

## Dataset

The notebook uses a Polish subset of the **Genius song lyrics with language information** dataset.

It's avaible on hugging face: 
https://huggingface.co/datasets/Flat-Earther/piosenki

Main filtering steps:

- keep only Polish lyrics
- remove rows with missing lyrics or year
- drop the `misc` tag

The notebook saves the processed dataset as `piosenki.csv`.

---

## Preprocessing

The text preprocessing pipeline:

- lowercases text for uncased models
- removes bracketed annotations like `[chorus]`
- removes stutters
- removes punctuation
- keeps Polish characters

Example:

```python
def preprocess(lyrics):
    lyrics = str(lyrics)
    if "uncased" in MODEL_NAME:
        lyrics = lyrics.lower()
    lyrics = re.sub(r"\[.*?\]", "", lyrics)
    lyrics = re.sub(r"(([a-z])-)+\1", "", lyrics)
    lyrics = re.sub(r"[,-]", "", lyrics)
    lyrics = re.sub(r"[^a-zA-ZąćęńśłóżźĄĆĘŃŚŁÓŻŹ ]", " ", lyrics)
    return lyrics.strip()
```

---

## Target Setup

The target year is standardized before training:

- subtract the mean year
- divide by the standard deviation

This helps the regression model train more stably.

---

## Model Backbone

The notebook uses:

- **`allegro/herbert-base-cased`**

You can also switch to:

- `dkleczek/bert-base-polish-uncased-v1`

---

## Chunking Strategies

Because song lyrics can be longer than the model input length, the notebook evaluates three strategies.

### 1. Random Chunk
A random chunk is selected each time a lyric is seen during training.  
At inference time, multiple random chunks are sampled and averaged.

### 2. First + Last Chunk
The model encodes:

- the first chunk
- the last chunk

Then concatenates both representations before regression.

### 3. Hierarchical Encoder
Each chunk is encoded separately, then chunk embeddings are passed through a small Transformer encoder.

---

## Training Details

Typical settings used in the notebook:

- `MAX_LEN = 256`
- `MAX_CHUNKS = 8`
- `BATCH_SIZE = 8`
- `EPOCHS = 4`
- `LR = 2e-5`
- `SEED = 42`

Training setup:

- optimizer: **AdamW**
- loss: **HuberLoss**
- scheduler: **linear warmup**
- gradient clipping: **1.0**

---

## Evaluation

The notebook evaluates models using **mean absolute error (MAE)** in years.

Validation results recorded in the notebook:

| Model | MAE (years) | Total time (s) | ms/sample |
|---|---:|---:|---:|
| First + last | 3.08 | 326.38 | 51.77 |
| Random chunk | 3.10 | 4054.19 | 643.11 |
| Hierarchical | 4.13 | 511.11 | 511.11 |

The **first + last** strategy performed best in this run.

---

## Analysis and Explainability

The notebook includes several analysis tools:

- finding **confident but wrong** predictions
- checking error vs. uncertainty
- plotting error distributions by decade
- token-level occlusion explanations
- global token effect aggregation

This helps inspect whether the model is learning useful lyrical patterns or just memorizing biases.

---

## Example Inference

The notebook includes example predictions such as:

- random chunk inference with uncertainty
- first + last chunk prediction
- testing on custom Polish lyrics

---

## Files Produced by the Notebook

The notebook saves or generates:

- `random_chunk_herbert.pt`
- `first_last_herbert.pt`
- `hierarchical_herbert.pt`
- `suspicious_examples.csv`
- `all_prediction_scores.csv`

It also reads an example input file:

- `piosenki_zle.csv`

---

## Requirements

Typical packages used:

- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `safetensors`

---

## How to Run

1. Install dependencies.
2. Prepare the dataset.
3. Run the notebook top to bottom.
4. Train one or more chunking variants.
5. Evaluate the model on the validation split.

---

## Project Structure

```text
NLP.ipynb
├── data loading
├── preprocessing
├── target standardization
├── tokenizer and chunking
├── model definitions
├── training and evaluation
├── analysis and xAI
└── sample testing
```

---

## Notes

- The project is focused on **Polish lyrics**.
- It is a regression task, so predictions are continuous years rather than discrete classes.
- The hierarchical model is more expressive, but in this notebook run it was not the best performer.
