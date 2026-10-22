# Multi-Input Handwritten Digit Classifier

A dual-input PyTorch image-classification prototype. It uses scikit-learn's
bundled 8×8 handwritten **digit** dataset (classes 0–9), a compact CNN for
pixel input, and a second branch for simple image statistics. A logistic
regression baseline makes the value of learned image/metadata fusion
measurable. This small benchmark is an OCR learning exercise, not a production
document OCR system: it does not recognize letters, process scanned pages,
segment text, or handle real-world handwriting variation.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

The dataset is included in scikit-learn, so no dataset download is needed. A
stratified train/validation/test split is used; feature-scaler statistics are
fit on training data only. The CNN selects its checkpoint using validation
loss, and final comparisons use the held-out test set. Training history,
classification metrics, confusion matrices, example predictions, and model
weights are written to `outputs/`.

Change the training budget with `python train.py --epochs 60 --patience 8`.
Reported scores are generated at runtime and must not be replaced with an
unverified resume claim.

See [`docs/MODEL_CARD.md`](./docs/MODEL_CARD.md) for the intended use,
architecture, evaluation artifacts, and OCR limitations.

After training, classify one centered single-digit image with
`python predict.py path\to\digit.png --top-k 3`. Preprocessing assumes a
high-contrast digit on a plain background and resizes it to 8×8; it is not a
document text detector or full-page OCR pipeline.

The CI workflow runs the model/image unit checks and a short smoke-training
pass; this smoke run is not the benchmark result reported in a portfolio.
