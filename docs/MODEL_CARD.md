# Model Card

## Intended use

Teaching example of multimodal classification and train/validation/test
evaluation on scikit-learn's built-in 8×8 grayscale handwritten digits.
Classes are digits 0 through 9.

## Architecture

- Image branch: two convolution layers, pooling, and a learned dense feature
  representation.
- Auxiliary branch: four image statistics (mean, standard deviation, upper
  half mean, lower half mean) passed through a dense layer.
- Fusion: concatenate branch representations and predict 10 digit logits.
- Baseline: logistic regression over flattened pixel values only.

The scaler for auxiliary features is fitted only on the training split. The
CNN checkpoint is selected by validation cross-entropy loss with early
stopping; the held-out test split is evaluated after selection.

## Evaluation artifacts

`outputs/metrics.json` contains split sizes, per-model accuracy, macro F1,
per-class reports, confusion matrices, and limitations. Loss history, test
predictions, a loss plot, confusion-matrix plot, example predictions, and model
checkpoint are also written there.

After training, `python predict.py path\to\digit.png` runs one image through
the saved checkpoint and training-only feature scaling. The input path must be
a single digit on a plain, high-contrast background; it is not a page OCR tool.

## Limitations

This dataset does not contain letters, complete documents, diverse scanners,
natural backgrounds, skew, blur, handwriting variation, or realistic OCR
segmentation. A good score on this benchmark is not a claim about production
OCR accuracy. Do not use this model for document processing or high-impact
decisions.
