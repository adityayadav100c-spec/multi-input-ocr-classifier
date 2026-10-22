# Multi-Input Character Classifier

A dual-input PyTorch demo classifying handwritten digits from scikit-learn's bundled 8×8 Digits dataset. One branch learns from the image pixels; a second branch consumes simple image statistics (mean, standard deviation, and upper/lower half means). The branch features are fused for classification.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

The dataset is included in scikit-learn, so no dataset download is needed. Training uses a fixed random seed and stratified train/test split. The reported test accuracy is measured at runtime and should not be replaced with an unverified resume claim. Results are saved to `outputs/`.
