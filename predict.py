import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import MultiInputCNN

ROOT = Path(__file__).parent


def prepare_digit_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        gray = image_file.convert("L")
    pixels = np.asarray(gray, dtype=np.uint8)
    corner_values = pixels[[0, -1]][:, [0, -1]]
    background_is_light = float(np.median(corner_values)) >= 127
    ink = 255 - pixels if background_is_light else pixels
    rows, cols = np.where(ink > 32)
    if len(rows) == 0:
        raise ValueError("No visible digit strokes found in the image.")

    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    cropped = Image.fromarray(ink[top:bottom, left:right])
    width, height = cropped.size
    side = max(width, height)
    padded_side = max(side + 2, int(np.ceil(side * 1.25)))
    canvas = Image.new("L", (padded_side, padded_side), 0)
    canvas.paste(cropped, ((padded_side - width) // 2, (padded_side - height) // 2))
    resized = canvas.resize((8, 8), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def auxiliary_features(image: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[
            image.mean(),
            image.std(),
            image[:4, :].mean(),
            image[4:, :].mean(),
        ]],
        dtype=np.float32,
    )


def predict(image_path: Path, top_k: int = 3) -> list[tuple[int, float]]:
    if top_k < 1 or top_k > 10:
        raise ValueError("top-k must be between 1 and 10.")
    output = ROOT / "outputs"
    model_path = output / "multi_input_cnn.pt"
    preprocessing_path = output / "preprocessing.json"
    if not model_path.is_file() or not preprocessing_path.is_file():
        raise FileNotFoundError(
            "Train the model first with `python train.py`; required outputs are missing."
        )

    image = prepare_digit_image(image_path)
    stats = auxiliary_features(image)
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    scale = np.asarray(preprocessing["feature_scale"], dtype=np.float32)
    offset = np.asarray(preprocessing["feature_offset"], dtype=np.float32)
    scaled_stats = stats * scale + offset

    model = MultiInputCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(image[None, None, :, :], dtype=torch.float32),
            torch.tensor(scaled_stats, dtype=torch.float32),
        )
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
    indices = np.argsort(probabilities)[-top_k:][::-1]
    return [(int(index), float(probabilities[index])) for index in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Classify one centered, single-digit grayscale image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    for digit, probability in predict(args.image, top_k=args.top_k):
        print(f"digit={digit} probability={probability:.4f}")
    print("Prediction is a demo estimate, not an OCR guarantee.")


if __name__ == "__main__":
    main()
