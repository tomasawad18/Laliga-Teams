import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# Cleaner TensorFlow logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

try:
    import cv2
except Exception:
    cv2 = None


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
COMPAT_DIR = MODELS_DIR / "_compatible"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
DEFAULT_IMAGE_SIZE = (224, 224)  # width, height

MODEL_PATHS = {
    "Custom CNN": MODELS_DIR / "best_laliga_logo_cnn.keras",
    "EfficientNetB0": MODELS_DIR / "best_laliga_logo_efficientnet.keras",
    "DenseNet121": MODELS_DIR / "best_laliga_logo_densenet121.h5",
    "MobileNetV3Large": MODELS_DIR / "best_laliga_logo_mobilenetv3_large.keras",
}


def load_class_names() -> List[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json not found at {CLASS_NAMES_PATH}. "
            "Keep class_names.json beside app.py."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ValueError("class_names.json must be a JSON list of strings.")

    return names


def remove_key_recursively(obj, key_to_remove: str) -> int:
    """Remove unsupported Keras config keys from nested dictionaries/lists."""
    removed = 0
    if isinstance(obj, dict):
        if key_to_remove in obj:
            obj.pop(key_to_remove, None)
            removed += 1
        for value in obj.values():
            removed += remove_key_recursively(value, key_to_remove)
    elif isinstance(obj, list):
        for item in obj:
            removed += remove_key_recursively(item, key_to_remove)
    return removed


def fix_depthwiseconv_groups(obj) -> int:
    """
    Some Keras versions save `groups` inside DepthwiseConv2D configs.
    Older TensorFlow/Keras versions do not accept that argument.
    """
    removed = 0
    if isinstance(obj, dict):
        if obj.get("class_name") == "DepthwiseConv2D":
            config = obj.get("config", {})
            if isinstance(config, dict) and "groups" in config:
                config.pop("groups", None)
                removed += 1
        for value in obj.values():
            removed += fix_depthwiseconv_groups(value)
    elif isinstance(obj, list):
        for item in obj:
            removed += fix_depthwiseconv_groups(item)
    return removed


def make_keras_file_compatible(model_path: Path) -> Path:
    """
    Creates a temporary compatible copy of a .keras model if the original contains
    newer Keras config fields that older TensorFlow/Keras builds cannot deserialize.
    The original uploaded model is never modified.
    """
    COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    compatible_path = COMPAT_DIR / model_path.name

    try:
        with zipfile.ZipFile(model_path, "r") as zin:
            if "config.json" not in zin.namelist():
                shutil.copy2(model_path, compatible_path)
                return compatible_path

            config = json.loads(zin.read("config.json").decode("utf-8"))
            changed = 0
            for key in ("quantization_config", "quantization_mode"):
                changed += remove_key_recursively(config, key)
            changed += fix_depthwiseconv_groups(config)

            if changed == 0:
                shutil.copy2(model_path, compatible_path)
                return compatible_path

            with zipfile.ZipFile(compatible_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename == "config.json":
                        zout.writestr(item, json.dumps(config, ensure_ascii=False).encode("utf-8"))
                    else:
                        zout.writestr(item, zin.read(item.filename))

    except zipfile.BadZipFile:
        # H5 and non-zip files land here.
        shutil.copy2(model_path, compatible_path)

    return compatible_path


def get_input_size(model: tf.keras.Model) -> Tuple[int, int]:
    """Return model input size as (width, height)."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]

    if isinstance(shape, tuple) and len(shape) >= 4:
        height, width = shape[1], shape[2]
        if isinstance(height, int) and isinstance(width, int):
            return (width, height)

    return DEFAULT_IMAGE_SIZE


def load_one_model(model_name: str, model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file for {model_name}: {model_path}")

    try:
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as first_error:
        if model_path.suffix.lower() != ".keras":
            raise RuntimeError(
                f"Could not load {model_name} from {model_path}.\n"
                f"Error: {first_error}"
            ) from first_error

        compatible_path = make_keras_file_compatible(model_path)
        try:
            return tf.keras.models.load_model(str(compatible_path), compile=False)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load {model_name}.\n"
                f"Original path: {model_path}\n"
                f"First error: {first_error}\n"
                f"Second error after compatibility fix: {second_error}"
            ) from second_error


def load_models() -> Tuple[Dict[str, tf.keras.Model], Dict[str, Tuple[int, int]]]:
    loaded = {}
    sizes = {}

    for model_name, model_path in MODEL_PATHS.items():
        model = load_one_model(model_name, model_path)
        loaded[model_name] = model
        sizes[model_name] = get_input_size(model)
        print(f"Loaded {model_name} from {model_path} with input size {sizes[model_name][0]}x{sizes[model_name][1]}")

    return loaded, sizes


def remove_logo_background(image_rgb: np.ndarray) -> np.ndarray:
    """Optional GrabCut cleanup. If it fails, safely returns the original image."""
    if cv2 is None:
        return image_rgb

    try:
        h, w = image_rgb.shape[:2]
        if h < 10 or w < 10:
            return image_rgb

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        margin_x = max(1, int(w * 0.05))
        margin_y = max(1, int(h * 0.05))
        rect = (
            margin_x,
            margin_y,
            max(1, w - 2 * margin_x),
            max(1, h - 2 * margin_y),
        )

        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype("uint8")

        foreground_ratio = float(mask2.mean())
        if not (0.05 < foreground_ratio < 0.95):
            return image_rgb

        white = np.ones_like(image_rgb, dtype=np.uint8) * 255
        cleaned = image_rgb * mask2[:, :, None] + white * (1 - mask2[:, :, None])
        return cleaned.astype(np.uint8)

    except Exception:
        return image_rgb


def resize_with_padding(image_rgb: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
    """Resize without stretching and place on a white background."""
    target_w, target_h = target_size
    old_h, old_w = image_rgb.shape[:2]

    scale = min(target_w / old_w, target_h / old_h)
    new_w = max(1, int(old_w * scale))
    new_h = max(1, int(old_h * scale))

    if cv2 is not None:
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return Image.fromarray(canvas)

    resized = Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return canvas


def preprocess_image(image: Image.Image, model_name: str, clean_background: bool) -> Tuple[np.ndarray, Image.Image]:
    if image is None:
        raise ValueError("Please upload an image first.")

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGBA")
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white_bg, image).convert("RGB")

    rgb = np.asarray(image).astype(np.uint8)
    if clean_background:
        rgb = remove_logo_background(rgb)

    target_size = MODEL_INPUT_SIZES.get(model_name, DEFAULT_IMAGE_SIZE)
    processed = resize_with_padding(rgb, target_size)

    # The trained models contain their own preprocessing/rescaling layer.
    arr = np.asarray(processed).astype("float32")
    batch = np.expand_dims(arr, axis=0)
    return batch, processed


def normalize_predictions(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype="float64").reshape(-1)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

    # If the model output is logits instead of probabilities, apply softmax.
    if np.any(preds < 0) or not np.isclose(preds.sum(), 1.0, atol=1e-3):
        exp = np.exp(preds - np.max(preds))
        denom = exp.sum()
        if denom <= 0:
            return np.ones_like(exp) / max(1, len(exp))
        preds = exp / denom

    return preds


def predict_single_model(model_name: str, image: Image.Image, clean_background: bool, top_k: int):
    if image is None:
        return {}, None, "Please upload an image first."

    batch, processed = preprocess_image(image, model_name, clean_background)
    raw = MODELS[model_name].predict(batch, verbose=0)[0]
    preds = normalize_predictions(raw)

    if len(preds) != len(CLASS_NAMES):
        message = (
            f"Model output has {len(preds)} classes, but class_names.json has "
            f"{len(CLASS_NAMES)} labels. Use the exact class order from training."
        )
        return {}, processed, message

    top_k = int(max(1, min(top_k, len(CLASS_NAMES))))
    top_indices = np.argsort(preds)[::-1][:top_k]

    result = {CLASS_NAMES[i]: float(preds[i]) for i in top_indices}
    best_idx = int(top_indices[0])
    input_size = MODEL_INPUT_SIZES.get(model_name, DEFAULT_IMAGE_SIZE)

    summary = (
        f"Best prediction: {CLASS_NAMES[best_idx]}\n"
        f"Confidence: {preds[best_idx] * 100:.2f}%\n"
        f"Model: {model_name}\n"
        f"Input size: {input_size[0]}×{input_size[1]}\n"
        f"Background cleanup: {'ON' if clean_background else 'OFF'}"
    )
    return result, processed, summary


def compare_all_models(image: Image.Image, clean_background: bool, top_k: int):
    if image is None:
        return {}, {}, {}, {}, None, "Please upload an image first."

    outputs = []
    summaries = []
    processed_preview = None

    for model_name in MODEL_PATHS.keys():
        result, processed, summary = predict_single_model(model_name, image, clean_background, top_k)
        outputs.append(result)
        summaries.append(summary)
        if processed_preview is None:
            processed_preview = processed

    return (*outputs, processed_preview, "\n\n".join(summaries))


CLASS_NAMES = load_class_names()
MODELS, MODEL_INPUT_SIZES = load_models()

with gr.Blocks(title="LaLiga Teams Logo Classifier") as demo:
    gr.Markdown(
        "# LaLiga Teams Logo Classifier\n"
        "Upload a football club logo and classify it using a selected Keras model, "
        "or compare all four trained models."
    )

    with gr.Tab("Single Model Prediction"):
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload logo image")
            processed_image = gr.Image(type="pil", label="Processed model input")

        with gr.Row():
            model_choice = gr.Radio(
                choices=list(MODEL_PATHS.keys()),
                value="DenseNet121",
                label="Model",
            )
            clean_background = gr.Checkbox(value=True, label="Clean logo background")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K predictions")

        predict_button = gr.Button("Predict")
        prediction_output = gr.Label(num_top_classes=10, label="Prediction probabilities")
        summary_output = gr.Textbox(label="Summary", lines=6)

        predict_button.click(
            fn=predict_single_model,
            inputs=[model_choice, input_image, clean_background, top_k],
            outputs=[prediction_output, processed_image, summary_output],
            api_name="predict",
        )

    with gr.Tab("Compare All Models"):
        with gr.Row():
            compare_image = gr.Image(type="pil", label="Upload logo image")
            compare_processed_image = gr.Image(type="pil", label="Processed model input preview")

        with gr.Row():
            compare_clean_background = gr.Checkbox(value=True, label="Clean logo background")
            compare_top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K predictions")

        compare_button = gr.Button("Compare All Models")
        with gr.Row():
            cnn_output = gr.Label(num_top_classes=10, label="Custom CNN")
            efficientnet_output = gr.Label(num_top_classes=10, label="EfficientNetB0")
        with gr.Row():
            densenet_output = gr.Label(num_top_classes=10, label="DenseNet121")
            mobilenet_output = gr.Label(num_top_classes=10, label="MobileNetV3Large")

        compare_summary = gr.Textbox(label="Comparison Summary", lines=10)

        compare_button.click(
            fn=compare_all_models,
            inputs=[compare_image, compare_clean_background, compare_top_k],
            outputs=[
                cnn_output,
                efficientnet_output,
                densenet_output,
                mobilenet_output,
                compare_processed_image,
                compare_summary,
            ],
            api_name="compare",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # 0.0.0.0 works locally and on deployment platforms such as Render/Hugging Face Spaces.
    demo.launch(server_name="0.0.0.0", server_port=port)
