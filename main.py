from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image as PILImage, ImageSequence
import numpy as np
import onnxruntime as ort

# -----------------------
# Constants
# -----------------------
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR2IDX = {c: i for i, c in enumerate(CHARSET)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHARSET)
NUM_POS = 5
EXPECTED_SIZE = (224, 224)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "assets", "holako_bag.onnx")


# -----------------------
# Load ONNX model once
# -----------------------
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Captcha Recognition API",
    description="Upload an image and get the predicted CAPTCHA text.",
    version="1.0"
)

# Allow CORS for all origins (customize in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"]
)


def preprocess_image(pil: PILImage.Image) -> np.ndarray:
    # Resize and normalize
    img = pil.resize(EXPECTED_SIZE).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    # HWC to CHW
    x = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return x


def otsu_binarize(frames: list) -> PILImage.Image:
    # Median background
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    gray = (0.2989 * bg[..., 0] + 0.5870 * bg[..., 1] + 0.1140 * bg[..., 2]).astype(np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_tot = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    thresh = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_tot - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > max_var:
            max_var = varBetween
            thresh = i
    binary = PILImage.fromarray(gray, 'L').point(lambda p: 255 if p > thresh else 0)
    return binary


@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    # Read image bytes
    data = await file.read()
    try:
        pil = PILImage.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Extract frames
    frames = []
    try:
        for frame in ImageSequence.Iterator(pil.convert('RGB')):
            frames.append(np.array(frame, dtype=np.uint8))
    except Exception:
        frames = [np.array(pil.convert('RGB'), dtype=np.uint8)]

    # Binarize
    binary = otsu_binarize(frames)

    # Preprocess for model
    x = preprocess_image(binary)

    # Run ONNX inference
    ort_outs = session.run(None, {'input': x})[0]
    ort_outs = ort_outs.reshape(1, NUM_POS, NUM_CLASSES)
    idxs = np.argmax(ort_outs, axis=2)[0]
    pred = ''.join(IDX2CHAR[i] for i in idxs)

    return {"prediction": pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
