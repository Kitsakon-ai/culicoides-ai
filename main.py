import base64
import io
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2


app = FastAPI(title="Sandfly AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ตอน deploy ค่อยเปลี่ยนเป็น domain frontend จริง
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES: List[str] = ["guttifer", "peregrinus"]
SPECIES_TO_GENUS = {
    "guttifer": "Culicoides",
    "peregrinus": "Culicoides",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class ExplainRequest(BaseModel):
    species: str
    confidence: float
    topK: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    message: str
    prediction: Optional[Dict[str, Any]] = None
    ai_model: str = Field(default="gemini", alias="aiModel")

    class Config:
        populate_by_name = True


def build_taxonomy(species: str) -> Dict[str, str]:
    return {
        "kingdom": "Animalia",
        "phylum": "Arthropoda",
        "class": "Insecta",
        "order": "Diptera",
        "family": "Ceratopogonidae",
        "genus": SPECIES_TO_GENUS.get(species, "Unknown"),
        "species": species,
    }


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.50:
        return "low"
    return "ood"


def normalize_model_name(name: str) -> str:
    name = (name or "").strip().lower()
    aliases = {
        "efficientnet": "efficientnet",
        "efficientnet_b0": "efficientnet",
        "effb0": "efficientnet",
        "resnet": "resnet",
        "resnet50": "resnet",
        "densenet": "densenet",
        "densenet121": "densenet",
    }
    return aliases.get(name, name)


def create_efficientnet_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def create_resnet_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_densenet_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_model(path: str, model_type: str):
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    if model_type == "efficientnet":
        loaded_model = create_efficientnet_model(len(CLASS_NAMES))
    elif model_type == "resnet":
        loaded_model = create_resnet_model(len(CLASS_NAMES))
    elif model_type == "densenet":
        loaded_model = create_densenet_model(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dict = torch.load(full_path, map_location=DEVICE)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


MODELS: Dict[str, nn.Module] = {}

try:
    efficientnet_model = load_model("EfficientNet_B0.pth", "efficientnet")
    MODELS["efficientnet"] = efficientnet_model
    MODELS["efficientnet_b0"] = efficientnet_model
except Exception as e:
    print(f"[WARN] Failed to load EfficientNet_B0.pth: {e}")

try:
    resnet_model = load_model("ResNet50_best.pth", "resnet")
    MODELS["resnet"] = resnet_model
    MODELS["resnet50"] = resnet_model
except Exception as e:
    print(f"[WARN] Failed to load ResNet50_best.pth: {e}")

try:
    densenet_model = load_model("DenseNet121_best.pth", "densenet")
    MODELS["densenet"] = densenet_model
    MODELS["densenet121"] = densenet_model
except Exception as e:
    print(f"[WARN] Failed to load DenseNet121_best.pth: {e}")

if not MODELS:
    raise RuntimeError("No model loaded successfully. Please check your model files.")


def read_image_from_upload(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def pil_to_input_tensor(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(DEVICE)


def predict_tensor(active_model: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        logits = active_model(x)
        probs = F.softmax(logits, dim=1)[0]
    return logits, probs


def build_prediction_response(probs: torch.Tensor) -> Dict[str, Any]:
    probs_list = probs.detach().cpu().tolist()
    best_idx = int(torch.argmax(probs).item())
    species = CLASS_NAMES[best_idx]
    conf = float(probs[best_idx].item())

    top_k = [
        {"name": CLASS_NAMES[i], "probability": float(probs_list[i])}
        for i in range(len(CLASS_NAMES))
    ]
    top_k.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "species": species,
        "genus": SPECIES_TO_GENUS.get(species, "Unknown"),
        "confidence": conf,
        "topK": top_k,
        "confidenceLevel": confidence_level(conf),
        "taxonomy": build_taxonomy(species),
    }


def generate_explanation_text(species: str, confidence: float, top_k: List[Dict[str, Any]]) -> str:
    conf_percent = round(confidence * 100, 2)
    runner_up = top_k[1]["name"] if len(top_k) > 1 else "unknown"

    return (
        f"โมเดลทำนายว่าเป็น {species} ด้วยความเชื่อมั่น {conf_percent}% "
        f"โดยเปรียบเทียบกับตัวเลือกอื่น เช่น {runner_up} "
        f"การตัดสินใจของโมเดลอาศัยลักษณะเชิงภาพจากบริเวณปีกและโครงสร้างโดยรวมของตัวอย่าง "
        f"ผลนี้ควรใช้ร่วมกับการตรวจสอบโดยผู้เชี่ยวชาญ โดยเฉพาะเมื่อค่าความเชื่อมั่นไม่สูงมาก"
    )


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def make_gradcam(image: Image.Image, active_model: nn.Module, model_name: str):
    model_name = normalize_model_name(model_name)

    if model_name == "efficientnet":
        target_layer = active_model.features[-3]
    elif model_name == "resnet":
        target_layer = active_model.layer4[-1]
    elif model_name == "densenet":
        target_layer = active_model.features.denseblock4
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    x = pil_to_input_tensor(image)
    logits = active_model(x)
    class_idx = int(torch.argmax(logits, dim=1).item())

    active_model.zero_grad()
    logits[0, class_idx].backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks failed.")

    act = activations[0][0]
    grad = gradients[0][0]

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(dim=0)

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.cpu().numpy()
    cam_np = cv2.resize(cam_np, (224, 224))
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

    original = image.resize((224, 224))
    original_np = np.array(original).astype(np.float32) / 255.0

    fig, ax = plt.subplots()
    ax.imshow(original_np)
    ax.imshow(cam_np, cmap="jet", alpha=0.6)
    ax.axis("off")

    return fig_to_base64(fig), class_idx


@app.get("/")
def root():
    return {
        "message": "Sandfly API running",
        "available_models": list(MODELS.keys()),
        "normalized_models": ["efficientnet", "resnet", "densenet"],
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    ml_model = normalize_model_name(ml_model)

    active_model = MODELS.get(ml_model)
    if active_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml_model: {ml_model}. Available: {list(MODELS.keys())}"
        )

    content = await file.read()
    image = read_image_from_upload(content)

    try:
        x = pil_to_input_tensor(image)
        _, probs = predict_tensor(active_model, x)
        result = build_prediction_response(probs)

        return {
            **result,
            "modelUsed": ml_model,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}") from exc


@app.post("/predict-with-gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    ml_model = normalize_model_name(ml_model)

    active_model = MODELS.get(ml_model)
    if active_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml_model: {ml_model}. Available: {list(MODELS.keys())}"
        )

    content = await file.read()
    image = read_image_from_upload(content)

    try:
        x = pil_to_input_tensor(image)
        _, probs = predict_tensor(active_model, x)
        result = build_prediction_response(probs)

        gradcam_image, _ = make_gradcam(image, active_model, ml_model)
        explanation = generate_explanation_text(
            species=result["species"],
            confidence=result["confidence"],
            top_k=result["topK"],
        )

        return {
            **result,
            "gradcam": gradcam_image,
            "explanation": explanation,
            "modelUsed": ml_model,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference with Grad-CAM failed: {str(exc)}"
        ) from exc


@app.post("/explain")
def explain(req: ExplainRequest):
    text = generate_explanation_text(
        species=req.species,
        confidence=req.confidence,
        top_k=req.topK,
    )
    return {"explanation": text}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip().lower()
    pred = req.prediction or {}
    species = pred.get("species", "unknown")
    confidence = pred.get("confidence", 0)
    model_used = pred.get("modelUsed", "unknown")
    ai_model_used = req.ai_model

    if "confidence" in msg or "มั่นใจ" in msg:
        answer = (
            f"โมเดล {model_used} มีความเชื่อมั่นประมาณ "
            f"{round(confidence * 100, 2)}% สำหรับชนิด {species}"
        )
    elif "species" in msg or "ชนิด" in msg or "อะไร" in msg:
        answer = f"ผลที่ทำนายได้คือ {species} โดยใช้โมเดล {model_used}"
    elif "grad-cam" in msg or "heatmap" in msg or "อธิบาย" in msg:
        answer = (
            "Grad-CAM ใช้เพื่อแสดงบริเวณของภาพที่มีอิทธิพลต่อการตัดสินใจของโมเดล "
            "สำหรับตัวอย่างนี้ โมเดลให้ความสำคัญกับลักษณะเชิงรูปแบบของปีกเป็นหลัก"
        )
    else:
        answer = (
            f"จากผลล่าสุด โมเดล {model_used} ทำนายว่าเป็น {species} "
            f"ด้วยความเชื่อมั่น {round(confidence * 100, 2)}% "
            f"และระบบแชทกำลังใช้ {ai_model_used} "
            "คุณถามต่อได้เรื่องความมั่นใจ, ชนิด, โมเดลที่ใช้, หรือการตีความ Grad-CAM"
        )

    return {"answer": answer}