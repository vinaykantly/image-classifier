"""Image classification with a pretrained torchvision model (ImageNet, 1000 classes)."""
import os
import threading

import torch
from torchvision import models

MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v2").lower()
# Below this top-1 probability the app refuses to name the image.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
TOP_K = 5
# Model produced by train.py. If this file exists it replaces the ImageNet model.
CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH", "models/custom_model.pt")

_REGISTRY = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}

VEHICLE_WORDS = (
    "car", "cab", "truck", "van", "bus", "jeep", "limousine", "convertible", "minivan",
    "pickup", "trailer", "tractor", "bike", "bicycle", "moped", "scooter", "motor",
    "train", "locomotive", "tram", "trolley", "streetcar", "airliner", "aircraft", "warplane",
    "airship", "balloon", "boat", "ship", "liner", "canoe", "kayak", "yawl", "catamaran",
    "trimaran", "gondola", "speedboat", "submarine", "schooner", "fireboat", "lifeboat",
    "container ship", "wagon", "snowmobile", "sports car", "racer", "taxi", "ambulance",
    "fire engine", "garbage truck", "tow truck", "forklift", "go-kart", "unicycle", "space shuttle",
)


def group_of(index: int, label: str) -> str:
    """Rough ImageNet category group, derived from the class index / label."""
    if index <= 397:
        return "animal"
    if 924 <= index <= 969:
        return "food"
    if 970 <= index <= 980:
        return "landscape or scene"
    if 981 <= index <= 983:
        return "person"
    if 984 <= index <= 998:
        return "plant or fungus"
    words = label.lower()
    if any(w in words for w in VEHICLE_WORDS):
        return "vehicle"
    return "object"


class Classifier:
    def __init__(self, name: str = MODEL_NAME):
        if name not in _REGISTRY:
            raise ValueError(f"Unknown MODEL_NAME '{name}'. Use one of: {', '.join(_REGISTRY)}")
        self.name = name
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._labels = None
        self._custom = False

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            if os.path.exists(CUSTOM_MODEL_PATH):
                ckpt = torch.load(CUSTOM_MODEL_PATH, map_location="cpu", weights_only=True)
                model = models.mobilenet_v2(weights=None)
                model.classifier[1] = torch.nn.Linear(model.last_channel, len(ckpt["classes"]))
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._preprocess = models.MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
                self._labels = ckpt["classes"]
                self._custom = True
                self._model = model
                return
            builder, weights = _REGISTRY[self.name]
            model = builder(weights=weights)  # downloads weights on first run, then cached
            model.eval()
            self._preprocess = weights.transforms()
            self._labels = weights.meta["categories"]
            self._model = model

    def predict(self, image):
        """image: RGB PIL image -> list of top-K predictions."""
        self.load()
        batch = self._preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self._model(batch)[0], dim=0)
        values, indices = probs.topk(min(TOP_K, len(self._labels)))
        results = []
        for p, i in zip(values.tolist(), indices.tolist()):
            label = self._labels[i]
            results.append({
                "label": label,
                "confidence": round(p * 100, 2),
                "group": "custom category" if self._custom else group_of(i, label),
            })
        return results


def explain(results, threshold=CONFIDENCE_THRESHOLD):
    """Build a plain-language explanation from the model output."""
    top = results[0]
    conf = top["confidence"]
    if conf / 100 < threshold:
        return {
            "recognized": False,
            "message": (
                "The model is not confident about this image "
                f"(best guess scored only {conf:.1f}%). It may show something outside the "
                "1,000 categories the model knows, or the image may be unclear. "
                "Try a sharper photo where the main subject fills more of the frame."
            ),
        }
    level = "very likely" if conf >= 70 else "probably" if conf >= 45 else "possibly"
    text = f"This image {level} shows: {top['label']} (category: {top['group']})."
    others = [r["label"] for r in results[1:3] if r["confidence"] >= 5]
    if others:
        text += " The model also considered: " + ", ".join(others) + "."
    if top["group"] == "person":
        text += " The model has no general 'person' class, so this label describes a scene or role."
    return {"recognized": True, "message": text}
