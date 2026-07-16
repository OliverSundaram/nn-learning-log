import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---- 1. Class order (must match your ImageFolder alphabetical order) ----
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---- 2. Device: use GPU if available, otherwise fall back to CPU ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 3. Rebuild the exact architecture used in training ----
def build_model():
    model = models.resnet18(weights=None)  # weights=None: we're loading OUR weights, not ImageNet's
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 output classes
    return model

# ---- 4. Load weights into that architecture ----
model = build_model()
state_dict = torch.load("brain_tumor_model.pth", map_location=device)  # map_location handles GPU->CPU
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---- 5. Preprocessing pipeline (must match training exactly) ----
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# ---- 6. The actual prediction function ----
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs, dim=0).item()
    cls_pred = CLASS_NAMES[pred]
    confidence = probs[pred].item()

    all_scores = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(4)}
    return cls_pred, confidence, all_scores



if __name__ == "__main__":
    test_image_path = "data/test/pituitary/Te-pi_16.jpg"
    predicted_class, confidence, all_scores = predict(test_image_path)
    print(f"Predicted: {predicted_class} ({confidence:.2%} confidence)")
    print(f"All class scores: {all_scores}")