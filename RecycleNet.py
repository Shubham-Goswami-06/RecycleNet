import os
import json
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
from huggingface_hub import InferenceClient
import re

# -----------------------------------
# Paths
# -----------------------------------
TIPS_FILE = "llm_tips.json"

# -----------------------------------
# Model Definition
# -----------------------------------
def get_mobilenet_v2(num_classes=6):
    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# -----------------------------------
# Load Model Checkpoint
# -----------------------------------
def load_model(checkpoint_path, num_classes=6, device="cpu"):
    model = get_mobilenet_v2(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -----------------------------------
# Preprocessing / Augmentations
# -----------------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # convert any input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -----------------------------------
# Prediction
# -----------------------------------
def predict(image_path, model, class_names, device="cpu"):
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]


# -----------------------------------
# Save New LLM Response
# -----------------------------------
def save_llm_tip(predicted_class, response, tips_file=TIPS_FILE):
    tips_data = {}
    if os.path.exists(tips_file):
        with open(tips_file, "r") as f:
            tips_data = json.load(f)

    tips_list = tips_data.get(predicted_class, [])
    if response not in tips_list:   # avoid duplicates
        tips_list.append(response)
    tips_data[predicted_class] = tips_list

    with open(tips_file, "w") as f:
        json.dump(tips_data, f, indent=4)


# -----------------------------------
# Fallback: Get Stored Tips
# -----------------------------------
def get_fallback_tips(predicted_class, tips_file=TIPS_FILE):
    if os.path.exists(tips_file):
        with open(tips_file, "r") as f:
            tips_data = json.load(f)
        tips_list = tips_data.get(predicted_class, [])
        if tips_list:
            return random.choice(tips_list)
    return "Recycle responsibly and reduce waste."


# -----------------------------------
# Hugging Face LLM Call (Nebius)
# -----------------------------------
def clean_llm_output(text: str) -> str:
    """Remove <think>...</think> blocks if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def get_recycling_tips(predicted_class, model_name="Qwen/Qwen3-4B"):
    token = os.getenv("HF_TOKEN")
    if not token:
        return get_fallback_tips(predicted_class)

    try:
        client = InferenceClient(provider="nebius", api_key=token)
        prompt = f"Give me 4 (max 10 - 15 words each) tips to recycle {predicted_class} type waste. and Respond in numbered list."

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,   # ✅ simpler, one-shot response
        )

        # Extract content
        tips = response.choices[0].message.content if response.choices else ""
        tips = clean_llm_output(tips)

        if tips:
            save_llm_tip(predicted_class, tips)

        return tips or get_fallback_tips(predicted_class)

    except Exception:
        return f"⚠️ Using fallback tips.\n" + get_fallback_tips(predicted_class)


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV2 Inference + Recycling Tips")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--classmap", type=str, required=True, help="Path to JSON file with class mapping")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load class mapping
    with open(args.classmap, "r") as f:
        mapping = json.load(f)
    class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]

    # Load model
    model = load_model(args.checkpoint, num_classes=len(class_names), device=device)

    # Predict class
    predicted_class = predict(args.image, model, class_names, device=device)
    print(f"✅ Prediction: {predicted_class}")

    # Get recycling tips (LLM or fallback)
    tips = get_recycling_tips(predicted_class)
    print(f"\n💡 Recycling Tips:\n{tips}")
