import streamlit as st
import torch
import json
from PIL import Image
import requests
from io import BytesIO

from RecycleNet import load_model, predict, get_recycling_tips

# -------------------------------
# Load class mapping
# -------------------------------
with open("class_map.json", "r") as f:
    mapping = json.load(f)
class_names = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]

# -------------------------------
# Load model (CPU for Streamlit)
# -------------------------------
device = torch.device("cpu")
model = load_model("MobileNetV2.pth", num_classes=len(class_names), device=device)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="RecycleNet", page_icon="♻️", layout="centered")
st.title("♻️ RecycleNet")
st.write("Upload an image or paste a link to identify waste and get recycling tips.\n\nNote that the image should only contain single object.")

# Input: file or URL
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or paste an image URL:")

if uploaded_file or image_url:
    # Load image
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    else:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict
    predicted_class = predict(img, model, class_names, device=device)
    st.success(f"✅ Predicted Class: **{predicted_class}**")

    # Get tips
    with st.spinner("Fetching recycling tips..."):
        tips = get_recycling_tips(predicted_class)
    st.info(f"💡 Recycling Tips:\n{tips}")
