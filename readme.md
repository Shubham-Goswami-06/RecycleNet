# ♻️ RecycleNet

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# ♻️ RecycleNet

RecycleNet is an end-to-end AI application that not only classifies waste into categories (plastic, glass, paper, metal, cardboard, and trash) using deep learning, but also acts as an intelligent assistant by generating contextual recycling tips powered by a Large Language Model.

## 🚀 Key Features

- **Advanced Computer Vision**: Fine-tuned MobileNetV2 architecture for highly accurate, lightweight waste classification.
- **LLM Integration**: Seamlessly hooks into Hugging Face's API to generate creative, context-aware recycling suggestions.
- **Resilient Architecture**: Engineered with an automatic fallback to locally stored JSON tips (llm_tips.json) if the external API is unavailable, ensuring zero downtime.
- **Cost-Effective Deployment**: Strictly optimized for CPU inference, making it incredibly easy and cheap to host anywhere (like Vercel or local edge devices).

## 📂 Project Structure

- `RecycleNet.py` → Main inference and API handling script
- `MobileNetV2_best.pth` → Trained neural network weights
- `class_map.json` → Dynamic class label mapping
- `llm_tips.json` → Local database that grows continuously with collected tips
- `requirements.txt` → Project dependencies

## 🛠️ Usage

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token

python RecycleNet.py \
  --image test.jpg \
  --checkpoint MobileNetV2_best.pth \
  --classmap class_map.json
```

## 📜 License

This project is licensed under the MIT License — you’re free to use, modify, and share it with attribution.
