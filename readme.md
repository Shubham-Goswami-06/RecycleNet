# ♻️ RecycleNet

RecycleNet is an AI model that classifies waste into categories like **plastic, glass, paper, metal, cardboard, and trash**, and suggests **short recycling tips** powered by an LLM.

## 🚀 Features

- Fine-tuned **MobileNetV2** for waste classification
- Hugging Face **LLM integration** for creative recycling suggestions
- Fallback to locally stored tips if API unavailable
- Works on CPU → easy to host anywhere (like Vercel)

## 📂 Project Structure

- `RecycleNet.py` → main inference script
- `MobileNetV2_best.pth` → trained weights
- `class_map.json` → class label mapping
- `llm_tips.json` → grows with collected tips
- `requirements.txt` → dependencies

## 🛠️ Usage

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token

python RecycleNet.py \
  --image test.jpg \
  --checkpoint MobileNetV2_best.pth \
  --classmap class_map.json
```
