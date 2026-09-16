import streamlit as st
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import os
import gdown

# -------------------------------
# 1. MODEL ARCHITECTURE
# -------------------------------
class AdvancedFusionModel(nn.Module):
    def __init__(self, model_id, common_embed_dim=512, hidden_dim=512):
        super().__init__()

        # ✅ Use full CLIP model (fix for mismatch issue)
        self.clip = CLIPModel.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(common_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        # ✅ Use CLIP embeddings directly
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        fused = (image_embeds + text_embeds) / 2
        return self.classifier(fused)


# -------------------------------
# 2. LOAD MODEL
# -------------------------------
@st.cache_resource
def load_hateful_model_and_processor():
    st.info("Loading model...")

    MODEL_PATH = "best_hateful_meme_model.pth"
    MODEL_ID = "openai/clip-vit-base-patch32"
    FILE_ID = "1u5_9KkYV_D8CQhMeace_vV6h9RHe9cuI"

    # Download if not exists
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    model = AdvancedFusionModel(MODEL_ID).to(device)

    # ✅ FIX: allow partial loading
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    st.success(f"Model loaded on {device}")
    return model, processor, device


# -------------------------------
# 3. EMOTION MODELS
# -------------------------------
@st.cache_resource
def load_text_emotion_model():
    return pipeline("text-classification",
                    model="SamLowe/roberta-base-go_emotions",
                    top_k=3)


@st.cache_resource
def load_image_emotion_model():
    return pipeline("image-classification",
                    model="trpakov/vit-face-expression",
                    top_k=2)


# -------------------------------
# 4. STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Emotion & Hate Detector", layout="wide")

st.title("🧠 Multi-Modal Emotion & Hate Detection")

with st.spinner("Loading models..."):
    model, processor, device = load_hateful_model_and_processor()
    text_model = load_text_emotion_model()
    image_model = load_image_emotion_model()

mode = st.radio(
    "Choose Mode",
    ["Image + Text", "Text Only", "Image Only"]
)

image = None
text = ""

if mode == "Image + Text":
    image = st.file_uploader("Upload Image")
    text = st.text_area("Enter text")

elif mode == "Text Only":
    text = st.text_area("Enter text")

elif mode == "Image Only":
    image = st.file_uploader("Upload Image")

if st.button("Analyze"):

    # -----------------------
    # Image + Text
    # -----------------------
    if mode == "Image + Text":
        if image and text:
            img = Image.open(image).convert("RGB")

            inputs = processor(
                text=text,
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)

            prob = torch.sigmoid(output).item()
            pred = "Hateful" if prob > 0.5 else "Not Hateful"

            st.image(img)
            st.write(f"Prediction: {pred} ({prob:.2f})")

    # -----------------------
    # Text Only
    # -----------------------
    elif mode == "Text Only":
        if text:
            res = text_model(text)
            st.json(res)

    # -----------------------
    # Image Only
    # -----------------------
    elif mode == "Image Only":
        if image:
            img = Image.open(image).convert("RGB")
            st.image(img)
            res = image_model(img)
            st.json(res)
