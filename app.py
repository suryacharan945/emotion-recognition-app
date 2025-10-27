import streamlit as st
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel, pipeline
from PIL import Image
import os
import gdown  # ✅ NEW: to download model dynamically

# --- 1. Define the Hateful Meme Model Architecture ---
class AdvancedFusionModel(nn.Module):
    def __init__(self, text_model_id, vision_model_id, common_embed_dim=512, hidden_dim=512):
        super(AdvancedFusionModel, self).__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_id)
        self.text_model = CLIPTextModel.from_pretrained(text_model_id)
        self.vision_projection = nn.Linear(self.vision_model.config.hidden_size, common_embed_dim)
        self.text_projection = nn.Linear(self.text_model.config.hidden_size, common_embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=common_embed_dim,
                                                     num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(common_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.pooler_output
        image_proj = self.vision_projection(image_embeds)
        text_proj = self.text_projection(text_embeds)
        image_proj_seq = image_proj.unsqueeze(1)
        text_proj_seq = text_proj.unsqueeze(1)
        attn_output, _ = self.cross_attention(query=text_proj_seq, key=image_proj_seq,
                                              value=image_proj_seq)
        fused_features = attn_output.squeeze(1)
        return self.classifier(fused_features)

# --- 2. Load Models (Cached) ---

@st.cache_resource
def load_hateful_model_and_processor():
    st.info("Loading Hateful Meme model (AdvancedFusionModel)...")
    MODEL_PATH = "./best_hateful_meme_model.pth"
    MODEL_ID = "openai/clip-vit-base-patch32"
    FILE_ID = "1u5_9KkYV_D8CQhMeace_vV6h9RHe9cuI"  # ✅ your Google Drive file ID

    # --- If model not found locally, download it ---
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found locally. Downloading from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully from Google Drive!")
        except Exception as e:
            st.error(f"❌ Failed to download model from Google Drive. Error: {e}")
            st.stop()

    # --- Load model + processor ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = AdvancedFusionModel(
        text_model_id=MODEL_ID,
        vision_model_id=MODEL_ID,
        common_embed_dim=512,
        hidden_dim=512
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        st.error(f"Failed to load state_dict for Hateful Model. Error: {e}")
        st.stop()

    model.eval()
    st.success(f"✅ Hateful Meme model loaded! (Running on {device})")
    return model, processor, device

# --- Emotion models (unchanged) ---

@st.cache_resource
def load_text_emotion_model():
    st.info("Loading Text Emotion model (roberta-base-go_emotions)...")
    classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)
    st.success("✅ Text Emotion model loaded.")
    return classifier

@st.cache_resource
def load_image_emotion_model():
    st.info("Loading Image Emotion model (vit-face-expression)...")
    classifier = pipeline("image-classification", model="trpakov/vit-face-expression", top_k=2)
    st.success("✅ Image Emotion model loaded.")
    return classifier

# --- Helper Function ---
def get_interpretive_analysis(prediction, text_results, image_results):
    try:
        top_text_emotion = text_results[0][0]['label']
        top_image_emotion = image_results[0]['label']

        if prediction == "Hateful":
            analysis = (
                f"The model flagged this as **Hateful** due to conflicting emotional signals.\n\n"
                f"* Text emotion: **'{top_text_emotion}'**\n"
                f"* Image emotion: **'{top_image_emotion}'**\n\n"
                f"The dissonance suggests sarcasm or hostility — common markers of hate speech."
            )
        else:
            analysis = (
                f"Classified as **Not Hateful** — likely coherent or neutral content.\n\n"
                f"* Text emotion: **'{top_text_emotion}'**\n"
                f"* Image emotion: **'{top_image_emotion}'**\n\n"
                f"The emotions appear aligned, implying harmless context."
            )
        return analysis
    except Exception as e:
        return f"Could not generate interpretation. Error: {e}"

# --- 4. Streamlit UI ---
st.set_page_config(
    page_title="Multi-Modal Emotion & Hate Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Multi-Modal Analysis: Emotion & Hate Detection")
st.write("Upload an image, provide text, or both. This app uses three distinct AI models to analyze content.")

with st.spinner("Warming up AI models..."):
    hateful_model, clip_processor, device = load_hateful_model_and_processor()
    text_emotion_pipeline = load_text_emotion_model()
    image_emotion_pipeline = load_image_emotion_model()

# --- 5. UI Layout ---
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.header("Input")
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ("Image & Text (Hatefulness)", "Text Only (Emotion)", "Image Only (Emotion)"),
        horizontal=True,
        key="analysis_mode"
    )

    uploaded_image, meme_text = None, ""
    if analysis_mode == "Image & Text (Hatefulness)":
        st.info("Upload an image and provide text.")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        meme_text = st.text_area("Enter meme text", "", key="text_full")
    elif analysis_mode == "Text Only (Emotion)":
        st.info("Enter text for emotion analysis.")
        meme_text = st.text_area("Enter text", "", key="text_only")
    elif analysis_mode == "Image Only (Emotion)":
        st.info("Upload an image for facial emotion analysis.")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

with col2:
    st.header("Analysis Results")

    if analyze_button:
        # --- Mode 1: Image + Text ---
        if analysis_mode == "Image & Text (Hatefulness)":
            if uploaded_image and meme_text:
                try:
                    image = Image.open(uploaded_image).convert("RGB")
                    tab1, tab2, tab3 = st.tabs(["Main Prediction", "Detailed Emotions", "Interpretation"])

                    with tab1:
                        st.subheader("Hatefulness Prediction")
                        with st.spinner("Running main model..."):
                            inputs = clip_processor(
                                text=meme_text, images=image, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=77
                            )
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = hateful_model(
                                    input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    pixel_values=inputs['pixel_values']
                                )
                            prob = torch.sigmoid(outputs).squeeze().item()
                            prediction = "Hateful" if prob > 0.5 else "Not Hateful"
                            confidence = prob if prediction == "Hateful" else 1 - prob

                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        if prediction == "Hateful":
                            st.error(f"**Result: {prediction}** (Confidence: {confidence*100:.2f}%) 😠")
                        else:
                            st.success(f"**Result: {prediction}** (Confidence: {confidence*100:.2f}%) 👍")
                        st.progress(prob)
                        st.caption(f"Raw Logit: {outputs.item():.4f} | Probability: {prob:.4f}")

                    with tab2:
                        st.subheader("Emotion Breakdown")
                        with st.spinner("Analyzing text and image..."):
                            text_results = text_emotion_pipeline(meme_text)
                            image_results = image_emotion_pipeline(image)
                        st.markdown("#### 📝 Text Emotion")
                        st.json({r['label']: f"{r['score']*100:.1f}%" for r in text_results[0]})
                        st.markdown("#### 🖼️ Image Emotion")
                        st.json({r['label']: f"{r['score']*100:.1f}%" for r in image_results})

                    with tab3:
                        st.subheader("Interpretation")
                        analysis_text = get_interpretive_analysis(prediction, text_results, image_results)
                        st.markdown(analysis_text)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
            else:
                st.warning("⚠️ Please upload an image and enter text.")
        
        # --- Mode 2: Text Only ---
        elif analysis_mode == "Text Only (Emotion)":
            if meme_text:
                try:
                    st.subheader("Text Emotion Analysis")
                    text_results = text_emotion_pipeline(meme_text)
                    st.json({r['label']: f"{r['score']*100:.1f}%" for r in text_results[0]})
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Please enter text.")
        
        # --- Mode 3: Image Only ---
        elif analysis_mode == "Image Only (Emotion)":
            if uploaded_image:
                try:
                    image = Image.open(uploaded_image).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    image_results = image_emotion_pipeline(image)
                    st.json({r['label']: f"{r['score']*100:.1f}%" for r in image_results})
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Please upload an image.")
