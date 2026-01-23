🧠 Emotion Recognition on Multi-Modal Data using Deep Learning
🎯 Overview

This project presents a multi-modal emotion recognition system that analyzes image + text data (such as memes) to detect both emotional context and hateful intent.
By combining OpenAI’s CLIP architecture with attention-based fusion, the system captures deeper relationships between visual and textual modalities — achieving more accurate and interpretable predictions.
Streamlit link - https://emotion-recognition-app-mvtrwgbjnvlj23sdcvjdyz.streamlit.app/ 

🚀 Features

✅ Multi-modal hate and emotion detection (Image + Text)
✅ Uses CLIP for visual–textual feature extraction
✅ Cross-attention fusion model for deep contextual alignment
✅ Separate text and image emotion analysis using:

RoBERTa (GoEmotions) for text

ViT-Face-Expression for images
✅ Streamlit-based interactive UI for easy testing and visualization
✅ Provides AI-driven interpretive analysis explaining how emotions contribute to the final decision

🧩 Project Structure
📁 DL-Project/
│
├── app.py                      # Streamlit app script
├── best_hateful_meme_model.pth # Trained multimodal model
├── requirements.txt             # Dependencies for deployment
└── README.md                    # Project documentation
⚙️ Installation
1️⃣ Clone this repository
git clone https://github.com/<your-username>/emotion-recognition-app.git
cd emotion-recognition-app

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py

🧠 Model Architecture
1. Feature Extraction

Image Encoder: CLIP Vision Transformer (openai/clip-vit-base-patch32)

Text Encoder: CLIP Text Transformer
Both produce 512-D embeddings for each modality.

2. Fusion Mechanism

Cross-Attention (torch.nn.MultiheadAttention) aligns visual and textual embeddings.

3. Classification Head

Two-layer fully connected neural network with ReLU + Dropout

Sigmoid output for binary classification (Hateful / Not Hateful)

4. Emotion Analysis

Text Emotion: RoBERTa-GoEmotions

Image Emotion: ViT-Face-Expression

🧪 Example Use Case

You can upload a meme image and enter its caption.
The system will:

Predict whether the meme is Hateful or Not Hateful.

Extract and display dominant emotions from both image and text.

Generate a human-readable explanation connecting emotions to the final decision.

📊 Model Performance
Metric	Value	Comment
Test Accuracy	64.06%	Solid baseline for multimodal hate/emotion classification
ROC-AUC	0.7107	Reliable discrimination ability
Weighted F1	~0.65	Balanced precision and recall
Validation ROC-AUC	0.7381	Good generalization

🛠️ Technologies Used
Category	Tools
Language	Python 3.10
Frameworks	PyTorch, Streamlit
Models	CLIP, RoBERTa-GoEmotions, ViT-Face-Expression
Libraries	Transformers, Pillow, NumPy, Pandas, tqdm
Deployment	Streamlit Cloud / Render
💡 Future Scope

🔹 Add audio modality (MFCC + LSTM) for tri-modal emotion fusion
🔹 Fine-tune CLIP on emotion-specific datasets for better domain adaptation
🔹 Extend to datasets like Flickr8k or AffectNet
🔹 Integrate advanced interpretability (Grad-CAM, SHAP) for visual reasoning
