import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import requests
from io import BytesIO
import pandas as pd

st.set_page_config(page_title="Garbage Classifier", layout="wide")
st.title("🗑️ Garbage Image Classifier")

@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("model/resnet50_garbage.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Layout: 3 columns
col1, col2, col3 = st.columns([1, 1, 1])

# LEFT COLUMN — Image input
with col1:
    st.header("🧾 Input Image")
    image_source = st.radio("Select method:", ["📁 Upload", "🌐 URL", "📷 Webcam"])
    image = None

    if image_source == "📁 Upload":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    elif image_source == "🌐 URL":
        url = st.text_input("Paste image URL")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except:
                st.error("Failed to load image from URL.")

    elif image_source == "📷 Webcam":
        camera = st.camera_input("Take a photo")
        if camera:
            image = Image.open(camera).convert("RGB")

# MIDDLE COLUMN — Image preview + prediction
# MIDDLE COLUMN — Image preview + prediction
with col2:
    st.header("Prediction")
    probs = None  # define it outside the button to access it in col3
    if image:
        st.image(image, caption="Input Image", width=300)
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    label = class_names[pred.item()]
                    confidence_pct = conf.item() * 100

                st.success(f"🧠 Predicted Class: **{label.capitalize()}**")
                st.info(f"📊 Confidence: {confidence_pct:.2f}%")

# RIGHT COLUMN — Class probabilities
with col3:
    st.header("Probabilities")
    if image and probs is not None:
        df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": [f"{p * 100:.2f}" for p in probs[0].cpu().numpy()]
        })
        st.dataframe(df, use_container_width=True)