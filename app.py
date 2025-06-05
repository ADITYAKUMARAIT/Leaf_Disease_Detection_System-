import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
from sklearn.metrics import confusion_matrix
from tempfile import NamedTemporaryFile
from io import BytesIO 

# Paths and configuration
MODEL_PATH = "model.h5"  # Path to your saved model
CLASS_NAMES = sorted([
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Target_Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Leaf_Mold",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Bacterial_spot"
])  # Classes from the training directory

# Define solutions for each disease class
SOLUTIONS = {
    "English": {
        "Tomato___Early_blight": "Cause: Alternaria solani fungus.\n"
            "Use Chlorothalonil (2 g/L), Mancozeb (2.5 g/L), or Azoxystrobin (1 ml/L) at 7–10 day intervals. "
            "Remove and destroy infected leaves. Avoid overhead irrigation. Maintain good spacing and rotate crops.",

        "Tomato___Late_blight": "Cause: Phytophthora infestans (oomycete).\n"
            "Spray Metalaxyl-M (0.35 g/L) or Chlorothalonil (2 g/L) every 5–7 days. Remove infected plants. "
            "Avoid wetting leaves, and use resistant tomato varieties.",

        "Tomato___Bacterial_spot": "Cause: Xanthomonas bacteria.\n"
            "Use Copper hydroxide (2.5 g/L) or Copper oxychloride (3 g/L) every 7–10 days. "
            "Avoid working on wet plants. Use certified disease-free seeds and maintain field hygiene.",

        "Tomato___Leaf_Mold": "Cause: Passalora fulva fungus.\n"
            "Spray Mancozeb (2.5 g/L) or Copper fungicides preventively. Improve ventilation. "
            "Avoid excessive nitrogen, and remove affected leaves promptly.",

        "Tomato___Septoria_leaf_spot": "Cause: Septoria lycopersici.\n"
            "Apply Chlorothalonil (2 g/L) or Copper oxychloride (2.5 g/L) every 10 days. "
            "Remove infected leaves. Avoid overhead watering. Rotate crops and sanitize tools.",

        "Tomato___Spider_mites Two-spotted_spider_mite": "Cause: Tetranychus urticae mites.\n"
            "Use Abamectin (0.5 ml/L) or Spiromesifen (1 ml/L). Maintain humidity. Encourage natural predators like Phytoseiulus persimilis.",

        "Tomato___Target_Spot": "Cause: Corynespora cassiicola.\n"
            "Use Chlorothalonil or Mancozeb every 7–10 days. Improve air circulation. Remove infected plant debris.",

        "Tomato___Tomato_mosaic_virus": "Cause: Transmitted via aphids and contaminated tools.\n"
            "Disinfect tools with 10% bleach, control aphids using Imidacloprid (0.5 ml/L), and remove infected plants.",

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Cause: Whiteflies (Bemisia tabaci).\n"
            "Use Imidacloprid (0.5 ml/L) or Thiamethoxam. Install yellow sticky traps and remove infected plants.",

        "Tomato___healthy": "No disease detected.\n"
            "Maintain regular inspection, use disease-free seeds, and apply balanced fertilizer and irrigation practices."
    },
    "Hindi": {
        "Tomato___Early_blight": "कारण: Alternaria solani फफूंद।\n"
            "Chlorothalonil (2 ग्राम/लीटर), Mancozeb (2.5 ग्राम/लीटर) या Azoxystrobin (1 मिली/लीटर) का छिड़काव करें। "
            "संक्रमित पत्तों को हटाएं, अधिक पानी से बचें, और पौधों के बीच दूरी बनाए रखें।",

        "Tomato___Late_blight": "कारण: Phytophthora infestans।\n"
            "Metalaxyl-M (0.35 ग्राम/लीटर) या Chlorothalonil का 5–7 दिन के अंतराल पर छिड़काव करें। "
            "संक्रमित पौधों को हटाएं और पत्तों को गीला न करें।",

        "Tomato___Bacterial_spot": "कारण: Xanthomonas बैक्टीरिया।\n"
            "Copper hydroxide या Copper oxychloride का छिड़काव करें। प्रमाणित बीजों का उपयोग करें और खेत की स्वच्छता बनाए रखें।",

        "Tomato___Leaf_Mold": "कारण: Passalora fulva फफूंद।\n"
            "Mancozeb या Copper आधारित कवकनाशी का उपयोग करें। हवादार वातावरण बनाए रखें और संक्रमित पत्तियों को हटाएं।",

        "Tomato___Septoria_leaf_spot": "कारण: Septoria lycopersici फफूंद।\n"
            "Chlorothalonil या Copper oxychloride का 10 दिन के अंतराल पर छिड़काव करें। संक्रमित पत्तियों को हटाएं।",

        "Tomato___Spider_mites Two-spotted_spider_mite": "कारण: Tetranychus urticae माइट्स।\n"
            "Abamectin या Spiromesifen का उपयोग करें। नमी बनाए रखें और जैविक शिकारियों का उपयोग करें।",

        "Tomato___Target_Spot": "कारण: Corynespora cassiicola।\n"
            "Chlorothalonil या Mancozeb का नियमित छिड़काव करें और संक्रमित पौधों को हटा दें।",

        "Tomato___Tomato_mosaic_virus": "कारण: एफिड और संक्रमित उपकरणों से फैलता है।\n"
            "उपकरणों को कीटाणुरहित करें, एफिड को नियंत्रित करें और संक्रमित पौधों को हटा दें।",

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "कारण: सफेद मक्खियां।\n"
            "Imidacloprid या Thiamethoxam का प्रयोग करें। पीले स्टिकी ट्रैप लगाएं।",

        "Tomato___healthy": "कोई बीमारी नहीं है। नियमित निगरानी करें और अच्छे खेती के तरीके अपनाएं।"
    },
    "Marathi": {
        "Tomato___Early_blight": "कारण: Alternaria solani नावाची बुरशी.\n"
            "Chlorothalonil (2 ग्रॅम/लिटर), Mancozeb (2.5 ग्रॅम/लिटर), किंवा Azoxystrobin (1 मिली/लिटर) फवारणी करा. "
            "संक्रमित पाने काढा. योग्य अंतर ठेवा.",

        "Tomato___Late_blight": "कारण: Phytophthora infestans नावाचा ओमायसिट.\n"
            "Metalaxyl-M किंवा Chlorothalonil दर 5–7 दिवसांनी फवारणी करा. संक्रमित झाडे काढून टाका.",

        "Tomato___Bacterial_spot": "कारण: Xanthomonas बॅक्टेरिया.\n"
            "Copper hydroxide किंवा Copper oxychloride वापरा. अधिक ओलावा टाळा.",

        "Tomato___Leaf_Mold": "कारण: Passalora fulva फंगस.\n"
            "Mancozeb किंवा Copper आधारित फंगिसाइड वापरा. हवा खेळती ठेवा.",

        "Tomato___Septoria_leaf_spot": "कारण: Septoria lycopersici.\n"
            "Chlorothalonil किंवा Copper oxychloride वापरून फवारणी करा. संक्रमित पानं हटवा.",

        "Tomato___Spider_mites Two-spotted_spider_mite": "कारण: Tetranychus urticae कोळ्यांमुळे होणारा रोग.\n"
            "Abamectin किंवा Spiromesifen फवारणी करा. नमी ठेवा.",

        "Tomato___Target_Spot": "कारण: Corynespora cassiicola बुरशी.\n"
            "Chlorothalonil किंवा Mancozeb वापरा. झाडांमधील जागा मोकळी ठेवा.",

        "Tomato___Tomato_mosaic_virus": "कारण: उपकरणे आणि एफिड द्वारे पसरतो.\n"
            "साफसफाई ठेवा. एफिड नियंत्रण करा.",

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "कारण: पांढऱ्या माश्यांमुळे विषाणूचा प्रादुर्भाव.\n"
            "Imidacloprid किंवा Thiamethoxam वापरा. पीळ ट्रॅप्स लावा.",

        "Tomato___healthy": "रोग सापडला नाही. नियमित तपासणी करा. योग्य पाणी आणि खत द्या."
    }
}

lang_codes = {"English": "en", "Hindi": "hi", "Marathi": "mr"}

def generate_audio_bytes(text, lang):
    # Fallback Marathi audio to Hindi if needed
    if lang == "mr":
        lang = "hi"
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

def predict_image(image):
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence, predictions[0]

# UI Layout
st.title("Plant Disease Detection")
language = st.selectbox("Choose Language / भाषा निवडा / भाषा चुनें", ["English", "Hindi", "Marathi"])
st.write("Upload a leaf image, and the model will predict the disease class.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display uploaded image
    image = load_img(uploaded_file, target_size=(150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    st.write("**Prediction in Progress...**")
    predicted_class, confidence, predictions = predict_image(image)

    # Display prediction
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Display bar chart for all classes
    st.write("**Class Probabilities:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=CLASS_NAMES, y=predictions, ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Confidence")
    st.pyplot(fig)

    # Add Solution Button
    if st.button("Show Solution and Play Audio"):
        solution = SOLUTIONS[language].get(predicted_class, "No solution available in selected language.")
        st.write(f"### Solution in {language}:")
        st.info(solution)
        try:
            audio_bytes = generate_audio_bytes(solution, lang_codes[language])
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")

# Display confusion matrix (optional)
if st.button("Show Confusion Matrix"):
    st.write("**Confusion Matrix for Validation Set:**")
    val_dir = 'data/val'
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False
    )
    y_true = val_generator.classes
    y_pred = model.predict(val_generator)
    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)
