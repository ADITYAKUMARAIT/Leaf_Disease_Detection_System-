import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths and configuration
MODEL_PATH = "mode.h5"  # Path to your saved model
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
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Solution: This viral disease is transmitted by whiteflies (Bemisia tabaci). "
        "Use systemic insecticides such as Imidacloprid or Thiamethoxam as soil drench or foliar spray to control whitefly populations. "
        "Remove and destroy infected plants promptly to reduce virus reservoirs. Maintain weed control and use reflective mulches to deter whiteflies. "
        "Ensure proper field sanitation and crop rotation to reduce vector habitats."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Solution: This virus spreads mechanically through contaminated tools and aphids. "
        "Disinfect tools regularly with 10% bleach or 70% alcohol solution. "
        "Control aphids using insecticides like Acephate or Spinosad. "
        "Remove infected plants immediately and practice crop rotation. Use virus-resistant tomato varieties if available."
    ),
    "Tomato___Target_Spot": (
        "Solution: Caused by Corynespora cassiicola, a fungal pathogen. "
        "Apply broad-spectrum fungicides such as Chlorothalonil or Mancozeb at recommended rates every 7–10 days. "
        "Remove infected leaves and plant debris to reduce inoculum. Ensure good airflow by proper plant spacing and pruning. Avoid overhead irrigation to reduce leaf wetness duration."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Solution: Spider mites (Tetranychus urticae) thrive in hot, dry conditions. "
        "Apply miticides like Abamectin or Spiromesifen as per label instructions. "
        "Incorporate insecticidal soaps or horticultural oils for mild infestations. "
        "Increase humidity around plants and encourage natural predators like Phytoseiulus persimilis. Rotate miticides to prevent resistance."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Solution: Caused by Septoria lycopersici fungus. "
        "Use fungicides containing Chlorothalonil or Copper oxychloride early in the season and repeat applications every 7–14 days during favorable conditions. "
        "Remove and destroy infected leaves and crop residues. Practice crop rotation and avoid overhead irrigation. Maintain plant vigor through balanced fertilization."
    ),
    "Tomato___Leaf_Mold": (
        "Solution: Caused by Passalora fulva fungus, favored by high humidity. "
        "Apply fungicides such as Copper-based compounds or Mancozeb preventatively. "
        "Increase ventilation in greenhouses or field plantings by pruning and proper spacing. "
        "Avoid excessive nitrogen fertilization that promotes dense foliage. Remove infected leaves promptly."
    ),
    "Tomato___Late_blight": (
        "Solution: Caused by Phytophthora infestans, a devastating oomycete pathogen. "
        "Use fungicides containing Metalaxyl-M (Mefenoxam) or Chlorothalonil as preventive and curative sprays. "
        "Begin treatments at first sign of disease or during favorable wet conditions, applying every 5–7 days. "
        "Remove and destroy infected plant material. Avoid overhead irrigation and improve air circulation."
    ),
    "Tomato___healthy": (
        "Solution: No disease detected. Continue with best management practices: balanced irrigation, fertilization, pest scouting, and maintaining plant health. "
        "Use disease-resistant varieties and rotate crops to reduce disease buildup."
    ),
    "Tomato___Early_blight": (
        "Solution: Caused by Alternaria solani fungus. "
        "Apply fungicides like Chlorothalonil, Mancozeb, or Azoxystrobin every 7–10 days starting at early fruit set. "
        "Remove and destroy infected plant debris. Avoid overhead irrigation and ensure good airflow by proper spacing and pruning."
    ),
    "Tomato___Bacterial_spot": (
        "Solution: Caused by Xanthomonas campestris pv. vesicatoria. "
        "Apply copper-based bactericides such as Copper hydroxide or Copper oxychloride every 7–10 days. "
        "Remove infected plant material and avoid working in wet fields to prevent spread. Use certified disease-free seed and resistant varieties when available. Avoid overhead irrigation to minimize leaf wetness."
    )
}

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
st.write("Upload a leaf image, and the model will predict the disease class.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
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
    if st.button("Show Solution"):
        # Display the solution for the predicted class
        solution = SOLUTIONS.get(predicted_class, "Solution not available for this class.")
        st.write(f"**Solution for {predicted_class}:**")
        st.info(solution)

# Display confusion matrix (optional)
if st.button("Show Confusion Matrix"):
    st.write("**Confusion Matrix for Validation Set:**")
    val_dir = 'detection/data/val'
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
