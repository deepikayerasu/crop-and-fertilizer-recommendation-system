import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load models
crop_model = pickle.load(open("models/crop_model.pkl", "rb"))
fert_model = pickle.load(open("models/fertilizer_model.pkl", "rb"))

crop_features = pickle.load(open("models/crop_features.pkl", "rb"))
crop_accuracy = pickle.load(open("models/crop_accuracy.pkl", "rb"))
fert_accuracy = pickle.load(open("models/fertilizer_accuracy.pkl", "rb"))

le_soil = pickle.load(open("models/le_soil.pkl", "rb"))
le_crop = pickle.load(open("models/le_crop.pkl", "rb"))
le_fertilizer = pickle.load(open("models/le_fertilizer.pkl", "rb"))

st.set_page_config(page_title="Smart Agriculture ML", layout="wide")

st.title("🌾 Smart Agriculture Recommendation System")

menu = st.sidebar.selectbox(
    "Choose Service",
    ["Crop Recommendation", "Fertilizer Recommendation"]
)

# =====================================
# 🌱 CROP SECTION
# =====================================

if menu == "Crop Recommendation":

    st.header("🌱 Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0)
        K = st.number_input("Potassium (K)", min_value=0.0)
        temperature = st.number_input("Temperature", min_value=0.0)

    with col2:
        humidity = st.number_input("Humidity", min_value=0.0)
        ph = st.number_input("pH", min_value=0.0)
        rainfall = st.number_input("Rainfall", min_value=0.0)

    if st.button("Predict Crop"):

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(input_data)

        st.success(f"Recommended Crop: {prediction[0]}")
        st.info(f"Model Accuracy: {round(crop_accuracy*100,2)}%")

        # Feature Importance (Only if Random Forest)
        if hasattr(crop_model, "feature_importances_"):

            st.subheader("📊 Feature Importance")

            importances = crop_model.feature_importances_

            fig, ax = plt.subplots()
            ax.barh(crop_features, importances)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            ax.set_title("Feature Importance")

            st.pyplot(fig)

# =====================================
# 🌿 FERTILIZER SECTION
# =====================================

if menu == "Fertilizer Recommendation":

    st.header("🌿 Fertilizer Recommendation")

    soil = st.selectbox("Soil Type", le_soil.classes_)
    crop = st.selectbox("Crop Type", le_crop.classes_)

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen", min_value=0.0)
        P = st.number_input("Phosphorus", min_value=0.0)

    with col2:
        K = st.number_input("Potassium", min_value=0.0)
        temperature = st.number_input("Temperature", min_value=0.0)
        humidity = st.number_input("Humidity", min_value=0.0)

    if st.button("Recommend Fertilizer"):

        soil_encoded = le_soil.transform([soil])[0]
        crop_encoded = le_crop.transform([crop])[0]

        input_data = np.array([[temperature, humidity, soil_encoded, crop_encoded, N, P, K]])

        prediction = fert_model.predict(input_data)
        fertilizer = le_fertilizer.inverse_transform(prediction)

        st.success(f"Recommended Fertilizer: {fertilizer[0]}")
        st.info(f"Model Accuracy: {round(fert_accuracy*100,2)}%")
