import os
import pandas as pd
import tensorflow as tf
import numpy as np
import requests
import logging
import json
import re
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from google.generativeai import configure, GenerativeModel

# ========== Flask App Initialization ==========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ========== Logging Setup ==========
logging.basicConfig(level=logging.DEBUG)

# ========== Model Loading ==========
custom_objects = {"loss": CategoricalCrossentropy(reduction="sum_over_batch_size")}

tomato_stage_model = load_model("readymodels/tomato_stage_classifier.h5", custom_objects=custom_objects)
tomato_disease_model = load_model("readymodels/tomato_disease_classifier.h5", custom_objects=custom_objects)
chilli_stage_model = load_model("readymodels/CHILLIPEPPER_stage_classifier.h5", custom_objects=custom_objects)
chilli_disease_model = load_model("readymodels/CHILLIPEPPER_diseases_prediction.h5", custom_objects=custom_objects)
# ========== Labels ==========
tomato_stage_labels = ["Growing", "Vegetative", "Flowering"]
tomato_disease_labels = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato_healthy"
]
chilli_stage_labels = ["Growing", "Vegetative", "Flowering"]
chilli_disease_labels = [
    "Bacterial_Spot", "Cercospora_Leaf_Spot", "chilli_healthy",
    "Curl_Virus", "Nutrition__Deficiency", "White_spot"
]

# ========== API Keys ==========
WEATHER_API_KEY = "6212ccd8bd7b6f3657b18b690bc9ba25"
GEMINI_API_KEY = "AIzaSyB7QIqPNg89yzr-t3msQANz13gbOsNh3BI"

# ========== Gemini Setup ==========
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-pro")

# ========== Weather Function ==========
def get_weather_data(latitude, longitude):
    if latitude is None or longitude is None:
        logging.warning("Missing latitude or longitude. Using default values.")
        return 25, 60, 0

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        weather = response.json()
        temperature = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        rainfall = weather.get("rain", {}).get("1h", 0)
        return temperature, humidity, rainfall
    except Exception as e:
        logging.error(f"Weather API Error: {e}")
        return 25, 60, 0

# ========== Gemini Recommendation ==========
def gemini_fertilizer_crop_recommendation(soil_data):
    prompt = (
        f"Given the soil data below, suggest suitable crops and fertilizers in JSON format:\n"
        f"{json.dumps(soil_data, indent=2)}\n\n"
        f"Respond strictly in this JSON format:\n"
        f'{{"crops": ["crop1", "crop2"], "fertilizer": "recommended fertilizer"}}'
    )

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        logging.debug(f"Raw Gemini Response: {content}")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in Gemini response.")

        cleaned_json = json_match.group(0)
        recommendations = json.loads(cleaned_json)

        crops = recommendations.get("crops", ["No crops recommended"])
        fertilizer = recommendations.get("fertilizer", "No fertilizer recommended")

        logging.info(f"Recommended Crops: {crops}")
        logging.info(f"Recommended Fertilizer: {fertilizer}")

        return crops, fertilizer

    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return ["No crops recommended"], "No fertilizer recommended"

# ========== Soil Report Endpoint ==========
@app.route("/soilreport", methods=["POST"])
def handle_soil_report():
    try:
        data = request.json
        logging.debug(f"Received soil report data: {data}")

        def safe_float(value, default=0):
            try:
                return float(value) if value not in (None, "") else default
            except ValueError:
                return default

        n = safe_float(data.get("nitrogen"))
        p = safe_float(data.get("phosphorus"))
        k = safe_float(data.get("potassium"))
        ph = safe_float(data.get("ph"))
        soil_type = data.get("soilTexture", "Loam")
        latitude = data.get("latitude", 12.9716)
        longitude = data.get("longitude", 77.5946)

        temperature, humidity, rainfall = get_weather_data(latitude, longitude)

        soil_data = {
            "nitrogen": n,
            "phosphorus": p,
            "potassium": k,
            "ph": ph,
            "soil_type": soil_type,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall
        }

        logging.info(f"Processed Soil Data: {soil_data}")

        crop_recommendations, fertilizer_recommendation = gemini_fertilizer_crop_recommendation(soil_data)

        return jsonify({
            "cropRecommendations": crop_recommendations,
            "fertilizerRecommendation": fertilizer_recommendation,
            "weather": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall
            }
        }), 200

    except Exception as e:
        logging.error(f"Error in /soilreport: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========== Image Preprocessing ==========
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ========== Prediction Function ==========
def predict_stage_and_disease(plant_name, img_path):
    img_array = preprocess_image(img_path)

    if plant_name.lower() == "tomato":
        stage_pred = tomato_stage_model.predict(img_array)
        disease_pred = tomato_disease_model.predict(img_array)
        stage = tomato_stage_labels[np.argmax(stage_pred)]
        disease = tomato_disease_labels[np.argmax(disease_pred)]

    elif plant_name.lower() == "chilli":
        stage_pred = chilli_stage_model.predict(img_array)
        disease_pred = chilli_disease_model.predict(img_array)
        stage = chilli_stage_labels[np.argmax(stage_pred)]
        disease = chilli_disease_labels[np.argmax(disease_pred)]

    else:
        raise ValueError("Unsupported plant name")

    logging.info(f"Predicted stage: {stage}, disease: {disease}")
    return {"stage": stage, "disease": disease}

# ========== New Plant Upload Endpoint ==========
@app.route("/plantupload", methods=["POST"])
def handle_plant_upload():
    try:
        logging.debug(f"Form keys: {list(request.form.keys())}")
        logging.debug(f"Files: {list(request.files.keys())}")
        if 'image' not in request.files or 'plantName' not in request.form:
            return jsonify({"error": "Image and plantName are required"}), 400

        img_file = request.files['image']
        plant_name = request.form['plantName']

        if img_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, img_file.filename)
        img_file.save(temp_path)

        prediction = predict_stage_and_disease(plant_name, temp_path)

        os.remove(temp_path)
        return jsonify(prediction), 200

    except Exception as e:
        logging.error(f"Error in /plantupload: {e}")
        return jsonify({"error": str(e)}), 500

# ========== Start Server ==========
if __name__ == "__main__":
    app.run(debug=True)
