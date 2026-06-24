import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_health(age, bmi, blood_pressure, cholesterol, glucose_level, heart_rate,
                   sleep_hours, exercise_hours, water_intake, stress_level,
                   smoking, alcohol, diet, mental_health, physical_activity,
                   medical_history, allergies,
                   diet_vegan, diet_vegetarian,
                   blood_group_ab, blood_group_b, blood_group_o):

    # Map categorical inputs to numbers
    cat_map = {"Never": 0, "Sometimes": 1, "Always": 2}
    quality_map = {"Poor": 0, "Moderate": 1, "Good": 2}

    smoking_val = cat_map[smoking]
    alcohol_val = cat_map[alcohol]
    diet_val = quality_map[diet]
    mental_val = quality_map[mental_health]
    physical_val = cat_map[physical_activity]
    medical_val = cat_map[medical_history]
    allergy_val = cat_map[allergies]

    # Boolean inputs
    vegan_val = 1 if diet_vegan == "Yes" else 0
    veg_val = 1 if diet_vegetarian == "Yes" else 0
    ab_val = 1 if blood_group_ab == "Yes" else 0
    b_val = 1 if blood_group_b == "Yes" else 0
    o_val = 1 if blood_group_o == "Yes" else 0

    # Build input array in exact column order
    input_data = np.array([[age, bmi, blood_pressure, cholesterol, glucose_level,
                            heart_rate, sleep_hours, exercise_hours, water_intake,
                            stress_level, smoking_val, alcohol_val, diet_val,
                            mental_val, physical_val, medical_val, allergy_val,
                            vegan_val, veg_val, ab_val, b_val, o_val]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        result = "✅ Healthy"
        confidence = f"{probability[1]*100:.1f}%"
        message = f"The model predicts the patient is **Healthy** with {confidence} confidence."
    else:
        result = "⚠️ Unhealthy"
        confidence = f"{probability[0]*100:.1f}%"
        message = f"The model predicts the patient is **Unhealthy** with {confidence} confidence."

    return result, message

# UI
with gr.Blocks(title="NovaSense — Health Risk Predictor") as app:
    gr.Markdown("# 🏥 NovaSense — Health Risk Predictor")
    gr.Markdown("Enter patient details below to predict whether the patient is **Healthy or Unhealthy**.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Basic Health Metrics")
            age = gr.Slider(1, 100, value=30, label="Age")
            bmi = gr.Slider(10, 50, value=25, label="BMI")
            blood_pressure = gr.Slider(60, 200, value=120, label="Blood Pressure")
            cholesterol = gr.Slider(100, 300, value=200, label="Cholesterol")
            glucose_level = gr.Slider(50, 200, value=100, label="Glucose Level")
            heart_rate = gr.Slider(40, 150, value=72, label="Heart Rate")
            sleep_hours = gr.Slider(0, 12, value=7, label="Sleep Hours")
            exercise_hours = gr.Slider(0, 10, value=2, label="Exercise Hours per Day")
            water_intake = gr.Slider(0, 10, value=3, label="Water Intake (Litres)")
            stress_level = gr.Slider(0, 10, value=5, label="Stress Level (0=Low, 10=High)")

        with gr.Column():
            gr.Markdown("### 🧬 Lifestyle & History")
            smoking = gr.Radio(["Never", "Sometimes", "Always"], value="Never", label="Smoking")
            alcohol = gr.Radio(["Never", "Sometimes", "Always"], value="Never", label="Alcohol")
            diet = gr.Radio(["Poor", "Moderate", "Good"], value="Moderate", label="Diet Quality")
            mental_health = gr.Radio(["Poor", "Moderate", "Good"], value="Moderate", label="Mental Health")
            physical_activity = gr.Radio(["Never", "Sometimes", "Always"], value="Sometimes", label="Physical Activity")
            medical_history = gr.Radio(["Never", "Sometimes", "Always"], value="Never", label="Medical History")
            allergies = gr.Radio(["Never", "Sometimes", "Always"], value="Never", label="Allergies")

            gr.Markdown("### 🩸 Diet Type & Blood Group")
            diet_vegan = gr.Radio(["Yes", "No"], value="No", label="Vegan Diet")
            diet_vegetarian = gr.Radio(["Yes", "No"], value="No", label="Vegetarian Diet")
            blood_group_ab = gr.Radio(["Yes", "No"], value="No", label="Blood Group AB")
            blood_group_b = gr.Radio(["Yes", "No"], value="No", label="Blood Group B")
            blood_group_o = gr.Radio(["Yes", "No"], value="No", label="Blood Group O")

    predict_btn = gr.Button("🔍 Predict", variant="primary")

    with gr.Row():
        result_label = gr.Label(label="Prediction")
        result_message = gr.Markdown()

    predict_btn.click(
        fn=predict_health,
        inputs=[age, bmi, blood_pressure, cholesterol, glucose_level, heart_rate,
                sleep_hours, exercise_hours, water_intake, stress_level,
                smoking, alcohol, diet, mental_health, physical_activity,
                medical_history, allergies,
                diet_vegan, diet_vegetarian,
                blood_group_ab, blood_group_b, blood_group_o],
        outputs=[result_label, result_message]
    )

    gr.Markdown("---")
    gr.Markdown("**Model:** Random Forest · **Accuracy:** 93.5% · **Recall:** 95.7% · Built with Python & Scikit-learn")

app.launch()
