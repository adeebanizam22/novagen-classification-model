# NovaSense — Health Risk Predictor

A machine learning app that predicts whether a patient is **Healthy or Unhealthy** based on health parameters.

## Live Demo

https://huggingface.co/spaces/AdeebaNizam/novagen-health-risk-predictor

## What this app does
- Takes patient health inputs like Age, BMI, Blood Pressure, Cholesterol etc.
- Predicts if the patient is Healthy (1) or Unhealthy (0)
- Uses Random Forest model with 93.5% accuracy and 95.7% recall

## Models Compared
| Model | Accuracy | Recall |
|-------|----------|--------|
| Random Forest | 93.5% | 95.7% |
| Gradient Boosting | 92.7% | 94.1% |
| KNN | 93.5% | 93.8% |
| SVM | 88.3% | 87.0% |
| Logistic Regression | 81.4% | 82.7% |
| Decision Tree | 79.5% | 89.4% |

## Tech Stack
Python · Scikit-learn · Random Forest · Gradio · Pandas
