import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/ott_dropoff_model.pkl")

# Sample event data for prediction
sample_event = {
    "platform": "Netflix",
    "genre": "Drama",
    "dataset_version": "v1",
    "dialogue_density": "High",
    "attention_required": "Medium",
    "episode_duration_min": 45,
    "pacing_score": 7,
    "hook_strength": 8,
    "visual_intensity": 6,
    "avg_watch_percentage": 60,
    "pause_count": 3,
    "rewind_count": 1,
    "skip_intro": 1,
    "cognitive_load": 5,
    "season_number": 2,
    "episode_number": 6,
    "release_year": 2022,
    "night_watch_safe": 1
}

# Convert to DataFrame
df = pd.DataFrame([sample_event])

# Generate predictions
prediction = model.predict(df)
probability = model.predict_proba(df)[:, 1]

# Display results
print("Drop-off Prediction (0 = No, 1 = Yes):", int(prediction[0]))
print("Drop-off Probability:", round(probability[0], 3))