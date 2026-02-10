import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/processed.csv")

# Drop leakage columns
df = df.drop(columns=['show_id','drop_off_probability','retention_risk'])

X = df.drop(columns=['drop_off'])
y = df['drop_off']

numeric_features = [
    "episode_duration_min",
    "pacing_score",
    "hook_strength",
    "visual_intensity",
    "avg_watch_percentage",
    "pause_count",
    "rewind_count",
    "skip_intro",
    "cognitive_load",
    "season_number",
    "episode_number",
    "release_year",
    "night_watch_safe"
]

categorical_nominal = [
    "platform",
    "genre",
    "dataset_version"
]

categorical_ordinal = [
    "dialogue_density",
    "attention_required"
]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('nom', OneHotEncoder(handle_unknown='ignore'), categorical_nominal),
        ('ord', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), categorical_ordinal)
    ]
)

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ott_dropoff_model1.pkl")

print("✅ Model trained & saved")