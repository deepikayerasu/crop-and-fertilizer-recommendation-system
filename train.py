# =========================================
# SMART AGRICULTURE - ADVANCED TRAINING
# =========================================

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Create models directory
os.makedirs("models", exist_ok=True)

# =====================================
# 1️⃣ CROP RECOMMENDATION MODEL
# =====================================

print("\n========== CROP RECOMMENDATION ==========\n")

crop_df = pd.read_csv("data/crop_recommendation.csv")

X_crop = crop_df.drop("label", axis=1)
y_crop = crop_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_crop, y_crop, cv=5).mean()

    print(f"{name}")
    print("Test Accuracy:", acc)
    print("Cross Validation Score:", cv_score)
    print("-" * 40)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\nBest Crop Model Selected: {best_model_name}")

# Save model and metadata
pickle.dump(best_model, open("models/crop_model.pkl", "wb"))
pickle.dump(X_crop.columns.tolist(), open("models/crop_features.pkl", "wb"))
pickle.dump(best_accuracy, open("models/crop_accuracy.pkl", "wb"))

print("Crop model saved successfully!\n")


# =====================================
# 2️⃣ FERTILIZER RECOMMENDATION MODEL
# =====================================

print("\n========== FERTILIZER RECOMMENDATION ==========\n")

fert_df = pd.read_csv("data/fertilizer_recommendation.csv")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

fert_df["soil_type"] = le_soil.fit_transform(fert_df["soil_type"])
fert_df["crop_type"] = le_crop.fit_transform(fert_df["crop_type"])
fert_df["fertilizer_name"] = le_fertilizer.fit_transform(fert_df["fertilizer_name"])

X_fert = fert_df.drop("fertilizer_name", axis=1)
y_fert = fert_df["fertilizer_name"]

X_train, X_test, y_train, y_test = train_test_split(
    X_fert, y_fert, test_size=0.2, random_state=42
)

models_fert = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model_fert = None
best_accuracy_fert = 0
best_model_name_fert = ""

for name, model in models_fert.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_fert, y_fert, cv=5).mean()

    print(f"{name}")
    print("Test Accuracy:", acc)
    print("Cross Validation Score:", cv_score)
    print("-" * 40)

    if acc > best_accuracy_fert:
        best_accuracy_fert = acc
        best_model_fert = model
        best_model_name_fert = name

print(f"\nBest Fertilizer Model Selected: {best_model_name_fert}")

pickle.dump(best_model_fert, open("models/fertilizer_model.pkl", "wb"))
pickle.dump(le_soil, open("models/le_soil.pkl", "wb"))
pickle.dump(le_crop, open("models/le_crop.pkl", "wb"))
pickle.dump(le_fertilizer, open("models/le_fertilizer.pkl", "wb"))
pickle.dump(best_accuracy_fert, open("models/fertilizer_accuracy.pkl", "wb"))

print("Fertilizer model saved successfully!")
