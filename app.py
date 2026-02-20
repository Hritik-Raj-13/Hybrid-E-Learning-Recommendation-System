from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Load trained RandomForest pipeline
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset used for recommendations
df = pd.read_csv("course_recommendations_dataset_large (1).csv")

# Keep required columns only
df = df[["CourseId", "Branch", "ParentCourse"]].dropna()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Dropdown inputs
        degree = request.form.get("degree")
        branch = request.form.get("branch")

        raw_degree = degree.strip()
        raw_branch = branch.strip()

        degree = raw_degree.lower()
        branch = raw_branch.lower()

        # Filter relevant courses
        filtered = df[
            df["ParentCourse"].str.lower().str.contains(degree, na=False)
            & df["Branch"].str.lower().str.contains(branch, na=False)
        ]

        # Fallback if no strict match
        if filtered.empty:
            filtered = df[df["Branch"].str.lower().str.contains(branch, na=False)]

        # If still empty, return message
        if filtered.empty:
            return render_template(
                "index.html",
                recommendations=[]
            )

        # Build text features exactly like training
        text_inputs = (
            filtered["ParentCourse"] + " "
            + filtered["Branch"] + " "
            + filtered["CourseId"]
        ).tolist()

        # Predict ratings for all candidate courses
        scores = model.predict(text_inputs)

        filtered = filtered.copy()
        filtered["Predicted Rating"] = np.round(scores, 2)

        # 🔥 FIX: remove duplicate courses
        filtered = (
            filtered
            .sort_values(by="Predicted Rating", ascending=False)
            .drop_duplicates(subset="CourseId")
        )

        # Take top 5 unique courses
        recommendations = (
            filtered
            .head(5)
            [["CourseId", "Branch", "Predicted Rating"]]
            .rename(columns={"CourseId": "Course"})
            .to_dict(orient="records")
        )

        return render_template(
            "index.html",
            recommendations=recommendations,
            selected_degree=raw_degree,
            selected_branch=raw_branch

        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
