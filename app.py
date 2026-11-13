#from flask import Flask, render_template, request, jsonify
#import joblib
#import numpy as np

#app = Flask(__name__)

# Load trained ML model=
#try:
 #   model = joblib.load("mental_health_model.pkl")  # Ensure this file exists
#except FileNotFoundError:
 #   print("❌ Model file not found! Train the model first.")

#@app.route("/", methods=["GET"])
#def home():
 #   return render_template("index.html")

#@app.route("/predict", methods=["POST"])
#def predict():
 #   try:
        # Get user input from form
  #      anxiety = request.form.get("anxiety")
   #     stress = request.form.get("stress")
    #    anger = request.form.get("anger")
     #   sleep_issues = request.form.get("sleep_issues")
      #  mood_swings = request.form.get("mood_swings")
       # depression = request.form.get("depression")

        # Check if any value is None (missing)
        #if None in [anxiety, stress, anger, sleep_issues, mood_swings, depression]:
         #   return jsonify({"error": "Missing input values"}), 400

        # Convert inputs to integers
        #user_data = np.array([[int(anxiety), int(stress), int(anger), int(sleep_issues), int(mood_swings), int(depression)]])

        # Make prediction
        #prediction = model.predict(user_data)[0]  # Get the first (and only) prediction

        # Convert prediction to human-readable format
        #status_dict = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        #result = status_dict.get(prediction, "Unknown")

        #return jsonify({"status": result})  # Return JSON response

    #except Exception as e:
     #   return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
 #   app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("mental_health_model.pkl")

# Feature names used during training
FEATURES = [
    "Anxiety",
    "Stress",
    "Anger",
    "Sleep Issues",
    "Mood Swings",
    "Depression"
]

STATUS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []

        for feature in FEATURES:
            value = request.form.get(feature.lower().replace(" ", "_"))

            if value is None or value.strip() == "":
                return jsonify({"error": f"Missing value: {feature}"}), 400

            score = int(value)

            if score < 1 or score > 5:
                return jsonify({
                    "error": f"{feature} must be between 1 and 5"
                }), 400

            values.append(score)

        # Create DataFrame with matching feature names
        user_data = pd.DataFrame([values], columns=FEATURES)

        # Predict mental health status
        prediction = int(model.predict(user_data)[0])

        return jsonify({
            "status": STATUS.get(prediction, "Unknown"),
            "message": "This is an educational screening result, not a medical diagnosis."
        })

    except ValueError:
        return jsonify({"error": "Please enter valid numeric values."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)