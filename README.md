# 🧠 Mental Health Status Check

An ML-based web application that allows users to reflect on different aspects of their emotional well-being through a simple interactive assessment.

## 🌐 Live Demo 
[Click here to use Mental Health Status Check](https://mental-health-status-analysis.onrender.com/)

## 📌 Project Overview

Mental Health Status Check is a machine learning-based web application developed using Python and Flask.

The application collects user responses across six parameters:

- Anxiety
- Stress
- Anger
- Sleep Issues
- Mood Swings
- Low Mood

Based on the submitted responses, a trained Random Forest classification model predicts an educational risk category:

- Low Risk
- Medium Risk
- High Risk

The application also displays general wellness suggestions based on the predicted category.

## ✨ Features

- Interactive mental health assessment.
- Six assessment parameters with a 1–5 rating scale.
- Dynamic assessment progress indicator.
- Machine learning-based classification.
- Personalized category-based general suggestions.
- Retake assessment functionality.
- Responsive user interface.
- Flask-based prediction API.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| HTML | Webpage structure |
| CSS | User interface design |
| JavaScript | Interactive functionality |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Random Forest | Classification algorithm |
| Joblib | Model serialization |

## 🤖 Machine Learning Model

The project uses a Random Forest Classifier trained on a synthetic dataset.

The dataset contains six assessment features, with responses represented on a scale from 1 to 5.

The model predicts one of three categories:

1. Low Risk
2. Medium Risk
3. High Risk

## 📂 Project Structure

```text
mental-health-status-check/
│
├── app.py
├── dataset.py
├── train_model.py
├── mental_health_data.csv
├── mental_health_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd mental-health-status-check
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

## 🧪 Model Training

To generate the synthetic dataset and train the model:

```bash
python dataset.py
python train_model.py
```

The trained model is saved as:

```text
mental_health_model.pkl
```

The application can then load this model for predictions.

## ⚠️ Important Disclaimer

This project is developed for educational and demonstration purposes only.

The model is trained using synthetic data and its predictions are not clinically validated. The displayed categories must not be interpreted as medical diagnoses, mental health assessments, or substitutes for professional advice.

Users experiencing emotional distress should consider seeking support from a qualified mental health professional.

## 👩‍💻 Developer

**Gayathri Peyyala**

GitHub: [gayathri-nag](https://github.com/gayathri-nag)