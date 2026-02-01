<h1 align="center">💳 Credit Risk Prediction — Streamlit + ML</h1>

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Made%20with-Python-blue?logo=python"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Framework-Scikit--learn-orange?logo=scikitlearn"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit"></a>
    <img src="https://img.shields.io/badge/Model-Random%20Forest-2b6cb0">
    <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

<p align="center">
    <img src="img.png" alt="Credit Risk App" width="900" />
</p>

> 🧠 An end-to-end credit risk prediction system: train on German credit data, export deployable artifacts, and use a Streamlit dashboard to score applicants with SHAP explanations in seconds.

---

## 📘 Project Overview

This repository contains:

- A full training notebook (EDA → preprocessing → training → evaluation → export)
- Saved model and encoders for reuse in apps/APIs
- A Streamlit UI to enter applicant details, run predictions, and view SHAP-driven explanations

---

## 🎯 Key Features

- End-to-end ML workflow with clear, reproducible steps
- Saved Random Forest model (`random_forest_credit_model.pkl`) and label encoders
- Streamlit “what-if” UI with presets and SHAP (local + global) explainability
- Displays verdict + class probabilities + top drivers

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `Credit Prediction.ipynb` | Notebook workflow: EDA → preprocessing → modeling → evaluation → export. |
| `app.py` | Streamlit app for live credit risk scoring with SHAP explanations. |
| `random_forest_credit_model.pkl` | Saved Random Forest model. |
| `encoders/` | Saved label encoders for categorical features. |
| `german_credit_data.csv` | Dataset (in-repo for convenience). |
| `requirements.txt` | Project dependencies. |

---

## 🔗 Dataset

- German Credit Data (included as `german_credit_data.csv`).

---

## 🌐 Live Project Link

If deployed: provide your Streamlit Cloud / hosting URL here.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy, Seaborn
- Scikit-learn
- SHAP, Matplotlib
- Streamlit
- Jupyter Notebook

---

## ⚙️ How to Run the Streamlit App

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## ⚙️ How to Load the Model

```python
import joblib
import pandas as pd

model = joblib.load("random_forest_credit_model.pkl")
encoders = {
    "Sex": joblib.load("encoders/Sex_label_encoder.pkl"),
    "Housing": joblib.load("encoders/Housing_label_encoder.pkl"),
    "Saving accounts": joblib.load("encoders/Saving accounts_label_encoder.pkl"),
    "Checking account": joblib.load("encoders/Checking account_label_encoder.pkl"),
}

# Example encoding for a single row
row = pd.DataFrame({
    "Age": [30],
    "Sex": ["male"],
    "Job": [1],
    "Housing": ["rent"],
    "Saving accounts": ["little"],
    "Checking account": ["moderate"],
    "Credit amount": [2000],
    "Duration": [12],
})
for c in ["Sex", "Housing", "Saving accounts", "Checking account"]:
    row[c] = encoders[c].transform(row[c])

pred = model.predict(row)
proba = model.predict_proba(row)
print(pred, proba)
```

---

## 👨‍💻 Author

Your Name Here  
📧 your.email@example.com  
🔗 LinkedIn/Portfolio

---

## 🌟 Support

If this project helped you:

⭐ Star this repo  
📢 Share it with others  
💬 Open an issue for suggestions or improvements

---

> _“Good ML isn’t only about accuracy — it’s about reliability, clarity, and real-world usability.”_
