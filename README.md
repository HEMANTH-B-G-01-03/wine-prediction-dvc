# 🍷 Wine Quality Prediction using DVC + AWS S3 + MLOps

A complete MLOps workflow project demonstrating **Data Version Control (DVC)** integrated with **AWS S3**, **GitHub**, and a **Machine Learning training pipeline** for Wine Quality Prediction.

---

# 🚀 Project Overview

This project demonstrates how modern MLOps workflows manage:

- 📊 Large datasets
- 🤖 Trained ML models
- ☁️ Cloud storage using AWS S3
- 🔄 Dataset and model versioning using DVC
- 🧠 Machine Learning model training
- 🗂️ GitHub metadata tracking

The workflow ensures reproducibility, scalability, and efficient handling of ML artifacts.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Machine Learning |
| DVC | Data & Model Version Control |
| AWS S3 | Remote Cloud Storage |
| Git & GitHub | Code Versioning |
| Scikit-learn | ML Model Training |
| Pandas | Data Processing |

---

# 📂 Project Structure

```bash
wine-prediction-dvc/
│
├── .dvc/
├── data/
│   ├── winequality-white.csv
│   └── winequality-white.csv.dvc
│
├── models/
│   ├── wine_model.pkl
│   └── wine_model.pkl.dvc
│
├── src/
│   └── train.py
│
├── .dvcignore
├── README.md
