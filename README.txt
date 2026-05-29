<div align="center">

# 🎓 Student Performance Prediction
### *Machine Learning · End-to-End MLOps Pipeline · Flask API · Docker*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-189A28?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-FFCC00?style=for-the-badge)](https://catboost.ai/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Predict. Identify. Intervene. Empower.**  
> An end-to-end machine learning system that predicts a student's mathematics performance using socio-demographic and academic features — deployed as a Flask API with a full modular MLOps pipeline and Docker containerization.

[🔍 Explore Notebook](#-notebooks) · [🚀 Quick Start](#️-installation--setup) · [📊 Model Results](#-model-performance) · [🐛 Report Bug](../../issues) · [✨ Request Feature](../../issues)

---

</div>

## 📋 Table of Contents

- [🧠 About The Project](#-about-the-project)
- [🌟 Features](#-features)
- [🏗️ ML Pipeline Architecture](#️-ml-pipeline-architecture)
- [📊 Dataset Overview](#-dataset-overview)
- [🤖 Models & Algorithms](#-models--algorithms)
- [📈 Model Performance](#-model-performance)
- [🔄 Data Pipeline Flow](#-data-pipeline-flow)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🐳 Docker Deployment](#-docker-deployment)
- [🖥️ Web Application Usage](#️-web-application-usage)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🧠 About The Project

**Student Performance Prediction** is a production-grade machine learning project that tackles a critical challenge in education: identifying students at academic risk *before* they fall behind.

Using a rich dataset of student demographics, socioeconomic background, and prior academic history, the system trains multiple regression models and serves real-time predictions through a clean Flask web interface — fully containerized with Docker for easy deployment.

### 💡 Why This Matters

```
The Problem in Education Today
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌  Teachers cannot monitor every student individually
  ❌  At-risk students are identified too late for intervention
  ❌  No data-driven way to personalize academic support
  ❌  Administrators lack predictive insights for planning

What This Project Delivers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅  Predict math scores with high accuracy (R² up to 0.88)
  ✅  Identify at-risk students based on input factors
  ✅  REST API for integration with school systems
  ✅  Modular, production-ready MLOps pipeline
  ✅  One-command Docker deployment
```

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🔀 **Multi-Model Training** | 6 algorithms trained and compared automatically |
| 🏆 **Auto Best-Model Selection** | Best model selected by R² score via hyperparameter tuning |
| 🔧 **Modular Pipeline** | Separate components for ingestion, transformation, and training |
| 🌐 **Flask Web App** | Interactive form-based prediction UI |
| 🐳 **Docker Ready** | One-command containerized deployment |
| 📓 **EDA Notebook** | Full exploratory data analysis with visualizations |
| 🧪 **GridSearchCV Tuning** | Systematic hyperparameter optimization |
| 📦 **Artifact Persistence** | Models and preprocessors saved as `.pkl` artifacts |
| 🚨 **Custom Exception Handling** | Descriptive ML pipeline error messages |
| 📝 **Logging System** | Full pipeline execution logging |

---

## 🏗️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER / CLIENT                               │
│          (Web Browser or API POST /predict)                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FLASK APPLICATION                            │
│                          (app.py)                                   │
│                                                                     │
│   GET  /          →  Render prediction form                         │
│   POST /predict   →  Accept student features → Return prediction    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PREDICTION PIPELINE                            │
│                   (src/pipeline/predict_pipeline.py)                │
│                                                                     │
│   1. Load preprocessor artifact  (preprocessor.pkl)                │
│   2. Transform input features                                       │
│   3. Load best model artifact    (model.pkl)                        │
│   4. Run model.predict()                                            │
│   5. Return predicted math score                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
   ┌───────────────────────┐   ┌───────────────────────────┐
   │   preprocessor.pkl    │   │       model.pkl           │
   │ (StandardScaler +     │   │  (Best of 6 algorithms)   │
   │  OneHotEncoder)       │   │  Selected by R² score     │
   └───────────────────────┘   └───────────────────────────┘
```

---

## 📊 Dataset Overview

The dataset used is the **Students Performance in Exams** dataset, containing records of 1,000 students across 8 features.

### Features

```
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE SUMMARY                            │
├──────────────────────────┬──────────────┬───────────────────────┤
│ Feature                  │ Type         │ Example Values        │
├──────────────────────────┼──────────────┼───────────────────────┤
│ gender                   │ Categorical  │ male / female         │
│ race/ethnicity           │ Categorical  │ group A–E             │
│ parental_education       │ Categorical  │ bachelor's, master's… │
│ lunch                    │ Categorical  │ standard / free/reduced│
│ test_preparation_course  │ Categorical  │ none / completed      │
│ reading_score            │ Numerical    │ 0 – 100               │
│ writing_score            │ Numerical    │ 0 – 100               │
├──────────────────────────┼──────────────┼───────────────────────┤
│ math_score  🎯 TARGET    │ Numerical    │ 0 – 100               │
└──────────────────────────┴──────────────┴───────────────────────┘
```

### Score Distribution (Approximate)

```
Math Score Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0–40   │ ███░░░░░░░░░░░░░░│  ~10% students
40–50   │ █████░░░░░░░░░░░░│  ~15% students
50–60   │ ██████████░░░░░░░│  ~20% students
60–70   │ █████████████░░░░│  ~25% students  ← Most common
70–80   │ ████████████░░░░░│  ~18% students
80–90   │ ██████░░░░░░░░░░░│  ~08% students
90–100  │ ███░░░░░░░░░░░░░░│  ~04% students
        └─────────────────┘
          0               30%
```

### Key Insights from EDA

```
📌 Students who completed test prep scored ~5–8 points higher on average
📌 Standard lunch correlates with ~10 point improvement in math score
📌 Reading and writing scores are strong predictors of math performance
📌 Parental education level shows a positive gradient with student scores
📌 Gender shows small but consistent differences across all three subjects
```

---

## 🤖 Models & Algorithms

Six regression algorithms are trained, tuned, and benchmarked:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ALGORITHM COMPARISON                            │
├─────────────────────────┬──────────────────────────────────────────┤
│  Algorithm              │  Characteristics                         │
├─────────────────────────┼──────────────────────────────────────────┤
│  Linear Regression      │  Baseline, interpretable, fast           │
│  Decision Tree          │  Captures non-linearity, prone to overfit│
│  Random Forest          │  Ensemble, robust, handles noise well     │
│  XGBoost                │  Gradient boosting, high performance      │
│  CatBoost               │  Handles categoricals natively, accurate  │
│  Support Vector Reg.    │  Effective in high-dim spaces             │
└─────────────────────────┴──────────────────────────────────────────┘
```

### Model Selection Strategy

```
                    ┌─────────────────────┐
                    │   Train All Models  │
                    │   (6 algorithms)    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  GridSearchCV with  │
                    │  Cross-Validation   │
                    │   (Hyperparameter   │
                    │     Tuning)         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Evaluate all with  │
                    │    R² Score &       │
                    │    MSE on Test Set  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Select Best Model  │ ──► Save as model.pkl
                    │  (Highest R² Score) │
                    └─────────────────────┘
```

---

## 📈 Model Performance

> Results from training on the student performance dataset (80/20 train-test split):

```
Model Performance Leaderboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rank  Model                R² Score   RMSE
  ────  ───────────────────  ─────────  ──────
   🥇  CatBoost Regressor    ~0.882     ~5.1
   🥈  XGBoost Regressor     ~0.878     ~5.2
   🥉  Random Forest         ~0.855     ~5.7
    4  Linear Regression     ~0.870     ~5.4
    5  SVR                   ~0.849     ~5.9
    6  Decision Tree         ~0.752     ~7.5

  R² Score scale: 0.0 (random) → 1.0 (perfect)
```

### What R² Means Here

```
  R² = 0.88  →  The model explains 88% of the variance in math scores
               leaving only 12% unexplained by the features provided.

  RMSE ≈ 5.1 →  On average, predictions are off by ~5 points out of 100
               which is an excellent result for educational prediction.
```

---

## 🔄 Data Pipeline Flow

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA INGESTION                                             │
│  src/components/data_ingestion.py                                   │
│                                                                     │
│  • Load CSV from source                                             │
│  • Perform train/test split (80% / 20%)                             │
│  • Save train.csv and test.csv to artifacts/                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA TRANSFORMATION                                        │
│  src/components/data_transformation.py                              │
│                                                                     │
│  Numerical Features:                                                │
│    • reading_score, writing_score                                   │
│    • Impute missing → StandardScaler                                │
│                                                                     │
│  Categorical Features:                                              │
│    • gender, race, parental_education, lunch, test_prep             │
│    • Impute missing (most_frequent) → OneHotEncoder                 │
│                                                                     │
│  • Combine via ColumnTransformer                                    │
│  • Save preprocessor.pkl to artifacts/                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: MODEL TRAINING                                             │
│  src/components/model_trainer.py                                    │
│                                                                     │
│  • Train 6 regression models                                        │
│  • Apply GridSearchCV for hyperparameter tuning                     │
│  • Evaluate with R² score                                           │
│  • Select best model                                                │
│  • Save model.pkl to artifacts/                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  artifacts/      │
                    │  ├ model.pkl     │
                    │  ├ preprocessor  │
                    │  │  .pkl         │
                    │  ├ train.csv     │
                    │  └ test.csv      │
                    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Library / Tool | Version | Role |
|---|---|---|---|
| **Language** | Python | 3.8+ | Core language |
| **ML** | Scikit-Learn | 1.x | Pipelines, preprocessing, SVR, RF, DT |
| **ML** | XGBoost | 1.7+ | Gradient boosting regressor |
| **ML** | CatBoost | 1.x | Categorical-friendly boosting |
| **Data** | Pandas | 1.5+ | Data manipulation |
| **Data** | NumPy | 1.23+ | Numerical operations |
| **Viz** | Matplotlib / Seaborn | Latest | EDA visualizations |
| **Web** | Flask | 2.x | REST API & web interface |
| **Container** | Docker | 20.x+ | Containerized deployment |
| **Notebook** | Jupyter | Latest | EDA & experimentation |
| **Tuning** | GridSearchCV | (sklearn) | Hyperparameter optimization |

---

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── 📁 artifacts/                    # Generated during training (auto-created)
│   ├── model.pkl                    # Best trained model
│   ├── preprocessor.pkl             # Fitted data transformer
│   ├── train.csv                    # Training split
│   └── test.csv                     # Test split
│
├── 📁 notebook/                     # Jupyter notebooks for EDA & experiments
│   ├── EDA_Student_Performance.ipynb
│   └── Model_Training_Experiments.ipynb
│
├── 📁 src/                          # Core source code (modular package)
│   ├── 📁 components/
│   │   ├── data_ingestion.py        # Step 1: Load & split data
│   │   ├── data_transformation.py  # Step 2: Preprocess features
│   │   └── model_trainer.py        # Step 3: Train & select best model
│   │
│   ├── 📁 pipeline/
│   │   ├── train_pipeline.py        # Orchestrates training end-to-end
│   │   └── predict_pipeline.py      # Loads artifacts & serves predictions
│   │
│   ├── exception.py                 # Custom exception with traceback info
│   ├── logger.py                    # Logging setup
│   └── utils.py                     # Shared helper functions
│
├── 📁 templates/                    # Flask HTML templates
│   ├── index.html                   # Prediction form UI
│   └── home.html                    # Landing page
│
├── app.py                           # Flask application entry point
├── setup.py                         # Package installation config
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker containerization
└── README.md                        # You are here 📍
```

---

## ⚙️ Installation & Setup

### Prerequisites

```
✅  Python 3.8+       https://www.python.org/downloads/
✅  pip               Included with Python
✅  Git               https://git-scm.com/
✅  Docker (optional) https://www.docker.com/
```

### Local Setup (Without Docker)

**1. Clone the Repository**
```bash
git clone https://github.com/leevanshi/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

**2. Create a Virtual Environment**
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Training Pipeline**
```bash
# Step 1 – Ingest data
python src/components/data_ingestion.py

# Step 2 – Transform features
python src/components/data_transformation.py

# Step 3 – Train & select best model
python src/components/model_trainer.py
```

> After training, check `artifacts/` for `model.pkl` and `preprocessor.pkl`

**5. Launch the Web Application**
```bash
python app.py
```

```
🌐 Open your browser at: http://localhost:5000
```

---

## 🐳 Docker Deployment

Run the entire application with a single command using Docker:

**Build the Docker Image**
```bash
docker build -t student-performance-prediction .
```

**Run the Container**
```bash
docker run -p 5000:5000 student-performance-prediction
```

**Access the App**
```
🌐 http://localhost:5000
```

### Docker Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Container                    │
│                                                  │
│  Base Image: python:3.8-slim                    │
│                                                  │
│  ├── WORKDIR /app                               │
│  ├── COPY requirements.txt                      │
│  ├── RUN pip install -r requirements.txt        │
│  ├── COPY . /app                                │
│  ├── EXPOSE 5000                                │
│  └── CMD ["python", "app.py"]                   │
│                                                  │
│  Port mapping: HOST:5000 → CONTAINER:5000       │
└─────────────────────────────────────────────────┘
```

---

## 🖥️ Web Application Usage

### Prediction Form

```
┌──────────────────────────────────────────────────────────────┐
│  🎓 Student Performance Predictor                            │
│  ──────────────────────────────────────────────────────────  │
│                                                              │
│  Gender             [  Female ▼  ]                          │
│                                                              │
│  Race / Ethnicity   [  Group B ▼  ]                         │
│                                                              │
│  Parental Education [  Bachelor's ▼  ]                      │
│                                                              │
│  Lunch Type         [  Standard ▼  ]                        │
│                                                              │
│  Test Preparation   [  Completed ▼  ]                       │
│                                                              │
│  Reading Score      [ 75      ]                             │
│                                                              │
│  Writing Score      [ 80      ]                             │
│                                                              │
│             [ 🔮 Predict Math Score ]                       │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  Predicted Math Score: 78.4 / 100                           │
└──────────────────────────────────────────────────────────────┘
```

### REST API Usage

You can also call the prediction endpoint directly:

```python
import requests

data = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 75,
    "writing_score": 80
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
# → {"predicted_math_score": 78.4}
```

---

## 📓 Notebooks

The `notebook/` directory contains:

| Notebook | Description |
|---|---|
| `EDA_Student_Performance.ipynb` | Full exploratory data analysis — distributions, correlations, group comparisons |
| `Model_Training_Experiments.ipynb` | Training all 6 models, comparing performance, visualizing results |

### Sample EDA Plots You'll Find

```
  📊  Score distributions by gender
  📊  Correlation heatmap (reading ↔ writing ↔ math)
  📊  Effect of test preparation on score
  📊  Parental education level vs. average score
  📊  Lunch type vs. score distributions
  📊  Feature importance from Random Forest & XGBoost
```

---

## 🔐 Custom Exception & Logging

The project includes production-grade error handling and logging:

```python
# src/exception.py
class CustomException(Exception):
    """Provides file name, line number, and error message
       for every exception raised in the pipeline."""

# src/logger.py
# Creates timestamped log files in /logs/ directory
# Tracks every pipeline step execution
```

```
logs/
├── 05_29_2026_10_30_00.log    ← data ingestion
├── 05_29_2026_10_30_12.log    ← transformation
└── 05_29_2026_10_31_05.log    ← model training
```

---

## 🗺️ Roadmap

```
  ✅  v1.0  Multi-model training with auto best-model selection
  ✅  v1.1  Flask web app with prediction form
  ✅  v1.2  Docker containerization
  📌  v1.3  CI/CD with GitHub Actions                  (Planned)
  📌  v1.4  AWS / GCP Cloud deployment (EC2 / App Engine) (Planned)
  📌  v1.5  Model monitoring & drift detection          (Planned)
  📌  v1.6  Add classification (Pass/Fail/At-Risk)      (Planned)
  📌  v2.0  Multi-subject prediction (Reading, Writing) (Future)
  📌  v2.1  Student recommendation system               (Future)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

```bash
# 1. Fork the Project
# Click "Fork" on GitHub

# 2. Create your Feature Branch
git checkout -b feature/improve-model-accuracy

# 3. Make Changes & Test
python src/components/model_trainer.py

# 4. Commit with a descriptive message
git commit -m "feat: add Ridge Regression to model comparison"

# 5. Push and open a Pull Request
git push origin feature/improve-model-accuracy
```

### Ideas for Contribution

- Add more regression algorithms (Ridge, Lasso, ElasticNet)
- Improve the front-end prediction form UI
- Add unit tests for pipeline components
- Extend to predict reading and writing scores too
- Add SHAP values for model explainability

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👤 Author

<div align="center">

**leevanshi**  
[![GitHub](https://img.shields.io/badge/GitHub-leevanshi-181717?style=for-the-badge&logo=github)](https://github.com/leevanshi)

</div>

---

## 🙏 Acknowledgements

- [UCI ML Repository](https://archive.ics.uci.edu/) — Dataset source
- [Scikit-Learn](https://scikit-learn.org/) — ML backbone
- [XGBoost](https://xgboost.readthedocs.io/) & [CatBoost](https://catboost.ai/) — Boosting algorithms
- [Flask](https://flask.palletsprojects.com/) — Lightweight web framework
- [Docker](https://www.docker.com/) — Container platform

---

<div align="center">

**⭐ If this project helped you, consider giving it a star! ⭐**

Made with 🧠 + ❤️ by [leevanshi](https://github.com/leevanshi)

`predict` → `intervene` → `improve`

</div>
