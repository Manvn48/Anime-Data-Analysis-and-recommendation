# 🚀 Multi-Model Anime Recommendation System

Welcome to the **Multi-Model Anime Recommendation System**! This project is a full-stack, production-ready machine learning web application that allows users to seamlessly switch between three powerful recommendation algorithms to discover their next favorite anime.

---

## 🌟 Features

- **Interactive UI**: A stunning, modern, glassmorphism-styled dashboard built with HTML, CSS, and Vanilla JavaScript.
- **Lightning Fast**: Designed to run seamlessly on free-tier hosting (like Render) with pre-computed models optimizing 512MB RAM constraints.
- **Multi-Model Comparison**: Users can dynamically switch between three distinct algorithms from the UI dropdown:
  1. 🤖 **TF-IDF (Content-Based)**: Recommends anime based on textual similarity (Genres, Type).
  2. 👥 **SVD Matrix Factorization (Collaborative Filtering)**: Recommends based on patterns found in 7.8 million user ratings.
  3. 🚀 **XGBoost (Advanced Hybrid Feature Transformation)**: Uses XGBoost decision tree leaves as dense embeddings to compute non-linear similarities between anime!

---

## 🏗️ Architecture

Because Collaborative Filtering on 7.8 million rows requires significant computational power, this project utilizes an intelligent **Offline Pre-Training** architecture:

1. **Offline Training Scripts (`train_cf.py` & `train_xgb.py`)**: These scripts process the massive dataset locally and export the optimized intelligence into lightweight JSON files (`cf_model.json` & `xgb_model.json`).
2. **Live FastAPI Backend (`main.py` & `recommender.py`)**: The server loads the JSON files on startup, computing real-time recommendations in milliseconds without overloading memory.
3. **Frontend (`static/`)**: Fetches data from the API and instantly updates the DOM.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, Pandas, Numpy
- **Machine Learning**: Scikit-Learn (TF-IDF, TruncatedSVD), XGBoost
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

---

## 🚀 Step-by-Step Initialization (Run Locally)

If you'd like to run this application on your own machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Manvn48/Anime-Data-Analysis-and-recommendation.git
cd Anime-Data-Analysis-and-recommendation
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Environment
- **Windows**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Start the Server
```bash
uvicorn main:app --reload
```
Once started, open your browser and navigate to **`http://localhost:8000`** to interact with the app!

*(Note: The offline models `cf_model.json` and `xgb_model.json` are already pre-generated and included in the repository. You do not need to run the `train_cf.py` or `train_xgb.py` scripts unless you download the massive `rating.csv` dataset and wish to re-train the models from scratch!)*

---

## ☁️ Deployment

This project is fully configured for 1-click deployment on **Render.com**.

1. Create a free account on [Render.com](https://render.com/).
2. Click **New** -> **Blueprint**.
3. Connect your GitHub account and select this repository.
4. Render will read the `render.yaml` configuration file and deploy the web service automatically!

---
*Built with ❤️ and FastAPI.*
