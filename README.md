# 🚕 NYC Taxi Trip Duration Prediction

This project predicts the duration of taxi rides in New York City using pickup/dropoff coordinates, passenger count, and trip distance. It demonstrates a complete data science workflow—from data preprocessing and feature engineering to model training and deployment using Streamlit.

---

## 📦 Features

- Input pickup & dropoff coordinates  
- Specify number of passengers  
- Real-time trip duration prediction  
- Interactive map visualization using **pydeck**

---

## 📊 Dataset

Trained on the [NYC Taxi Trip Duration dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) from Kaggle.  
**Note:** The dataset is not included in this repository. Please download it manually from the Kaggle link above.

---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
```

---

## 🚀 Usage

Launch the Streamlit app locally:

```bash
streamlit run app/nyc_taxi_trip.py
```

Open your browser and enter trip details to get a predicted duration.

---

## 📥 Clone This Repository

To get a local copy of this project, run:

```bash
git clone https://github.com/Shiv-23-art/nyc-taxi-trip-duration.git
```

---

## 🧠 Model

- Algorithm: **Random Forest Regressor**  
- Features: Latitude/longitude, passenger count, calculated distance & time metrics  
- Output: Predicted trip duration in seconds  
- Saved model: `random_forest_model.pkl`  
**Note:** The trained model file is excluded from this repository via `.gitignore`.

---

## 📁 Project Structure

```
nyc-taxi-trip-duration/
├── app/
│   ├── nyc_taxi_trip.py          # Streamlit app script
│   └── random_forest_model.pkl   # Trained model file (excluded from repo via .gitignore)
├── notebooks/
│   └── nyc_taxi_trip.ipynb       # Jupyter notebook for exploration and model training
├── .gitignore                    # Specifies files to exclude from version control
└── requirements.txt              # Python dependencies for the project
```

---

## 🌐 Live Demo

Try the app here: [nyc-taxi-trip-duration.streamlit.app](https://nyc-taxi-trip-duration.streamlit.app/)

---

## 🔮 Future Enhancements

- Upgrade model with **XGBoost** or **LightGBM**  
- Integrate traffic and weather data for better accuracy  
- Add user authentication and session tracking  
- Host live demo with public access

---

## 🙌 Acknowledgments

Special thanks to Kaggle for the dataset and the open-source community for tools like Streamlit, scikit-learn, and pydeck.

---
