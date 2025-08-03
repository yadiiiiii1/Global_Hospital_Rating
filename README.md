# 🌍 Global Hospital Rating - Data Cleaning & Analysis

This project analyzes hospital survey data to compute star ratings based on various quality indicators. It performs data cleaning, exploratory data analysis, visualization, and applies a Decision Tree classifier to predict hospital performance.

---

## 📁 Project Structure

Global_Hospital_Rating/
├── Data/
│   └── hospital_survey_data_50.csv # Raw hospital survey data
│
├── Notebooks/
│   └── hospital_data_cleaning.py # Python script for data processing and analysis
│
├── README.md # Project description and instructions
└── requirements.txt # List of required Python packages

---

## 📌 Objectives

- Clean and validate hospital survey data
- Calculate average quality scores and assign star ratings
- Visualize score distributions and country-wise performance
- Identify key features influencing hospital ratings
- Build a simple decision tree classifier for predictive analysis

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yadiiiiii1/Global_Hospital_Rating
cd Global_Hospital_Rating

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

cd Notebooks
python hospital_data_cleaning.py
```

## 📊 Features

- **Data Cleaning:** Handles missing values, invalid entries, and standardizes scores between 0–100.
- **Rating Assignment:** Converts average scores to 0–5 star ratings.
- **Visualizations:** Histograms, bar plots, boxplots by country.
- **Machine Learning:** Trains a Decision Tree Classifier to predict star ratings.
- **Feature Importance:** Identifies which survey metrics matter most.


## ✅ Requirements

- **pandas:** Data manipulation and analysis  
- **matplotlib:** Data visualization  
- **seaborn:** Statistical data visualization  
- **scikit-learn:** Machine learning and predictive modeling  

All dependencies are listed in **requirements.txt**.

---

## 🧪 Sample Output

- **Cleaned Data Summary:** Overview of cleaned dataset  
- **Top and Bottom 5 Hospitals:** Ranked hospital listings  
- **Visualizations:** Histograms, bar plots, boxplots  
- **Classification Report & Confusion Matrix:** Model performance metrics  
- **Feature Importance Plot:** Key features influencing predictions  


