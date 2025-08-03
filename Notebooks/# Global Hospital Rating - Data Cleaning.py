# Global Hospital Rating - Data Cleaning and Analysis
# This script performs data loading, cleaning, scoring, visualization, and decision tree classification.

import os
import pandas as pd

# Geçerli Python dosyasının bulunduğu klasörü bul
current_dir = os.path.dirname(os.path.abspath(__file__))

# Veri dosyasının yolu: bir üst klasöre çık, Data klasörüne gir
data_path = os.path.join(current_dir, "..", "Data", "hospital_survey_data_50.csv")

# CSV dosyasını oku
df = pd.read_csv(data_path)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys
import subprocess

# -- Optional: Install packages if not present (better to do this outside script) --
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import sklearn
except ImportError:
    print("scikit-learn is not installed, installing now...")
    install_package("scikit-learn")

# Load dataset
data_path = os.path.join(current_dir, "..", "Data", "hospital_survey_data_50.csv")
df = pd.read_csv(data_path)

# Preview the first 5 rows
print("Data Preview:")
print(df.head())

# Identify score columns (assuming first two columns are 'Hospital' and 'Country')
score_columns = df.columns[2:]  

# Data Cleaning: Fix invalid and missing values in score columns
for col in score_columns:
    # Convert to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')  
    # Clip values to the range 0-100
    df[col] = df[col].clip(lower=0, upper=100)         
    # Fill missing values with the median of the column
    df[col] = df[col].fillna(df[col].median())         

print("\nCleaned Data Summary Statistics:")
print(df.describe())

# Calculate average score across all score columns
df["Average_Score"] = df.iloc[:, 2:].mean(axis=1)

# Function to assign star ratings based on average score
def assign_stars(score):
    if score >= 90:
        return 5
    elif score >= 75:
        return 4
    elif score >= 60:
        return 3
    elif score >= 45:
        return 2
    elif score >= 30:
        return 1
    else:
        return 0

# Apply star rating function
df["Star_Rating"] = df["Average_Score"].apply(assign_stars)

print("\nFirst 10 Hospitals with Ratings:")
print(df[["Hospital", "Country", "Average_Score", "Star_Rating"]].head(10))

# Visualization of star rating distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Star_Rating"], bins=6, kde=False)  # bins=6 for 0-5 star ratings
plt.title("Hospital Star Rating Distribution")
plt.xlabel("Stars")
plt.ylabel("Number of Hospitals")
plt.show()

# Average score by star rating bar plot
df.groupby("Star_Rating")["Average_Score"].mean().plot(kind="bar", title="Average Score by Star Rating")
plt.ylabel("Average Score")
plt.xlabel("Star Rating")
plt.show()

# Top 5 hospitals by average score
print("\nTop 5 Hospitals:")
print(df.sort_values("Average_Score", ascending=False)[["Hospital", "Country", "Average_Score", "Star_Rating"]].head())

# Bottom 5 hospitals by average score
print("\nBottom 5 Hospitals:")
print(df.sort_values("Average_Score")[["Hospital", "Country", "Average_Score", "Star_Rating"]].head())

# Boxplot of average scores by country
plt.figure(figsize=(12, 6))
sns.boxplot(x="Country", y="Average_Score", data=df)
plt.title("Average Hospital Score by Country")
plt.xticks(rotation=45)
plt.show()

# Prepare features (X) and target (y) for machine learning
# Exclude non-numeric columns and calculated columns (Hospital, Country, Average_Score, Star_Rating)
X = df.iloc[:, 2:-2]  
y = df["Star_Rating"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict star ratings on test data
y_pred = model.predict(X_test)

# Print confusion matrix to evaluate classification accuracy
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report with precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance to understand which scores influence the model most
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Score Category")
plt.show()
