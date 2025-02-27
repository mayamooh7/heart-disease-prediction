
# importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# reading the csv file
file_path = "D:/# RASTUC CLASSES/ML Projects/1. Heart dx prediction/heart.csv"
data = pd.read_csv(file_path)

# a) initial inspection
print('First 5 rows:')
print(data.head())
print('\n\nLast 5 rows:')
print(data.tail())
print('\n\nShape of the DataFrame:')
print(data.shape)

# b) Data types and missing values
print('Data info:')
print(data.info())
print('\nMissing values:')
print(data.isnull().sum())

# c) Descriptive stats
data.describe()

# to count missing values per column
data.isnull().sum()

# feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. prepare data
x = data.drop('target', axis=1) # Features (everything except the target)
y = data['target']              # Target variable

# 2. Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=30)

# 3. Create a Random Forest model
model = RandomForestClassifier(random_state=30)

# 4. Train the model
# The model learns the relationship between x_train and y_train by creating multiple decision trees based on the data
model.fit(x_train, y_train)

# 5. Get feature importances
importances = model.feature_importances_

# 6. Print the feature importances
feature_importances = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False).set_index('Feature')
print(feature_importances)



# Calculating the correlation matrix

correlation_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the correlation matrix and proper annotations
sns.heatmap(correlation_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .8})

plt.title('Correlation Matrix')
plt.show()

# import remaining libraries

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train the model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)

# Make predictions
y_pred = logistic_model.predict(x_test)

# Evaluate the model
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# N.B Precision aims to lower the false positives and recall(sensitivity) lowers false negatives
# F1 score balances precision and recall
# Create and train the model
random_forest_model = RandomForestClassifier(random_state=30)
random_forest_model.fit(x_train, y_train)

# Make predictions
y_pred_rf = random_forest_model.predict(x_test)

# Evaluate the model
print("Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# Create and train the model
knn_model = KNeighborsClassifier (n_neighbors=10)  # You can adjust n_neighbors as needed
knn_model.fit(x_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(x_test)

# Evaluate the model
print("K-Nearest Neighbors Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

import joblib
joblib_path = "D:\# RASTUC CLASSES\ML Projects\1. Heart dx prediction\heart.csv\maryam.pkl"

joblib.dump(model, 'maryam.pkl')

import streamlit as st

# Load the trained model
trained_model = joblib.load('maryam.pkl')


# Streamlit UI
st.title("Heart Disease Predictor")
st.write("Enter individual's details to predict heart disease.")

# User inputs
age = st.number_input("Enter your age")
sex = st.selectbox("Select your sex", ["Male", "Female"])
cholesterol = st.number_input("Enter your cholesterol level (mg/dl)")
cp = st.selectbox("How would you describe your chest pain?", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
thalach = st.number_input("Enter your maximum heart rate achieved during exercise (bpm)")
ca = st.number_input("How many major blood vessels are colored by fluoroscopy? (0-3)", min_value=0, max_value=3, step=1)
resting_bp_systolic = st.number_input("Resting systolic blood pressure (mm Hg)")
fbs = st.number_input("Enter your fasting blood sugar (mg/dl)")
rest_ecg = st.number_input("Enter your resting electrocardiographic results (0-2)", min_value=0, max_value=2, step=1)
exang = st.number_input("Do you experience exercise-induced chest pain? (0 = no, 1 = yes)")
oldpeak = st.number_input("Enter your ST depression induced by exercise relative to rest")
slope = st.number_input("Enter the slope of the peak exercise ST segment (0-2)", min_value=0, max_value=2, step=1)
thal = st.number_input("Enter your thalassemia type (0-2)", min_value=0, max_value=2, step=1)

# Map categorical values to numerical values
cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 0}
sex_map = {"Male": 0, "Female": 1}


if st.button("Predict"):
    # Create a DataFrame with input values
    input_data = pd.DataFrame([[ca, cp_map[cp], cholesterol, age, sex_map[sex], thalach, resting_bp_systolic, fbs, rest_ecg, exang, oldpeak, slope, thal]],)
    
    # Make predictions using your loaded model
    prediction = trained_model.predict(input_data)
    st.write("Prediction:", prediction)


