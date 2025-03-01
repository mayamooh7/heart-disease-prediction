import streamlit as st
import joblib 

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