import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load model and encoders
with open('svm_employee.pkl', 'rb') as file:
    model, gender_encoder, education_encoder, city_encoder, ever_benched_encoder, payment_encoder = pickle.load(file)

# Create Streamlit app
st.title('Employee Payment Tier Prediction')

# Get user input for each variable
new_education = st.radio('Education', ['Bachelors', 'Masters', 'PHD'])
new_joining_year = st.number_input('Joining Year', min_value=1900, max_value=2100)
new_city = st.radio('City', ['Bangalore', 'Pune', 'New Delhi'])
new_age = st.number_input('Age', min_value=0)
new_gender = st.radio('Gender', ['Male', 'Female'])
new_ever_benched = st.radio('Ever Benched', ['Yes', 'No'])
new_experience = st.number_input('Experience in Current Domain', min_value=0)
new_leave_or_not = st.radio('Leave or Not', [0, 1])

# Create a DataFrame with user input
x_new = pd.DataFrame({
    'Education': [new_education],
    'JoiningYear': [new_joining_year],
    'City': [new_city],
    'Age': [new_age],
    'Gender': [new_gender],
    'EverBenched': [new_ever_benched],
    'ExperienceInCurrentDomain': [new_experience],
    'LeaveOrNot': [new_leave_or_not]
})

# Create a "Predict" button
predict_button = st.button('Predict')

if predict_button:
    if not x_new.empty:
        # Encode categorical features
        x_new['Education'] = education_encoder.transform(x_new['Education'])
        x_new['City'] = city_encoder.transform(x_new['City'])
        x_new['Gender'] = gender_encoder.transform(x_new['Gender'])
        x_new['EverBenched'] = ever_benched_encoder.transform(x_new['EverBenched'])
        
        # Scale numerical features
        numerical_features = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain', 'LeaveOrNot']
        x_new[numerical_features] = StandardScaler().fit_transform(x_new[numerical_features])
        
        # Predict Payment Tier
        y_pred_new = model.predict(x_new)
        
        # In the Streamlit app, display the predicted Payment Tier
        predicted_payment_tier = payment_encoder.inverse_transform(y_pred_new)[0]
        
        st.subheader('Predicted Payment Tier:')
        st.write(predicted_payment_tier)
    else:
        st.subheader('No data provided. Please enter values for the required fields.')