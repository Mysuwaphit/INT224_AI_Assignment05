import pickle
import pandas as pd
import streamlit as st

# Load model and encoders
with open('logistic_employee.pkl', 'rb') as file:
    model, gender_encoder, education_encoder, city_encoder, ever_benched_encoder, payment_encoder = pickle.load(file)

# Create a dictionary to map radio button choices to numeric values
education_mapping = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
city_mapping = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
gender_mapping = {'Male': 0, 'Female': 1}
ever_benched_mapping = {'Yes': 1, 'No': 0}

# Create Streamlit app
st.title('Employee Payment Tier Prediction')

# Get user input for each variable
new_education = st.radio('Education', list(education_mapping.keys()))
new_joining_year = st.number_input('Joining Year', min_value=1900, max_value=2100)
new_city = st.radio('City', list(city_mapping.keys()))
new_payment_tier = st.number_input('Payment Tier (1-3)', min_value=1)
new_age = st.number_input('Age', min_value=0)
new_gender = st.radio('Gender', list(gender_mapping.keys()))
new_ever_benched = st.radio('Ever Benched', list(ever_benched_mapping.keys()))
new_experience = st.number_input('Experience in Current Domain', min_value=0)
new_leave_or_not = st.radio('Leave or Not', [0, 1])

# Encode categorical features
x_new = pd.DataFrame({
    'Education': [education_mapping[new_education]],
    'JoiningYear': [new_joining_year],
    'City': [city_mapping[new_city]],
    'PaymentTier': [new_payment_tier],
    'Age': [new_age],
    'Gender': [gender_mapping[new_gender]],
    'EverBenched': [ever_benched_mapping[new_ever_benched]],
    'ExperienceInCurrentDomain': [new_experience],
    'LeaveOrNot': [new_leave_or_not]
})

y_pred_new = model.predict(x_new)

# In the Streamlit app, display the predicted Payment Tier
predicted_payment_tier = payment_encoder.inverse_transform(y_pred_new)[0]

st.subheader('Predicted Payment Tier:')
st.write(predicted_payment_tier)
