# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('📊 Customer Lifetime Value (CLV) Predictor')
st.text('Predict the Customer Lifetime Value (CLV) based on your inputs')

# Menambahkan sidebar dengan styling
st.sidebar.header("Please input your features")
st.sidebar.markdown("Fill in the details below to get the CLV prediction:")

def create_user_input():
    # Numerical Features
    total_claim_amount = st.sidebar.number_input('💸 Total Claim Amount', min_value=0.42, max_value=2552.34, value=100.0)
    income = st.sidebar.number_input('💼 Income', min_value=0, max_value=99934, value=50000)
    monthly_premium_auto = st.sidebar.number_input('🚗 Monthly Premium Auto', min_value=61, max_value=297, value=150)
    number_of_policies = st.sidebar.number_input('📄 Number of Policies', min_value=1, max_value=9, value=2)

    # Categorical Features
    vehicle_class = st.sidebar.selectbox('🚙 Vehicle Class', ['Four-Door Car', 'Two-Door Car', 'SUV', 'Luxury SUV', 'Sports Car', 'Luxury Car'])
    coverage = st.sidebar.selectbox('🔒 Coverage', ['Basic', 'Extended', 'Premium'])
    renew_offer_type = st.sidebar.selectbox('🔄 Renew Offer Type', ['Offer1', 'Offer2', 'Offer3', 'Offer4'])
    employment_status = st.sidebar.selectbox('👔 Employment Status', ['Employed', 'Unemployed', 'Medical Leave', 'Disabled', 'Retired'])
    marital_status = st.sidebar.selectbox('💍 Marital Status', ['Married', 'Divorced', 'Single'])
    education = st.sidebar.selectbox('🎓 Education', ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor'])

    # Creating a dictionary with user input
    user_data = {
        'Total Claim Amount': total_claim_amount,
        'Income': income,
        'Monthly Premium Auto': monthly_premium_auto,
        'Vehicle Class': vehicle_class,
        'Coverage': coverage,
        'Number of Policies' : number_of_policies,
        'Renew Offer Type': renew_offer_type,
        'Employment Status': employment_status,
        'Marital Status': marital_status,
        'Education': education
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat layout yang lebih terstruktur
st.markdown("---")  # Menambahkan garis pemisah

# Membuat 2 kolom dengan lebih estetis
col1, col2 = st.columns([2, 1])

# Kolom kiri untuk input fitur
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Kolom kanan untuk hasil prediksi
with col2:
    st.subheader("🔮 Prediction Result")
    st.markdown("###")  # Garis pemisah untuk visualisasi yang lebih baik
    
    # Load model
    with open('best_model_clv.sav', 'rb') as f:
        model_loaded = pickle.load(f)
    
    # Predict to data
    predicted_clv = model_loaded.predict(data_customer)
    
    st.write(f"**💰 Predicted Customer Lifetime Value (CLV):**")
    st.markdown(f"### 💸 $ **{predicted_clv[0]:.2f}**")
    
    st.markdown("---")  # Garis pemisah
    
    st.info("🔍 The prediction is based on the data you've provided.")
    st.warning("⚠️ Ensure all input fields are accurate to get the best prediction!")

