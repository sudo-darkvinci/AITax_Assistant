#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Define lists
states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
occupations = ["Software Engineer", "Retail Assistant", "Nurse", "Teacher", "Student Intern",
               "Data Analyst", "Delivery Driver", "Barista", "Accountant", "Construction Worker"]

st.set_page_config(page_title="AITax Assistant", layout="wide")
st.title("🧾 AITax Assistant")
st.markdown("**Explainable & Fair AI Nudges for Australian Tax Deductions**")

# Load model
pipeline = joblib.load('aitax_pipeline.pkl')

st.sidebar.header("Your Details")
occupation = st.sidebar.selectbox("Occupation", occupations)
age = st.sidebar.slider("Age", 18, 67, 33)
state = st.sidebar.selectbox("State", states)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
income = st.sidebar.number_input("Annual Income ($)", 20000, 300000, 65000)

work_ded = st.sidebar.number_input("Work-related deductions", 0.0, 50000.0, 500.0)
edu_ded = st.sidebar.number_input("Education deductions", 0.0, 20000.0, 0.0)
char_ded = st.sidebar.number_input("Charity donations", 0.0, 10000.0, 0.0)
trans_ded = st.sidebar.number_input("Transport deductions", 0.0, 15000.0, 0.0)
other_ded = st.sidebar.number_input("Other deductions", 0.0, 10000.0, 0.0)

input_data = pd.DataFrame([{
    'Occupation': occupation, 
    'Age': age, 
    'State': state, 
    'Gender': gender,
    'Income': income, 
    'Work_Ded': work_ded, 
    'Edu_Ded': edu_ded,
    'Char_Ded': char_ded, 
    'Trans_Ded': trans_ded, 
    'Other_Ded': other_ded,
    'Income_Bracket': pd.cut([income], bins=[0,50000,90000,150000,float('inf')], 
                             labels=['Low','Medium','High','Very High'])[0]
}])

if st.button("🔍 Get AI Nudge & Explanation"):
    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]
    
    if pred == 1:
        st.error(f"**High deduction risk detected** ({prob:.1%})")
        st.markdown("**Nudge:** Your claims appear higher than typical for your occupation. Double-check receipts.")
    else:
        st.success(f"**Low deduction risk** ({prob:.1%})")
        st.markdown("**Nudge:** Your deductions look reasonable.")
    
    # SHAP Waterfall - Fixed with short lines
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    processed = pipeline.named_steps['preprocessor'].transform(input_data)
    shap_values = explainer.shap_values(processed)
    
    fig = plt.figure(figsize=(10, 6))
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=processed[0],
        feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out()
    )
    shap.waterfall_plot(explanation)
    st.pyplot(fig)
    plt.close()

st.caption("AITax Assistant • Trained on 50,000 synthetic records with SHAP explanations")


# In[ ]:





# In[ ]:




