import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set the page configuration and title
st.set_page_config(page_title="Lauki Finance: Credit Risk Modelling", page_icon="📊")
st.title("Lauki Finance: Credit Risk Modelling")

# Load the training data for KDE plots
# This assumes your model data contains the training data or access to it
try:
    model_data = joblib.load('artifacts/model_data.joblib')
    training_data = model_data.get('training_data', None)
except:
    training_data = None

# Create rows of three columns each
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# Assign inputs to the first row with default values
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)

# Calculate Loan to Income Ratio and display it
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")  # Display as a text field

# Assign inputs to the remaining controls
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)


with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])


# Button to calculate risk
if st.button('Calculate Risk'):
    # Call the predict function from the helper module
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    # Display the results
    st.write(f"Deafult Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")
    
    # Create influence graphs for numerical inputs
    st.subheader("Feature Distribution Analysis")
    
    # Function to create KDE plots showing the distribution for defaulters and non-defaulters
    def create_kde_plot(feature_name, current_value, feature_label=None):
        if training_data is not None:
            # Create a KDE plot using matplotlib/seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot KDE for defaulters and non-defaulters
            sns.kdeplot(data=training_data[training_data['default'] == 0], 
                        x=feature_name, fill=True, color='blue', alpha=0.5, 
                        label='default=0', ax=ax)
            sns.kdeplot(data=training_data[training_data['default'] == 1], 
                        x=feature_name, fill=True, color='orange', alpha=0.5, 
                        label='default=1', ax=ax)
            
            # Add vertical line for current value
            plt.axvline(x=current_value, color='red', linestyle='--', 
                       label=f'Current Value: {current_value}')
            
            # Set labels and title
            plt.xlabel(feature_label or feature_name)
            plt.ylabel('Density')
            plt.title(f'{feature_label or feature_name} KDE Plot with Hue by default')
            plt.legend()
            
            return fig
        else:
            # If no training data is available, use simulated data to demonstrate
            # Create simulated distributions based on the current value
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate synthetic distributions around the current value
            # Non-defaulters are centered more on lower values of risk factors
            x = np.linspace(max(0, current_value * 0.5), current_value * 2, 1000)
            
            # Non-defaulter distribution (centered on lower values)
            y_non_default = np.exp(-0.5 * ((x - current_value * 0.8) / (current_value * 0.3)) ** 2)
            
            # Defaulter distribution (centered on higher values)
            y_default = np.exp(-0.5 * ((x - current_value * 1.5) / (current_value * 0.4)) ** 2)
            
            # Normalize
            y_non_default = y_non_default / np.sum(y_non_default)
            y_default = y_default / np.sum(y_default)
            
            # Plot KDE curves
            plt.fill_between(x, y_non_default, alpha=0.5, color='blue', label='default=0')
            plt.fill_between(x, y_default, alpha=0.5, color='orange', label='default=1')
            
            # Add vertical line for current value
            plt.axvline(x=current_value, color='red', linestyle='--', 
                       label=f'Current Value: {current_value}')
            
            # Set labels and title
            plt.xlabel(feature_label or feature_name)
            plt.ylabel('Density')
            plt.title(f'{feature_label or feature_name} KDE Plot with Hue by default')
            plt.legend()
            
            return fig
    
    # Create base parameters dict for reference
    base_params = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        "delinquency_ratio": delinquency_ratio,
        "credit_utilization_ratio": credit_utilization_ratio,
        "num_open_accounts": num_open_accounts,
        "residence_type": residence_type,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type
    }
    
    # Create distribution plots for each numerical feature
    with st.expander("Age Distribution", expanded=False):
        st.pyplot(create_kde_plot("age", age, "Age"))
    
    with st.expander("Loan to Income Ratio Distribution", expanded=False):
        st.pyplot(create_kde_plot("loan_to_income", loan_to_income_ratio, "Loan to Income Ratio"))
    
    with st.expander("Loan Tenure Distribution", expanded=False):
        st.pyplot(create_kde_plot("loan_tenure_months", loan_tenure_months, "Loan Tenure (months)"))
    
    with st.expander("Average DPD Distribution", expanded=False):
        st.pyplot(create_kde_plot("avg_dpd_per_delinquency", avg_dpd_per_delinquency, "Average DPD"))
    
    with st.expander("Delinquency Ratio Distribution", expanded=False):
        st.pyplot(create_kde_plot("delinquency_ratio", delinquency_ratio, "Delinquency Ratio"))
    
    with st.expander("Credit Utilization Ratio Distribution", expanded=False):
        st.pyplot(create_kde_plot("credit_utilization_ratio", credit_utilization_ratio, "Credit Utilization Ratio"))
    
    with st.expander("Number of Open Accounts Distribution", expanded=False):
        st.pyplot(create_kde_plot("number_of_open_accounts", num_open_accounts, "Number of Open Accounts"))

# Footer
# st.markdown('_Project From Codebasics ML Course_')
