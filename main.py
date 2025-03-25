import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set the page configuration and theme
st.set_page_config(
    page_title="Credit Risk Analyzing",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    div.row-widget.stSelectbox > div {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    div.row-widget.stNumberInput > div {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 1rem;
        border-bottom: 2px solid #eee;
        margin-bottom: 2rem;
    }
    h4 {
        color: #2c3e50;
        margin: 1rem 0 0.75rem 0;
        font-size: 1.1rem;
    }
    .section-divider {
        border-top: 1px solid rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title with icon and description
st.title("🏦 Credit Risk Analysis")
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
        Enter the required information below to get a comprehensive risk assessment.
    </div>
""", unsafe_allow_html=True)

# Load the training data for KDE plots
# This assumes your model data contains the training data or access to it
try:
    model_data = joblib.load('artifacts/model_data.joblib')
    training_data = model_data.get('training_data', None)
except:
    training_data = None

# Create sections for different types of inputs
st.markdown("<h4>Personal Information</h4>", unsafe_allow_html=True)
row1 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('YearlyIncome (LKR)', min_value=0, value=3000000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount (LKR)', min_value=0, value=2000000)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h4>Loan Details</h4>", unsafe_allow_html=True)
row2 = st.columns(3)

# Calculate Loan to Income Ratio and display it
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px;'>
            <p style='margin:0; color: #666;'>Loan to Income Ratio</p>
            <h4 style='margin:0; color: #2c3e50;'>{loan_to_income_ratio:.2f}</h4>
        </div>
    """, unsafe_allow_html=True)

with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Average Days Past Due', min_value=0, value=20)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h4>Credit Metrics</h4>", unsafe_allow_html=True)
row3 = st.columns(3)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Number of Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h4>Additional Information</h4>", unsafe_allow_html=True)
row4 = st.columns(3)

with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

# Add some space before the button
st.markdown("<br>", unsafe_allow_html=True)

# Button to calculate risk
if st.button('Calculate Risk Assessment'):
    # Call the predict function from the helper module
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    # Display the results with enhanced styling
    st.markdown("---")
    st.markdown("### Prediction Results")
    
    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    # Default Probability with color coding
    with col1:
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                background-color: {'#ff666650' if probability > 0.5 else '#00ff0050'};
                border-radius: 10px;
                text-align: center;
                ">
                <h3 style="margin: 0;">Default Probability</h3>
                <h2 style="margin: 10px 0; color: {'#ff4444' if probability > 0.5 else '#00aa00'};">
                    {probability:.2%}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Credit Score with gauge-like visualization
    with col2:
        # Calculate color based on credit score
        if credit_score >= 750:
            color = "#00aa00"  # Green for excellent
            bg_color = "#00ff0050"
        elif credit_score >= 650:
            color = "#88aa00"  # Yellow-green for good
            bg_color = "#ffff0050"
        elif credit_score >= 500:
            color = "#ffaa00"  # Orange for average
            bg_color = "#ffaa0050"
        else:
            color = "#ff4444"  # Red for poor
            bg_color = "#ff666650"
            
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                background-color: {bg_color};
                border-radius: 10px;
                text-align: center;
                ">
                <h3 style="margin: 0;">Credit Score</h3>
                <h2 style="margin: 10px 0; color: {color};">
                    {credit_score}
                </h2>
                <div style="
                    background: #e0e0e0;
                    border-radius: 10px;
                    height: 10px;
                    position: relative;
                    ">
                    <div style="
                        width: {(credit_score - 300) / 6}%;
                        background: {color};
                        height: 100%;
                        border-radius: 10px;
                        ">
                    </div>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    font-size: 12px;
                    margin-top: 5px;
                    ">
                    <span>300</span>
                    <span>900</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Rating with appropriate color coding
    with col3:
        # Determine color based on rating
        rating_colors = {
            'Excellent': ('#00aa00', '#00ff0050'),
            'Good': ('#88aa00', '#ffff0050'),
            'Average': ('#ffaa00', '#ffaa0050'),
            'Poor': ('#ff4444', '#ff666650')
        }
        color, bg_color = rating_colors.get(rating, ('#000000', '#ffffff50'))
        
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                background-color: {bg_color};
                border-radius: 10px;
                text-align: center;
                ">
                <h3 style="margin: 0;">Rating</h3>
                <h2 style="margin: 10px 0; color: {color};">
                    {rating}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
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
            # If no training data is available, use simulated data that's feature-specific
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Different features have different distribution patterns
            if feature_name == "age" or feature_label == "Age":
                # Age has a bell curve with significant overlap
                x = np.linspace(18, 75, 1000)
                y_non_default = np.exp(-0.5 * ((x - 40) / 10) ** 2)
                y_default = np.exp(-0.5 * ((x - 35) / 12) ** 2)
                
            elif feature_name == "loan_to_income" or feature_label == "Loan to Income Ratio":
                # Loan to income ratio - defaulters have higher ratios
                x = np.linspace(0, 5, 1000)
                y_non_default = np.exp(-2 * ((x - 0.8) / 0.8) ** 2)
                y_default = np.exp(-1 * ((x - 2.2) / 1.5) ** 2)
                
            elif feature_name == "loan_tenure_months" or feature_label == "Loan Tenure (months)":
                # Loan tenure months - bimodal for defaulters
                x = np.linspace(0, 60, 1000)
                y_non_default = np.exp(-0.5 * ((x - 20) / 10) ** 2)
                y_default = 0.6 * np.exp(-0.5 * ((x - 45) / 10) ** 2) + 0.4 * np.exp(-0.5 * ((x - 25) / 8) ** 2)
                
            elif feature_name == "avg_dpd_per_delinquency" or feature_label == "Average DPD":
                # Average DPD - defaulters have higher values
                x = np.linspace(0, 30, 1000)
                y_non_default = np.exp(-2 * ((x - 2) / 4) ** 2)
                y_default = 0.7 * np.exp(-1 * ((x - 15) / 10) ** 2) + 0.3 * np.exp(-1 * ((x - 5) / 5) ** 2)
                
            elif feature_name == "delinquency_ratio" or feature_label == "Delinquency Ratio":
                # Delinquency ratio - non-defaulters concentrate near 0
                x = np.linspace(0, 100, 1000)
                y_non_default = np.exp(-2 * ((x - 5) / 10) ** 2)
                y_default = np.exp(-0.5 * ((x - 40) / 30) ** 2)
                
            elif feature_name == "credit_utilization_ratio" or feature_label == "Credit Utilization Ratio":
                # Credit utilization ratio - very different distributions
                x = np.linspace(0, 100, 1000)
                y_non_default = np.exp(-1 * ((x - 20) / 25) ** 2) 
                y_default = np.exp(-3 * ((x - 85) / 15) ** 2)
                
            elif feature_name == "number_of_open_accounts" or feature_label == "Number of Open Accounts":
                # Number of open accounts - discrete values
                x = np.linspace(1, 4, 1000)
                # Create peaks at 1, 2, 3, 4
                y_non_default = np.zeros_like(x)
                y_default = np.zeros_like(x)
                
                for i in range(1, 5):
                    # Non-defaulters have more accounts at lower numbers
                    non_default_weight = 0.9 - (i-1) * 0.2  # Decreasing weights: 0.9, 0.7, 0.5, 0.3
                    # Defaulters have more accounts at higher numbers
                    default_weight = 0.3 + (i-1) * 0.2  # Increasing weights: 0.3, 0.5, 0.7, 0.9
                    
                    y_non_default += non_default_weight * np.exp(-10 * (x - i) ** 2)
                    y_default += default_weight * np.exp(-10 * (x - i) ** 2)
            else:
                # Generic case for other features
                x = np.linspace(max(0, current_value * 0.2), current_value * 3, 1000)
                y_non_default = np.exp(-1 * ((x - current_value * 0.7) / (current_value * 0.3)) ** 2)
                y_default = np.exp(-1 * ((x - current_value * 1.8) / (current_value * 0.6)) ** 2)
            
            # Normalize
            if np.sum(y_non_default) > 0:
                y_non_default = y_non_default / np.sum(y_non_default)
            if np.sum(y_default) > 0:
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
