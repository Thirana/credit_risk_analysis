import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Initialize session state for storing prediction results and input values
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False
if 'probability' not in st.session_state:
    st.session_state.probability = None
if 'credit_score' not in st.session_state:
    st.session_state.credit_score = None
if 'rating' not in st.session_state:
    st.session_state.rating = None

# Set the page configuration and theme
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💵",
    layout="centered"
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
    .nav-button {
        background-color: #2c3e50 !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App title with icon and description
st.title("Credit Risk Analyzer")
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
        Enter the required information below to get a comprehensive risk assessment.
    </div>
""", unsafe_allow_html=True)

# Load the training data for KDE plots and store in session state
try:
    if 'training_data' not in st.session_state:
        model_data = joblib.load('artifacts/model_data.joblib')
        st.session_state.training_data = model_data.get('training_data', None)
    training_data = st.session_state.training_data
except:
    training_data = None
    st.session_state.training_data = None

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

# Monthly Payment Calculator
interest_rate = st.slider('Annual Interest Rate (%)', min_value=5.0, max_value=25.0, value=12.0, step=0.5)
if loan_tenure_months > 0 and loan_amount > 0:
    # Convert annual interest rate to monthly rate
    monthly_rate = interest_rate / (12 * 100)
    # Calculate monthly payment (EMI)
    if monthly_rate > 0:
        emi = loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure_months / ((1 + monthly_rate)**loan_tenure_months - 1)
    else:
        emi = loan_amount / loan_tenure_months
    
    # Calculate what percentage of income this represents
    income_percentage = (emi * 12 / income) * 100 if income > 0 else 0
    
    # Display the EMI and percentage of income
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px;'>
                <p style='margin:0; color: #666;'>Estimated Monthly Payment (EMI)</p>
                <h3 style='margin:0; color: #2c3e50;'>LKR {emi:,.2f}</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px;'>
                <p style='margin:0; color: #666;'>Percentage of Yearly Income</p>
                <h3 style='margin:0; color: {"#ff4444" if income_percentage > 40 else "#00aa00"};'>{income_percentage:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

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

# Store all inputs in session state
st.session_state.age = age
st.session_state.income = income
st.session_state.loan_amount = loan_amount
st.session_state.loan_tenure_months = loan_tenure_months
st.session_state.avg_dpd_per_delinquency = avg_dpd_per_delinquency
st.session_state.loan_to_income_ratio = loan_to_income_ratio
st.session_state.delinquency_ratio = delinquency_ratio
st.session_state.credit_utilization_ratio = credit_utilization_ratio
st.session_state.num_open_accounts = num_open_accounts
st.session_state.residence_type = residence_type
st.session_state.loan_purpose = loan_purpose
st.session_state.loan_type = loan_type
st.session_state.interest_rate = interest_rate

if 'income_percentage' in locals():
    st.session_state.income_percentage = income_percentage
    st.session_state.emi = emi

# Add some space before the button
st.markdown("<br>", unsafe_allow_html=True)

# Button to calculate risk
if st.button('Calculate Risk'):
    # Call the predict function from the helper module
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)
    
    # Store results in session state
    st.session_state.has_predicted = True
    st.session_state.probability = probability
    st.session_state.credit_score = credit_score
    st.session_state.rating = rating

# Display results if prediction has been made
if st.session_state.has_predicted:
    probability = st.session_state.probability
    credit_score = st.session_state.credit_score
    rating = st.session_state.rating

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
    
    # Risk Improvement Suggestions
    st.subheader("Risk Improvement Suggestions")
    
    # Create suggestions based on the input values
    suggestions = []
    
    # Check loan to income ratio
    if loan_to_income_ratio > 2.0:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>High Loan-to-Income Ratio:</strong> Your loan amount is {:.1f}x your yearly income. 
                Consider reducing your loan amount or increasing your income before applying.
            </div>
        """.format(loan_to_income_ratio))
    
    # Check credit utilization
    if credit_utilization_ratio > 50:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>High Credit Utilization:</strong> Your credit utilization ratio of {}% is above the recommended level. 
                Try to reduce this to below 30% to improve your credit score.
            </div>
        """.format(credit_utilization_ratio))
    
    # Check delinquency ratio
    if delinquency_ratio > 20:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>Elevated Delinquency Ratio:</strong> Your delinquency ratio of {}% suggests a history of late payments.
                Focus on making timely payments for at least 6-12 months to improve this metric.
            </div>
        """.format(delinquency_ratio))
    
    # Check average DPD
    if avg_dpd_per_delinquency > 15:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>High Days Past Due:</strong> Your average DPD of {} days indicates significant payment delays.
                Setting up automatic payments could help ensure you pay on time.
            </div>
        """.format(avg_dpd_per_delinquency))
    
    # Check number of open accounts
    if num_open_accounts > 3:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>Multiple Open Accounts:</strong> Having {} open loan accounts may be seen as a risk.
                Consider paying off some smaller loans before applying for new credit.
            </div>
        """.format(num_open_accounts))
    
    # EMI percentage of income
    if 'income_percentage' in locals() and income_percentage > 40:
        suggestions.append("""
            <div style='margin-bottom: 0.5rem;'>
                <span style='color: #ff4444;'>●</span> <strong>High Debt-to-Income Ratio:</strong> Your monthly loan payment would be {:.2f}% of your monthly income.
                Financial experts recommend keeping this below 40% to maintain financial health.
            </div>
        """.format(income_percentage))
        
    # Display suggestions
    if suggestions:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                The following suggestions may help improve your credit risk profile:
            </div>
        """, unsafe_allow_html=True)
        
        for suggestion in suggestions:
            st.markdown(suggestion, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                <span style='color: #00aa00;'>✓</span> Your credit profile looks good! Continue maintaining your current financial habits.
            </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons to other pages (these become visible only after prediction)
    st.markdown("---")
    st.markdown("### Explore Further Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <a href="/what_if_analysis" target="_self">
                <button style="
                    width: 100%;
                    background-color: #2c3e50;
                    color: white;
                    padding: 0.75rem;
                    border-radius: 5px;
                    border: none;
                    margin-top: 1rem;
                    cursor: pointer;
                    font-weight: bold;
                ">
                    🔍 Try What-If Analysis
                </button>
            </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <a href="/feature_distribution" target="_self">
                <button style="
                    width: 100%;
                    background-color: #2c3e50;
                    color: white;
                    padding: 0.75rem;
                    border-radius: 5px;
                    border: none;
                    margin-top: 1rem;
                    cursor: pointer;
                    font-weight: bold;
                ">
                    📊 View Feature Distributions
                </button>
            </a>
        """, unsafe_allow_html=True)

# Footer
# st.markdown('_Project From Codebasics ML Course_')
