import streamlit as st
from prediction_helper import predict
import pandas as pd
import numpy as np

# Set the page configuration and theme
st.set_page_config(
    page_title="What-If Analysis | Credit Risk",
    page_icon="🔍"
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

# Page title
st.title("🔍 What-If Analysis")

# Check if prediction has been made
if 'has_predicted' not in st.session_state or not st.session_state.has_predicted:
    st.warning("Please make a risk assessment on the main page first.")
    st.markdown("""
        <a href="/" target="_self">
            <button style="
                background-color: #2c3e50;
                color: white;
                padding: 0.75rem;
                border-radius: 5px;
                border: none;
                margin-top: 1rem;
                cursor: pointer;
                font-weight: bold;
                width: 200px;
            ">
                ◀ Go to Main Page
            </button>
        </a>
    """, unsafe_allow_html=True)
else:
    # Get values from session state
    age = st.session_state.age
    income = st.session_state.income
    loan_amount = st.session_state.loan_amount
    loan_tenure_months = st.session_state.loan_tenure_months
    avg_dpd_per_delinquency = st.session_state.avg_dpd_per_delinquency
    loan_to_income_ratio = st.session_state.loan_to_income_ratio
    delinquency_ratio = st.session_state.delinquency_ratio
    credit_utilization_ratio = st.session_state.credit_utilization_ratio
    num_open_accounts = st.session_state.num_open_accounts
    residence_type = st.session_state.residence_type
    loan_purpose = st.session_state.loan_purpose
    loan_type = st.session_state.loan_type
    
    # Get original prediction results
    probability = st.session_state.probability
    credit_score = st.session_state.credit_score
    rating = st.session_state.rating
    
    # Introduction
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
            Adjust the sliders below to see how changing various factors would affect your credit risk assessment.
            This analysis helps you understand which factors have the most significant impact on your credit score.
        </div>
    """, unsafe_allow_html=True)
    
    # Create columns for the what-if analysis
    whatif_col1, whatif_col2 = st.columns(2)
    
    with whatif_col1:
        # Create sliders for key parameters
        st.markdown("<p style='margin-bottom: 0.3rem;'><strong>Credit Utilization Ratio (%)</strong></p>", unsafe_allow_html=True)
        whatif_credit_util = st.slider("Credit Utilization Ratio", min_value=0, max_value=100, value=int(credit_utilization_ratio), step=5, key="whatif_credit_util", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom: 0.3rem; margin-top: 1rem;'><strong>Delinquency Ratio (%)</strong></p>", unsafe_allow_html=True)
        whatif_delinquency = st.slider("Delinquency Ratio", min_value=0, max_value=100, value=int(delinquency_ratio), step=5, key="whatif_delinquency", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom: 0.3rem; margin-top: 1rem;'><strong>Average Days Past Due</strong></p>", unsafe_allow_html=True)
        whatif_avg_dpd = st.slider("Average Days Past Due", min_value=0, max_value=60, value=int(avg_dpd_per_delinquency), step=5, key="whatif_avg_dpd", label_visibility="collapsed")
    
    with whatif_col2:
        st.markdown("<p style='margin-bottom: 0.3rem;'><strong>Loan Amount (LKR)</strong></p>", unsafe_allow_html=True)
        whatif_loan_amount = st.slider("Loan Amount", min_value=0, max_value=5000000, value=int(loan_amount), step=100000, key="whatif_loan_amount", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom: 0.3rem; margin-top: 1rem;'><strong>Loan Tenure (months)</strong></p>", unsafe_allow_html=True)
        whatif_loan_tenure = st.slider("Loan Tenure", min_value=12, max_value=60, value=int(loan_tenure_months), step=6, key="whatif_loan_tenure", label_visibility="collapsed")
        
        st.markdown("<p style='margin-bottom: 0.3rem; margin-top: 1rem;'><strong>Number of Open Accounts</strong></p>", unsafe_allow_html=True)
        whatif_open_accounts = st.slider("Number of Open Accounts", min_value=1, max_value=4, value=int(num_open_accounts), key="whatif_open_accounts", label_visibility="collapsed")
    
    # Calculate what-if loan to income ratio
    whatif_loan_to_income = whatif_loan_amount / income if income > 0 else 0
    
    # Calculate what-if prediction
    whatif_probability, whatif_credit_score, whatif_rating = predict(
        age, income, whatif_loan_amount, whatif_loan_tenure, whatif_avg_dpd,
        whatif_delinquency, whatif_credit_util, whatif_open_accounts,
        residence_type, loan_purpose, loan_type
    )
    
    # Display comparison results
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Impact on Your Credit Assessment</h4>", unsafe_allow_html=True)
    
    # Create two columns for comparison
    compare_col1, compare_col2 = st.columns(2)
    
    with compare_col1:
        st.markdown("<h5 style='text-align: center;'>Current Assessment</h5>", unsafe_allow_html=True)
        
        # Credit Score - Current
        if credit_score >= 750:
            cs_color = "#00aa00"  # Green for excellent
        elif credit_score >= 650:
            cs_color = "#88aa00"  # Yellow-green for good
        elif credit_score >= 500:
            cs_color = "#ffaa00"  # Orange for average
        else:
            cs_color = "#ff4444"  # Red for poor
            
        # Default Probability - Current
        dp_color = "#ff4444" if probability > 0.3 else "#00aa00"
        
        st.markdown(f"""
        <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Credit Score</p>
                <h3 style='margin:0; color: {cs_color};'>{credit_score}</h3>
            </div>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Default Probability</p>
                <h3 style='margin:0; color: {dp_color};'>{probability:.2%}</h3>
            </div>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Rating</p>
                <h3 style='margin:0;'>{rating}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with compare_col2:
        st.markdown("<h5 style='text-align: center;'>What-If Assessment</h5>", unsafe_allow_html=True)
        
        # Credit Score - What-If
        if whatif_credit_score >= 750:
            whatif_cs_color = "#00aa00"  # Green for excellent
        elif whatif_credit_score >= 650:
            whatif_cs_color = "#88aa00"  # Yellow-green for good
        elif whatif_credit_score >= 500:
            whatif_cs_color = "#ffaa00"  # Orange for average
        else:
            whatif_cs_color = "#ff4444"  # Red for poor
            
        # Default Probability - What-If
        whatif_dp_color = "#ff4444" if whatif_probability > 0.3 else "#00aa00"
        
        # Calculate the difference for arrows
        cs_diff = whatif_credit_score - credit_score
        dp_diff = whatif_probability - probability
        
        cs_arrow = "↑" if cs_diff > 0 else "↓" if cs_diff < 0 else "→"
        dp_arrow = "↑" if dp_diff > 0 else "↓" if dp_diff < 0 else "→"
        
        cs_arrow_color = "#00aa00" if cs_diff > 0 else "#ff4444" if cs_diff < 0 else "#666666"
        dp_arrow_color = "#ff4444" if dp_diff > 0 else "#00aa00" if dp_diff < 0 else "#666666"
        
        st.markdown(f"""
        <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Credit Score</p>
                <h3 style='margin:0; color: {whatif_cs_color};'>
                    {whatif_credit_score} <span style='color: {cs_arrow_color};'>{cs_arrow} {abs(cs_diff)}</span>
                </h3>
            </div>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Default Probability</p>
                <h3 style='margin:0; color: {whatif_dp_color};'>
                    {whatif_probability:.2%} <span style='color: {dp_arrow_color};'>{dp_arrow} {abs(dp_diff):.2%}</span>
                </h3>
            </div>
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                <p style='margin:0; color: #666;'>Rating</p>
                <h3 style='margin:0;'>{whatif_rating}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Monthly Payment Calculation (if applicable)
    if 'interest_rate' in st.session_state:
        interest_rate = st.session_state.interest_rate
        
        # Convert annual interest rate to monthly rate
        monthly_rate = interest_rate / (12 * 100)
        
        # Calculate monthly payment (EMI) for both current and what-if scenarios
        if monthly_rate > 0:
            current_emi = loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure_months / ((1 + monthly_rate)**loan_tenure_months - 1)
            whatif_emi = whatif_loan_amount * monthly_rate * (1 + monthly_rate)**whatif_loan_tenure / ((1 + monthly_rate)**whatif_loan_tenure - 1)
        else:
            current_emi = loan_amount / loan_tenure_months
            whatif_emi = whatif_loan_amount / whatif_loan_tenure
        
        # Calculate income percentages
        current_income_pct = (current_emi * 12 / income) * 100 if income > 0 else 0
        whatif_income_pct = (whatif_emi * 12 / income) * 100 if income > 0 else 0
        
        # Show EMI information
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Monthly Payment Impact</h4>", unsafe_allow_html=True)
        
        emi_col1, emi_col2 = st.columns(2)
        
        with emi_col1:
            st.markdown("<h5 style='text-align: center;'>Current Payment</h5>", unsafe_allow_html=True)
            pct_color = "#ff4444" if current_income_pct > 40 else "#00aa00"
            
            st.markdown(f"""
            <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                    <p style='margin:0; color: #666;'>Monthly Payment</p>
                    <h3 style='margin:0; color: #2c3e50;'>LKR {current_emi:,.2f}</h3>
                </div>
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                    <p style='margin:0; color: #666;'>% of Monthly Income</p>
                    <h3 style='margin:0; color: {pct_color};'>{current_income_pct:.2f}%</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with emi_col2:
            st.markdown("<h5 style='text-align: center;'>What-If Payment</h5>", unsafe_allow_html=True)
            whatif_pct_color = "#ff4444" if whatif_income_pct > 40 else "#00aa00"
            
            # Calculate differences
            emi_diff = whatif_emi - current_emi
            emi_pct_diff = whatif_income_pct - current_income_pct
            
            emi_arrow = "↑" if emi_diff > 0 else "↓" if emi_diff < 0 else "→"
            pct_arrow = "↑" if emi_pct_diff > 0 else "↓" if emi_pct_diff < 0 else "→"
            
            emi_arrow_color = "#ff4444" if emi_diff > 0 else "#00aa00" if emi_diff < 0 else "#666666"
            pct_arrow_color = "#ff4444" if emi_pct_diff > 0 else "#00aa00" if emi_pct_diff < 0 else "#666666"
            
            st.markdown(f"""
            <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                    <p style='margin:0; color: #666;'>Monthly Payment</p>
                    <h3 style='margin:0; color: #2c3e50;'>
                        LKR {whatif_emi:,.2f} <span style='color: {emi_arrow_color};'>{emi_arrow} {abs(emi_diff):,.2f}</span>
                    </h3>
                </div>
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
                    <p style='margin:0; color: #666;'>% of Monthly Income</p>
                    <h3 style='margin:0; color: {whatif_pct_color};'>
                        {whatif_income_pct:.2f}% <span style='color: {pct_arrow_color};'>{pct_arrow} {abs(emi_pct_diff):.2f}%</span>
                    </h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Back to main page button
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <a href="/" target="_self">
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
                    ◀ Back to Main Page
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
                    Go to Feature Distributions ▶
                </button>
            </a>
        """, unsafe_allow_html=True) 