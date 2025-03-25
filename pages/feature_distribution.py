import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set the page configuration
st.set_page_config(
    page_title="Feature Distributions | Credit Risk",
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
st.title("📊 Feature Distributions")

# Function to create synthetic data for demonstration
def create_synthetic_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = 1000
    
    # Create dataframe with random data following specific distributions
    df = pd.DataFrame()
    
    # Age: defaulters tend to be younger
    df['age'] = np.concatenate([
        np.random.normal(35, 10, int(n_samples * 0.7)),  # non-defaulters
        np.random.normal(28, 8, int(n_samples * 0.3))    # defaulters
    ])
    
    # Loan to Income Ratio: defaulters have higher ratios
    df['loan_to_income_ratio'] = np.concatenate([
        np.random.beta(2, 5, int(n_samples * 0.7)) * 3,   # non-defaulters
        np.random.beta(4, 3, int(n_samples * 0.3)) * 3    # defaulters
    ])
    
    # Loan Tenure: bimodal for defaulters
    df['loan_tenure_months'] = np.concatenate([
        np.random.normal(36, 8, int(n_samples * 0.7)),    # non-defaulters
        np.concatenate([                                   # defaulters (bimodal)
            np.random.normal(24, 5, int(n_samples * 0.15)),
            np.random.normal(48, 5, int(n_samples * 0.15))
        ])
    ])
    
    # Average DPD: higher for defaulters
    df['avg_dpd_per_delinquency'] = np.concatenate([
        np.random.exponential(5, int(n_samples * 0.7)),   # non-defaulters
        np.random.exponential(15, int(n_samples * 0.3))   # defaulters
    ])
    
    # Delinquency Ratio: non-defaulters concentrated near 0
    df['delinquency_ratio'] = np.concatenate([
        np.random.beta(1, 8, int(n_samples * 0.7)) * 100,   # non-defaulters
        np.random.beta(2, 2, int(n_samples * 0.3)) * 100    # defaulters
    ])
    
    # Credit Utilization Ratio
    df['credit_utilization_ratio'] = np.concatenate([
        np.random.beta(2, 3, int(n_samples * 0.7)) * 100,   # non-defaulters
        np.random.beta(4, 2, int(n_samples * 0.3)) * 100    # defaulters
    ])
    
    # Number of Open Accounts: discrete
    df['num_open_accounts'] = np.concatenate([
        np.random.choice([1, 2, 3, 4], int(n_samples * 0.7), p=[0.15, 0.5, 0.25, 0.1]),  # non-defaulters
        np.random.choice([1, 2, 3, 4], int(n_samples * 0.3), p=[0.3, 0.3, 0.3, 0.1])      # defaulters
    ])
    
    # Default status
    df['default'] = np.concatenate([
        np.zeros(int(n_samples * 0.7)),  # non-defaulters
        np.ones(int(n_samples * 0.3))    # defaulters
    ])
    
    return df

# Generate synthetic data
if 'training_data' not in st.session_state or st.session_state.training_data is None:
    try:
        # Try to load model data first (though we know it doesn't have training data)
        model_data = joblib.load("artifacts/model_data.joblib")
        st.session_state.model_loaded = True
        
        # Create synthetic data since model_data doesn't have training data
        df = create_synthetic_data()
        st.session_state.training_data = df
        
        # Show a notice that we're using synthetic data
        st.info("""
            Using synthetic data for visualization purposes. 
            The model data file doesn't contain the training data required for feature distributions.
        """)
    except Exception as e:
        st.session_state.model_loaded = False
        # Create synthetic data as fallback
        df = create_synthetic_data()
        st.session_state.training_data = df
        
        st.warning("""
            Could not load model data file. Using synthetic data for visualization purposes.
            The distributions shown are for demonstration only and may not reflect your actual model.
        """)
else:
    # Use data from session state
    df = st.session_state.training_data

# Function to create KDE plot for a feature
def create_kde_plot(feature_name, current_value=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    defaulters = df[df['default'] == 1]
    non_defaulters = df[df['default'] == 0]
    
    # Define feature-specific visualizations
    if feature_name == "age":
        # Age: Bell curve with significant overlap but defaulters tend to be younger
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "loan_to_income_ratio":
        # Loan to Income Ratio: Higher ratios for defaulters
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "loan_tenure_months":
        # Loan Tenure: Bimodal distribution for defaulters
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "avg_dpd_per_delinquency":
        # Average DPD: Higher values for defaulters
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "delinquency_ratio":
        # Delinquency Ratio: Non-defaulters concentrated near 0
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "credit_utilization_ratio":
        # Credit Utilization Ratio: Distinct distributions
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
    
    elif feature_name == "num_open_accounts":
        # Number of Open Accounts: Discrete distribution
        sns.histplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', 
                     alpha=0.5, kde=True, bins=range(0, 6))
        sns.histplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', 
                     alpha=0.5, kde=True, bins=range(0, 6))
    
    else:
        # Generic case
        sns.kdeplot(non_defaulters[feature_name], ax=ax, color='blue', label='Non-Defaulters', fill=True, alpha=0.3)
        sns.kdeplot(defaulters[feature_name], ax=ax, color='orange', label='Defaulters', fill=True, alpha=0.3)
        
    # Mark the current value if provided
    if current_value is not None:
        ax.axvline(x=current_value, color='red', linestyle='--', linewidth=2, 
                  label=f'Current Value: {current_value:.2f}')
        
    ax.set_title(f'Distribution of {feature_name.replace("_", " ").title()}', fontsize=14)
    ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    return fig

# Check if prediction has been made
if 'has_predicted' not in st.session_state or not st.session_state.has_predicted:
    # Show distributions without current value markers
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
            These plots show the distribution of each feature for defaulters (orange) and non-defaulters (blue).
            <br>
            Make a prediction on the main page to see where your values fall on these distributions.
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different feature categories
    tab1, tab2, tab3 = st.tabs(["Demographics & Loan", "Credit History", "All Features"])
    
    with tab1:
        st.markdown("<h4>Age Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("age"))
        
        st.markdown("<h4>Loan to Income Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("loan_to_income_ratio"))
        
        st.markdown("<h4>Loan Tenure Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("loan_tenure_months"))
    
    with tab2:
        st.markdown("<h4>Credit Utilization Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("credit_utilization_ratio"))
        
        st.markdown("<h4>Delinquency Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("delinquency_ratio"))
        
        st.markdown("<h4>Average Days Past Due Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("avg_dpd_per_delinquency"))
        
        st.markdown("<h4>Number of Open Accounts Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("num_open_accounts"))
    
    with tab3:
        features = ["age", "loan_to_income_ratio", "loan_tenure_months", 
                  "credit_utilization_ratio", "delinquency_ratio", 
                  "avg_dpd_per_delinquency", "num_open_accounts"]
        
        for feature in features:
            st.markdown(f"<h4>{feature.replace('_', ' ').title()} Distribution</h4>", unsafe_allow_html=True)
            st.pyplot(create_kde_plot(feature))

else:
    # Get values from session state for marking on the distributions
    age = st.session_state.age
    income = st.session_state.income
    loan_amount = st.session_state.loan_amount
    loan_tenure_months = st.session_state.loan_tenure_months
    avg_dpd_per_delinquency = st.session_state.avg_dpd_per_delinquency
    loan_to_income_ratio = st.session_state.loan_to_income_ratio
    delinquency_ratio = st.session_state.delinquency_ratio
    credit_utilization_ratio = st.session_state.credit_utilization_ratio
    num_open_accounts = st.session_state.num_open_accounts
    
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
            These plots show the distribution of each feature for defaulters (orange) and non-defaulters (blue).
            <br>
            The red dashed line shows your current value for each feature, helping you understand how your profile compares.
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different feature categories
    tab1, tab2, tab3 = st.tabs(["Demographics & Loan", "Credit History", "All Features"])
    
    with tab1:
        st.markdown("<h4>Age Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("age", age))
        
        st.markdown("<h4>Loan to Income Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("loan_to_income_ratio", loan_to_income_ratio))
        
        st.markdown("<h4>Loan Tenure Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("loan_tenure_months", loan_tenure_months))
    
    with tab2:
        st.markdown("<h4>Credit Utilization Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("credit_utilization_ratio", credit_utilization_ratio))
        
        st.markdown("<h4>Delinquency Ratio Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("delinquency_ratio", delinquency_ratio))
        
        st.markdown("<h4>Average Days Past Due Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("avg_dpd_per_delinquency", avg_dpd_per_delinquency))
        
        st.markdown("<h4>Number of Open Accounts Distribution</h4>", unsafe_allow_html=True)
        st.pyplot(create_kde_plot("num_open_accounts", num_open_accounts))
    
    with tab3:
        feature_values = {
            "age": age,
            "loan_to_income_ratio": loan_to_income_ratio,
            "loan_tenure_months": loan_tenure_months,
            "credit_utilization_ratio": credit_utilization_ratio,
            "delinquency_ratio": delinquency_ratio,
            "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
            "num_open_accounts": num_open_accounts
        }
        
        for feature, value in feature_values.items():
            st.markdown(f"<h4>{feature.replace('_', ' ').title()} Distribution</h4>", unsafe_allow_html=True)
            st.pyplot(create_kde_plot(feature, value))

# Navigation buttons
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
                Go to What-If Analysis ▶
            </button>
        </a>
    """, unsafe_allow_html=True) 