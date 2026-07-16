
# ==========================================================
# Visit With Us
# Wellness Tourism Package Purchase Prediction
#
# Streamlit Web Application
# ==========================================================

# ==========================================================
# Import Required Libraries
# ==========================================================

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Visit With Us",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Configuration
# ==========================================================


CLASSIFICATION_THRESHOLD = 0.45

# ==========================================================
# Load Trained Pipeline
# ==========================================================

@st.cache_resource   
def load_model():

    try:
        with st.spinner("Loading prediction model..."):

            model_path = hf_hub_download(
                repo_id="ranjeetpatel29/Bank-Customer-Churn",
                repo_type="dataset",
                filename="model.pkl"
            )

            return joblib.load(model_path)

    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

    return model


model = load_model()
# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🌍 Visit With Us")

st.sidebar.markdown("---")

st.sidebar.header("Project Information")

st.sidebar.write("### Model")

st.sidebar.success("✔ XGBoost Classifier")

st.sidebar.write("### MLOps Pipeline")

st.sidebar.success("✔ Data Preprocessing")

st.sidebar.success("✔ Hyperparameter Tuning")

st.sidebar.success("✔ MLflow Tracking")

st.sidebar.success("✔ GitHub Actions CI/CD")

st.sidebar.success("✔ Docker")

st.sidebar.success("✔ Hugging Face Spaces")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** before being contacted by the marketing team.

The prediction is generated using an XGBoost model trained through an end-to-end MLOps pipeline.
"""
)

# ==========================================================
# Application Header
# ==========================================================

st.title("🌍 Visit With Us")

st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
"""
Use this application to predict whether a customer is likely to purchase the newly introduced **Wellness Tourism Package**.

Fill in the customer details below and click **Predict Purchase** to receive the prediction, purchase probability, and business recommendation.
"""
)

st.markdown("---")
# ==========================================================
# Customer Information
# ==========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

# ----------------------------------------------------------
# Column 1
# ----------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

# ----------------------------------------------------------
# Column 2
# ----------------------------------------------------------

with col2:

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        value=50000,
        step=1000
    )

# ----------------------------------------------------------
# Column 3
# ----------------------------------------------------------

with col3:

    passport = st.selectbox(
        "Passport",
        [
            "No",
            "Yes"
        ]
    )

    own_car = st.selectbox(
        "Own Car",
        [
            "No",
            "Yes"
        ]
    )

    city_tier = st.selectbox(
        "City Tier",
        [
            1,
            2,
            3
        ]
    )

st.markdown("---")
# ==========================================================
# Travel Preferences
# ==========================================================

st.header("🌴 Travel Preferences")

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# Column 1
# ----------------------------------------------------------

with col1:

    number_of_trips = st.number_input(
        "Number of Trips Per Year",
        min_value=0,
        max_value=20,
        value=2
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

# ----------------------------------------------------------
# Column 2
# ----------------------------------------------------------

with col2:

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [
            1,
            2,
            3,
            4,
            5
        ]
    )

st.markdown("---")


# ==========================================================
# Sales Interaction
# ==========================================================

st.header("📞 Sales Interaction")

col1, col2, col3 = st.columns(3)

# ----------------------------------------------------------
# Column 1
# ----------------------------------------------------------

with col1:

    typeof_contact = st.selectbox(
        "Type of Contact",
        [
            "Company Invited",
            "Self Enquiry"
        ]
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

# ----------------------------------------------------------
# Column 2
# ----------------------------------------------------------

with col2:

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    number_of_followups = st.slider(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=2
    )

# ----------------------------------------------------------
# Column 3
# ----------------------------------------------------------

with col3:

    duration_of_pitch = st.slider(
        "Duration of Pitch (Minutes)",
        min_value=5,
        max_value=60,
        value=20
    )

st.markdown("---")

# ==========================================================
# Predict Button
# ==========================================================

predict_button = st.button(
    "🚀 Predict Purchase",
    use_container_width=True,
    type="primary"
)
# ==========================================================
# Predict Purchase
# ==========================================================

if predict_button:

    # ------------------------------------------------------
    # Convert Binary Inputs
    # ------------------------------------------------------

    passport = 1 if passport == "Yes" else 0

    own_car = 1 if own_car == "Yes" else 0

    # ------------------------------------------------------
    # Create Age Group
    # ------------------------------------------------------

    if age < 25:
        age_group = "Young Adult"

    elif age < 35:
        age_group = "Adult"

    elif age < 45:
        age_group = "Middle Age"

    elif age < 55:
        age_group = "Senior Adult"

    else:
        age_group = "Senior Citizen"

    # ------------------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------------------

    input_df = pd.DataFrame({

        "Age":[age],

        "TypeofContact":[typeof_contact],

        "CityTier":[city_tier],

        "Occupation":[occupation],

        "Gender":[gender],

        "NumberOfPersonVisiting":[number_of_person_visiting],

        "PreferredPropertyStar":[preferred_property_star],

        "MaritalStatus":[marital_status],

        "NumberOfTrips":[number_of_trips],

        "Passport":[passport],

        "OwnCar":[own_car],

        "NumberOfChildrenVisiting":[number_of_children_visiting],

        "Designation":[designation],

        "MonthlyIncome":[monthly_income],

        "PitchSatisfactionScore":[pitch_satisfaction_score],

        "ProductPitched":[product_pitched],

        "NumberOfFollowups":[number_of_followups],

        "DurationOfPitch":[duration_of_pitch],

        "AgeGroup":[age_group]

    })

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # ------------------------------------------------------
    # Display Result
    # ------------------------------------------------------

    st.markdown("---")

    st.header("Prediction Result")

    st.metric(

        label="Purchase Probability",

        value=f"{probability*100:.2f}%"

    )

    st.progress(float(probability))

    if prediction == 1:

        st.success(
            "✅ Customer is likely to purchase the Wellness Tourism Package."
        )

    else:

        st.error(
            "❌ Customer is unlikely to purchase the Wellness Tourism Package."
        )

    # ------------------------------------------------------
    # Business Recommendation
    # ------------------------------------------------------

    st.subheader("Recommendation")

    if probability >= 0.80:

        st.success(
            """
            ⭐ Excellent Prospect

            Recommend immediate follow-up by the Sales Team.
            """
        )

    elif probability >= 0.60:

        st.warning(
            """
            👍 Good Prospect

            Schedule another follow-up and provide promotional offers.
            """
        )

    else:

        st.info(
            """
            ℹ Low Purchase Probability

            No immediate marketing action is recommended.
            """
        )
