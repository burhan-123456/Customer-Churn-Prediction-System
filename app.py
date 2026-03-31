import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD FILES ---------------- #

model = pickle.load(open("model.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- UI ---------------- #

st.title("📊 Customer Churn Prediction")
st.header("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tenure = st.number_input("Tenure (months)", min_value=0)
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

# ---------------- PROCESS ---------------- #

def convert_yes_no(val):
    return 1 if val == "Yes" else 0

if st.button("Predict Churn"):

    # Input dictionary
    input_data = {
        'gender': 1 if gender == "Female" else 0,
        'SeniorCitizen': convert_yes_no(SeniorCitizen),
        'Partner': convert_yes_no(Partner),
        'Dependents': convert_yes_no(Dependents),
        'tenure': tenure,
        'PhoneService': convert_yes_no(PhoneService),
        'MultipleLines': convert_yes_no(MultipleLines),
        'OnlineSecurity': convert_yes_no(OnlineSecurity),
        'OnlineBackup': convert_yes_no(OnlineBackup),
        'DeviceProtection': convert_yes_no(DeviceProtection),
        'TechSupport': convert_yes_no(TechSupport),
        'StreamingTV': convert_yes_no(StreamingTV),
        'StreamingMovies': convert_yes_no(StreamingMovies),
        'PaperlessBilling': convert_yes_no(PaperlessBilling),
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod
    }

    df = pd.DataFrame([input_data])

    # ---------------- ENCODING ---------------- #

    encoded = ohe.transform(df[['InternetService', 'Contract', 'PaymentMethod']])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(['InternetService','Contract','PaymentMethod'])
    ).astype(int)

    # Drop original categorical
    df = df.drop(columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Merge encoded
    df = pd.concat([df, encoded_df], axis=1)

    # ---------------- FINAL FIX ---------------- #

    # ❌ Ensure target not present
    df = df.drop(columns=['Churn'], errors='ignore')

    # ✅ Keep only training columns
    df = df.reindex(columns=columns, fill_value=0)

    # ---------------- DEBUG (optional) ---------------- #

    # st.write("Shape:", df.shape)

    # ---------------- PREDICTION ---------------- #

    prob = model.predict_proba(df)[:, 1][0]
    prediction = 1 if prob > 0.4 else 0

    # ---------------- OUTPUT ---------------- #

    st.subheader("Result")
    st.write(f"Churn Probability: {prob:.2f}")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")