import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Mushroom Edibility Prediction", page_icon="🍄", layout="centered")

st.title(" Mushroom Edibility Prediction App")

st.write("""
This app predicts whether a mushroom is **edible or poisonous** based on its characteristics.  
Model used: **Random Forest Classifier**
""")

# Load trained model
model = pickle.load(open("rf.pkl", "rb"))

# Load dataset to get column names
df = pd.read_csv("mushrooms.csv")

# Encode column options for dropdowns
feature_options = {}
for col in df.columns:
    if col != "class":
        feature_options[col] = sorted(df[col].unique())

st.header("Enter Mushroom Characteristics")

# Create inputs dynamically
user_input = {}
for col in df.columns:
    if col != "class":
        user_input[col] = st.selectbox(f"{col}", feature_options[col])

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Encode input using same LabelEncoder method as training
from sklearn.preprocessing import LabelEncoder
for col in input_df.columns:
    le = LabelEncoder()
    le.fit(df[col])
    input_df[col] = le.transform(input_df[col])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 0:
        st.success("The mushroom is **EDIBLE**.")
    else:
        st.error(" The mushroom is **POISONOUS**.")

st.markdown("---")
st.caption("Developed using Streamlit and Random Forest Classifier.")
