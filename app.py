import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction App")
st.divider()
st.write("With this app you can get an estimation of the salary")

years = st.number_input("Years of Experience", value=1, step=1, min_value=0)
jobrate = st.number_input("Job Rating", value=3.5, step=0.5, min_value=0.0)

# ✅ Use jobrate instead of joblib
x = [years, jobrate]

# Load your model
model = joblib.load("Linearmodel.pkl")

st.divider()
predict = st.button("Press the button for the salary prediction")
st.divider()

if predict:
    st.balloons()
    X1 = np.array([x])
    prediction = model.predict(X1)
    st.write(f"Salary prediction is {prediction[0]}")
else:
    st.write("Please press the button for the app to make the prediction")
