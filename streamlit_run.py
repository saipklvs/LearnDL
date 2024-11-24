import streamlit as st
import os
from PIL import Image
import pandas as pd

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Summary Plots", "Predicted Data", "Individual Plots"]
choice = st.sidebar.radio("Go to", options)

# Paths to the saved plots
summary_path = "temp/summary/"
waterfall_path = "temp/waterfall/"
saved_predictions_path = "temp/saved_predictions/"

if choice == "Summary Plots":
    st.title("SHAP Summary Plots")
    for file_name in os.listdir(summary_path):
        if file_name.endswith(".png"):
            st.image(os.path.join(summary_path, file_name), caption=file_name)

elif choice == "Predicted Data":
    st.title("Predicted Data")
    predictions_df = pd.read_csv(os.path.join(saved_predictions_path, "predictions.csv"))
    st.dataframe(predictions_df, width=1000, height=500, use_container_width=True)

elif choice == "Individual Plots":
    st.title("SHAP Individual Waterfall Plots")
    for file_name in os.listdir(waterfall_path):
        if file_name.endswith(".png"):
            st.image(os.path.join(waterfall_path, file_name), caption=file_name)
