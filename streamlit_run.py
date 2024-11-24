import streamlit as st
import os
from PIL import Image

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Summary Plots", "Predicted Data", "Individual Plots"]
choice = st.sidebar.radio("Go to", options)

# Paths to the saved plots
summary_path = "temp/summary/"
waterfall_path = "temp/waterfall/"

if choice == "Summary Plots":
    st.title("SHAP Summary Plots")
    for file_name in os.listdir(summary_path):
        if file_name.endswith(".png"):
            st.image(os.path.join(summary_path, file_name), caption=file_name)

elif choice == "Predicted Data":
    st.title("Predicted Data")
    # Load and display the predicted data
    # ...existing code to load and display predicted data...

elif choice == "Individual Plots":
    st.title("SHAP Individual Waterfall Plots")
    for file_name in os.listdir(waterfall_path):
        if file_name.endswith(".png"):
            st.image(os.path.join(waterfall_path, file_name), caption=file_name)
