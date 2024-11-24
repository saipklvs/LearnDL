import streamlit as st
import os
import pandas as pd

st.sidebar.title("Navigation")
options = ["Summary Plots", "Predicted Data and Individual Plots"]
choice = st.sidebar.selectbox("Go to", options)

# Paths to the saved plots
summary_path = "temp/summary/"
waterfall_path = "temp/waterfall/"
saved_predictions_path = "temp/saved_predictions/"

if choice == "Summary Plots":
    st.title("SHAP Summary Plots")
    for file_name in os.listdir(summary_path):
        if file_name.endswith(".png"):
            st.image(os.path.join(summary_path, file_name), caption=file_name)

elif choice == "Predicted Data and Individual Plots":
    st.title("Predicted Data and Individual SHAP Waterfall Plots")
    predictions_df = pd.read_csv(os.path.join(saved_predictions_path, "predictions.csv"))
    
    # Display the dataframe
    # st.dataframe(predictions_df, width=1000, height=500, use_container_width=True)
    
    # Dropdown to select a specific index
    selected_index = st.selectbox("Select DataFrame Index", predictions_df.index)
    
    # Display the selected row
    st.write("Selected Data:")
    st.write(predictions_df.loc[selected_index])
    
    # Display the corresponding SHAP waterfall plot
    plot_file_name = f"shap_waterfall_instance_{selected_index}.png"
    plot_file_path = os.path.join(waterfall_path, plot_file_name)
    if os.path.exists(plot_file_path):
        st.image(plot_file_path, caption=plot_file_name)
    else:
        st.write("No corresponding SHAP waterfall plot found.")
