
# Importing necessary libraries
import streamlit as st
import pandas as pd
import tensorflow as tf

# Streamlit app function
def streamlit_app():
    # Streamlit interface design
    st.title('Streamlit TensorFlow Interface')

    # Section for CSV Data Upload
    st.subheader('Upload CSV Data')
    data_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if data_file is not None:
        # Reading CSV data
        data = pd.read_csv(data_file)
        # Displaying CSV data
        st.write("Data from CSV file:")
        st.dataframe(data)

    # Section for TensorFlow Model Upload
    st.subheader('Upload TensorFlow Model')
    model_file = st.file_uploader("Choose a TensorFlow model file (.h5)", type=['h5'])

    if model_file is not None:
        # Loading TensorFlow model
        model = tf.keras.models.load_model(model_file)
        st.write("Model successfully loaded!")

# Function call
streamlit_app()
