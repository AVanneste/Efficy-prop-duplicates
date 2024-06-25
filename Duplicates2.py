import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz

# Load the dataframe from the API
# df = pd.read_csv('data.csv')
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Create a Streamlit app
st.title('Similarity Match Finder')

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file,dtype=str)
    st.write('Data uploaded successfully. These are the first 5 rows.')
    st.dataframe(df.head(5))
    # Select the columns to compare
    column1 = st.selectbox('Select the first column', df.columns)
    column2 = st.selectbox('Select the second column', df.columns)

    # Set the similarity threshold
    threshold = st.slider('Similarity Threshold', 0.0, 1.0, 0.5)

    # Create a function to find the best matches
    def find_best_matches(row1, row2, threshold):
        return process.extractOne(row1, choices=[row2], scorer=fuzz.ratio)[1] >= threshold

    # Create a function to display the results
    def display_results(row1, row2):
        matches = df.apply(lambda x: find_best_matches(x[column1], x[column2], threshold), axis=1)
        st.write(f'Best matches for {row1} and {row2} with a threshold of {threshold}:')
        st.write(df[matches].head(10))

    # Display the results
    display_results(column1, column2)