import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os # Import os to access environment variables

st.set_page_config(layout="wide")

st.title("Intent Classifier Interface")

st.markdown("""
Enter a text below to classify its intent. The app will show the predicted intent 
and a bar chart of probabilities for all possible intents.
""")

# FastAPI endpoint URL
# Get the API_BASE_URL from environment variable, with a fallback for local development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/intents")
API_URL = f"{API_BASE_URL}/confusion"

st.sidebar.info(f"Connecting to API at: {API_URL}") # Display for debugging/info

# Input text area
input_text = st.text_area("Enter your text here:", height=100)

if st.button("Classify Intent"):
    if input_text:
        try:
            payload = {"text": input_text}
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()
            
            st.subheader("Prediction Results")
            st.write(f"**Input Text:** {data.get('text')}")
            st.write(f"**Predicted Intent:** `{data.get('prediction')}`")
            st.write(f"**Certainty:** {data.get('certainty'):.4f}")

            all_probabilities = data.get("all_probabilities")
            if all_probabilities:
                st.subheader("Intent Probabilities")
                
                # Prepare data for Plotly
                prob_df = pd.DataFrame(list(all_probabilities.items()), columns=['Intent', 'Probability'])
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                
                # Create an interactive bar chart
                fig = px.bar(prob_df, 
                             x='Intent', 
                             y='Probability', 
                             title="Probabilities for All Intents",
                             color='Probability', # Color bars by probability value
                             color_continuous_scale=px.colors.sequential.Viridis,
                             text='Probability') # Show probability value on bars
                
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                                  xaxis_tickangle=-45,
                                  yaxis_title="Probability",
                                  xaxis_title="Intent")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Probability distribution not available in the API response.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to classify.")

# Instructions to run
st.sidebar.header("Header of sidebar")
st.sidebar.markdown("""
Here you have some space for adding content to your interface
""")
