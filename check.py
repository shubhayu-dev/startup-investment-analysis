import streamlit as st

# Access the API key stored in secrets.toml
api_key = st.secrets["openrouter"]["API_KEY"]

# You can now use the api_key securely
print(api_key)  # Test to ensure it's working correctly
