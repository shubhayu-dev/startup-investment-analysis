import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from ydata_profiling import ProfileReport # For generating detailed data profile reports
import requests # For making HTTP requests to APIs (like DeepSeek)
from dotenv import load_dotenv # For loading environment variables from a .env file
import numpy as np # For numerical operations, often used with pandas
import streamlit.components.v1 as components # For embedding HTML components

# Load environment variables from .env file (e.g., API keys)
load_dotenv()
# Default theme for the application, can be overridden by user selection
DEFAULT_THEME = "Light" 

# --- Helper Functions ---

def generate_deepseek_summary(data: pd.DataFrame, context_str: str = "the following dataset sample") -> str:
    """
    Generates a concise analytical summary of the provided DataFrame sample using the DeepSeek API.

    Args:
        data (pd.DataFrame): The DataFrame to summarize. A sample will be taken.
        context_str (str): A string describing the context of the data (e.g., "the full dataset", "filtered data for X").

    Returns:
        str: The AI-generated summary or an error message.
    """
    API_KEY = os.getenv('OPENROUTER_API_KEY')
    if not API_KEY: 
        return "Error: OPENROUTER_API_KEY not found in environment variables."
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    # Headers required for the OpenRouter API
    headers = {
        "Authorization": f"Bearer {API_KEY}", 
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost:8501"), # Recommended by OpenRouter for tracking usage
        "X-Title": os.getenv("YOUR_SITE_NAME", "Data Summary App")      # Recommended by OpenRouter for identifying your app
    }
    
    try:
        # Prepare a sample of the data to send to the API (as CSV)
        sample_size = min(len(data), 50) # Limit sample size to manage API request size, cost, and token limits
        # Ensure there's data to sample, otherwise send a specific message to the AI
        data_s_csv = data.sample(sample_size if sample_size > 0 else 1).to_csv(index=False) if not data.empty else "Dataset is empty."
    except Exception as e: 
        return f"Error preparing data for API: {e}" # Catch errors during data sampling or CSV conversion
    
    # Construct the message payload for the API, providing context and the data sample
    messages = [{
        "role": "user", 
        "content": f"Concise analytical summary of key insights, trends, anomalies for {context_str}:\n{data_s_csv}"
    }]
    payload = {
        "model": "deepseek/deepseek-coder", # Specify the DeepSeek model to be used
        "messages": messages, 
        "max_tokens": 700 # Limit the length of the generated summary to control response size
    }
    
    try:
        # Make the POST request to the API with a timeout
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45) 
        if response.status_code == 200: # Check for a successful HTTP response
            try:
                result = response.json() # Parse the JSON response
                # Extract the summary content from the API response structure
                # This path depends on the specific API's response format
                return result['choices'][0]['message']['content'] if 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message'] else f"Unexpected API response structure: {result}"
            except ValueError: # Handle cases where the response is not valid JSON
                return f"Invalid JSON received from API: {response.text}"
        else: # Handle API errors indicated by non-200 status codes
            return f"API Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e: # Handle network issues or other request-related errors
        return f"API Call Error: {e}"
    except Exception as e: # Catch-all for other unexpected errors during the API call
        return f"An unexpected error occurred during API call: {e}"

def generate_profile(df: pd.DataFrame, title: str = "Data Summary"):
    """
    Generates a detailed data profile report using ydata-profiling.

    Args:
        df (pd.DataFrame): The DataFrame to profile.
        title (str): The title for the profile report and part of the output filename.

    Returns:
        str: The filename of the generated HTML report.
    """
    # Create the profile report object from ydata-profiling
    # explorative=True enables more detailed analysis, minimal=False provides a comprehensive report
    profile = ProfileReport(df, title=title, explorative=True, minimal=False) 
    # Sanitize the title to create a valid and descriptive filename for the HTML report
    filename = title.lower().replace(" ", "_").replace("/", "_") + "_report.html" 
    profile.to_file(filename) # Save the report to an HTML file
    return filename # Return the name of the file for later use (e.g., display or download)

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads startup funding data from a CSV file, cleans it, and prepares it for analysis.
    This involves renaming columns, handling missing values, parsing dates, converting data types,
    and standardizing categorical values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The cleaned DataFrame, or an empty DataFrame if a critical error occurs.
    """
    try:
        # Load the raw data from the CSV file
        # 'latin1' encoding is often used for datasets that might contain special characters not handled by UTF-8
        df = pd.read_csv(file_path, encoding='latin1') 
        
        # Standardize column names to snake_case for easier and consistent access in Python
        df = df.rename(columns={
            'Sr no': 'sr_no', 'Date': 'date_orig', 'Startup name': 'startup_name', 
            'industry_vertical': 'industry_vertical', # Original CSV name matches the target
            'Sub Vertical': 'sub_vertical',         # Corrected key based on actual CSV column name
            'City Location': 'city_location', 
            'Investment Name': 'investors_name',    # 'Investment Name' from CSV becomes 'investors_name'
            'Investment Type': 'investment_type', 
            'amount_in_usd': 'amount_in_usd'          # Original CSV name matches the target
        })
        
        # Define a list of columns expected to be present after renaming for data integrity checks
        expected_cols = ['sr_no', 'date_orig', 'startup_name', 'industry_vertical', 'sub_vertical', 
                         'city_location', 'investors_name', 'investment_type', 'amount_in_usd']
        # Check for missing expected columns and add them as NA to prevent downstream errors
        for col in expected_cols:
            if col not in df.columns:
                if col == 'sub_vertical': 
                    st.sidebar.info(f"Info: Column '{col}' (derived from 'Sub Vertical') is not actively used in the dashboard.") 
                else: 
                    st.warning(f"Warning: Column '{col}' was expected but not found after renaming. Check CSV source and rename map.")
                df[col] = pd.NA # Add missing columns with Not Available (NA) values

        # Date Cleaning and Feature Engineering (Year, Month Name, Month Number)
        # Ensure 'date_orig' exists and is a string type before attempting to parse
        if 'date_orig' in df.columns and pd.api.types.is_string_dtype(df['date_orig']): 
            df['date'] = pd.to_datetime(df['date_orig'], errors='coerce') # Convert to datetime objects, invalid dates become NaT
            df = df.dropna(subset=['date']) # Remove rows where date conversion failed (NaT values)
            if not df.empty : # Proceed only if DataFrame is not empty after dropping NaT rows
                df['year'] = df['date'].dt.year
                df['month_name'] = df['date'].dt.month_name()
                df['month'] = df['date'].dt.month # Numeric month for potential sorting
            else: # If DataFrame becomes empty after date cleaning
                df['year'], df['month_name'], df['month'] = pd.NA, pd.NA, pd.NA
        else: # If 'date_orig' column is missing or not suitable for parsing
            df['date'],df['year'], df['month_name'], df['month'] = pd.NaT, pd.NA, pd.NA, pd.NA # Initialize date-related columns as NA/NaT
            if 'date_orig' not in df.columns: 
                st.error("Error: The original date column (expected as 'Date' in CSV) is missing.")

        # Amount Cleaning: Convert 'amount_in_usd' to a numeric type
        if 'amount_in_usd' in df.columns and not df['amount_in_usd'].isnull().all():
            df['amount_in_usd'] = df['amount_in_usd'].astype(str).str.replace(',', '', regex=False) # Remove thousands separators (commas)
            df['amount_in_usd'] = pd.to_numeric(df['amount_in_usd'], errors='coerce').fillna(0) # Convert to number, errors become NaN then filled with 0
        else: # If 'amount_in_usd' column is missing or entirely null
            df['amount_in_usd'] = 0 # Default to 0
            if 'amount_in_usd' not in df.columns: 
                st.error("Error: The amount column (expected as 'amount_in_usd' in CSV) is missing.")
        
        # Categorical Data Cleaning Loop (for City Location and Investment Type)
        for col_name, default_val in [('city_location', 'Unknown'), ('investment_type', 'Unknown')]:
            if col_name in df.columns and not df[col_name].isnull().all():
                # Standardize text: convert to string, strip whitespace, title case
                df[col_name] = df[col_name].astype(str).str.strip().str.title() 
                if col_name == 'city_location':
                    # Specific city name normalizations (e.g., 'Delhi' to 'New Delhi')
                    df[col_name] = df[col_name].replace({'Delhi': 'New Delhi', 'Bangalore': 'Bengaluru', 'Mumbai / Pune': 'Mumbai'}, regex=False)
                    # Standardize common missing value indicators to 'Unknown'
                    df.loc[df[col_name].str.contains('Missing|N/A', case=False, na=False), col_name] = 'Unknown' 
                elif col_name == 'investment_type':
                    # Specific investment type normalizations (e.g., handling '\N' and common variations)
                    df[col_name] = df[col_name].replace({ r'Seed\\Nfunding': 'Seed Funding', r'Private\\Nequity': 'Private Equity', r'\\N': ''}, regex=True)
                    df = df[df[col_name].str.len() > 0].copy() # Remove rows where type became empty after replacement, ensure a copy is made
                    # Further specific replacements for investment types
                    df[col_name] = df[col_name].replace({
                        'Pre-Series A': 'Pre-Series A', 'Seed Funding R': 'Seed Funding', 'Seed/ Angel Fu': 'Seed/Angel Fund', 
                        'Seed/ Angel Fun': 'Seed/Angel Fund', 'Seed / Angel Fu': 'Seed/Angel Fund', 'Seed/Angel Fun': 'Seed/Angel Fund',
                        'Seed / Angle Fu': 'Seed/Angel Fund', 'Angel / Seed Fu': 'Seed/Angel Fund', 'SeedFunding': 'Seed Funding',
                        'Crowd funding': 'Crowd Funding', 'Angel': 'Angel Round', 'Private Equity R': 'Private Equity', 'Debt Financing': 'Debt Funding', 
                        'Private\Equity': 'Private Equity', 'Seed\Funding': 'Seed Funding', 'Seed Round': 'Seed Funding', 'Seed': 'Seed Funding'
                    }, regex=False)
                # General cleanup for empty strings or common NaN string representations to 'Unknown'
                df.loc[df[col_name].isin(['', 'Nan', 'NaN', 'N/A']), col_name] = 'Unknown' 
                df[col_name] = df[col_name].fillna('Unknown') # Fill any remaining actual NaNs with 'Unknown'
            else: # If the column is missing or entirely null
                df[col_name] = default_val # Assign the predefined default value
        
        # Fill NaNs with 'Unknown' for other key text columns like startup_name and industry_vertical
        for col_fill in ['startup_name', 'industry_vertical']: 
            if col_fill in df.columns and not df[col_fill].isnull().all(): 
                df[col_fill] = df[col_fill].fillna('Unknown')
            else: # If column is missing or entirely null
                df[col_fill] = 'Unknown'
        return df # Return the cleaned DataFrame
    # Error handling for file operations during data loading
    except FileNotFoundError: 
        st.error(f"Error: Data file not found: '{file_path}'. Please check the path."); 
        return pd.DataFrame() # Return an empty DataFrame on error
    except pd.errors.EmptyDataError: 
        st.error(f"Error: Data file '{file_path}' is empty."); 
        return pd.DataFrame()
    except Exception as e: 
        st.error(f"An unexpected error occurred during data loading/cleaning: {e}"); 
        return pd.DataFrame()


def create_plotly_chart(chart_func, data, x_col, y_col, title, x_label, y_label, 
                        color_col=None, template='plotly_white', orientation=None, 
                        barmode=None, nbinsx_hist=None, text_auto=False, 
                        log_y=False, log_x=False):
    """
    Creates a Plotly chart with common configurations and error handling.

    Args:
        chart_func: The Plotly Express function to call (e.g., px.bar, px.line, px.histogram).
        data (pd.DataFrame): The data for the chart.
        x_col (str): Column name for the x-axis.
        y_col (str, optional): Column name for the y-axis. Can be None for 1D histograms.
        title (str, optional): Chart title. If None or empty, no title is rendered.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        color_col (str, optional): Column name for color encoding.
        template (str): Plotly template for styling (e.g., 'plotly_dark', 'plotly_white').
        orientation (str, optional): Chart orientation ('h' for horizontal, 'v' for vertical).
        barmode (str, optional): For bar charts ('group', 'stack', 'relative').
        nbinsx_hist (int, optional): Suggested number of bins for x-axis of histograms.
        text_auto (bool or str): If True or format string, displays text on marks.
        log_y (bool): If True, sets y-axis to logarithmic scale.
        log_x (bool): If True, sets x-axis to logarithmic scale.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure, or None if an error occurs or data is empty.
    """
    if data.empty: 
        # Display an informative message if no data is available for the chart
        st.info(f"No data available for chart generation (context: {title if title and title.strip() != '' else 'Untitled Chart'})")
        return None
    try:
        # Handle cases where no title is desired by passing None to Plotly (prevents "undefined" text)
        effective_title = title if title and title.strip() != "" else None

        # Prepare dictionary of arguments for the Plotly Express function
        fig_args = {
            'data_frame': data, 
            'x': x_col, 
            'y': y_col, 
            'labels': {x_col: x_label, y_col: y_label, color_col: color_col or ''}, # Dynamic labels
            'title': effective_title, 
            'color': color_col, 
            'template': template, 
            'orientation': orientation, 
            'barmode': barmode, 
            'text_auto': text_auto, 
            'log_y': log_y, 
            'log_x': log_x  
        }
        
        # Special handling for histogram's number of bins on x-axis
        # px.histogram accepts 'nbinsx' directly as an argument.
        if chart_func == px.histogram and nbinsx_hist is not None:
            fig_args['nbinsx'] = nbinsx_hist 

        # Clean out any arguments that are None, as Plotly Express prefers them to be absent entirely
        fig_args_cleaned = {k: v for k, v in fig_args.items() if v is not None} 
        
        # Create the Plotly figure using the specified chart function and arguments
        fig = chart_func(**fig_args_cleaned)

        # Post-creation layout adjustments
        if effective_title:
            fig.update_layout(title_x=0.5) # Center the title if it exists
        else:
            # If no title, reduce the top margin to avoid excessive empty space
            fig.update_layout(margin=dict(t=30)) # Adjust top margin value as needed

        return fig # Return the created figure
    except Exception as e: 
        # Display an error message if chart creation fails
        st.error(f"Chart creation error ('{title if title else 'Untitled Chart'}'): {e}")
        return None # Return None on error

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Startup Funding India")

    # --- Theme Selection Logic ---
    if 'selected_theme' not in st.session_state:
        st.session_state.selected_theme = "Light" # Assuming "Light" is the DEFAULT_THEME
    theme_options = ["Light", "Dark"]
    default_theme_index = theme_options.index(st.session_state.selected_theme)

    chosen_theme = st.sidebar.radio(
        "🌗 Choose Theme",
        theme_options,
        index=default_theme_index,
        key="theme_toggle_main"
    )
    if chosen_theme != st.session_state.selected_theme:
        st.session_state.selected_theme = chosen_theme
        st.rerun()

    # --- Apply Theme-Specific CSS ---
    if st.session_state.selected_theme == "Light":
        st.markdown("""
        <style>
            /* --- Light Theme Specific CSS --- */

            /* Light Theme Tab Fixes */
            /* Inactive Tab Text Color for Light Theme */
            div[data-baseweb="tab-list"] div[data-baseweb="tab"]:not([aria-selected="true"]) {
                color: #212529 !important; /* A good dark grey for light theme inactive tabs */
            }
            
            /* Active Tab Text Color for Light Theme */
            div[data-baseweb="tab-list"] div[data-baseweb="tab"][aria-selected="true"] {
                color: #000000 !important; /* Black for active tab text in light mode */
            }

            /* Light Theme Sidebar Expander Header Text Fix */
            section[data-testid="stSidebar"] div[data-testid="stExpander"] summary { 
                color: #000000 !important; /* Force black text for sidebar expander headers */
            }
            section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
            section[data-testid="stSidebar"] div[data-testid="stExpander"] summary div[data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] div[data-testid="stExpander"] summary div[data-testid="stMarkdownContainer"] { 
                color: #000000 !important; 
            }

            /* Light Theme Download Button Text Color Fix */
            /* Targets the text within the download button to ensure it's dark */
            div[data-testid="stDownloadButton"] button p, 
            div[data-testid="stDownloadButton"] button div[data-testid="stMarkdownContainer"] p {
                color: #000000 !important; /* Black text for download button in light mode */
            }
            
            /* Optional: If you want to ensure the download button background is a standard light grey */
            /*
            div[data-testid="stDownloadButton"] button {
                background-color: #f0f2f6 !important; 
                border: 1px solid #d0d0d0 !important;
            }
            div[data-testid="stDownloadButton"] button:hover {
                background-color: #e0e0e0 !important;
                border-color: #c0c0c0 !important;
            }
            */
        </style>
        """, unsafe_allow_html=True)

    elif st.session_state.selected_theme == "Dark":
        # Apply custom CSS for the Dark Theme if it's selected
        st.markdown("""
        <style>
            /* --- YOUR FULL DARK THEME CSS AS PROVIDED BEFORE --- */
            /* Ensure your dark theme CSS correctly styles the download button text to be light.
            The existing rule:
            p, li, .stMarkdown, label { color: #f0f2f6 !important; }
            should cover the text inside the download button's <p> tag.
            If not, you might need a specific rule here like:
            div[data-testid="stDownloadButton"] button p,
            div[data-testid="stDownloadButton"] button div[data-testid="stMarkdownContainer"] p {
                color: #f0f2f6 !important; 
            }
            Also, ensure the button background is dark in dark mode.
            The existing rule for stButton:
            div[data-testid="stButton"] button { background-color: #238636; color: white; ... }
            might not apply to stDownloadButton. If download button needs specific dark background:
            div[data-testid="stDownloadButton"] button {
                background-color: #21262d !important; /* Example dark background */
                border: 1px solid #30363d !important; /* Example dark border */
            }
            */
            
            body { background-color: #0e1117 !important; } .stApp { background-color: #0e1117; color: #f0f2f6; } h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, label { color: #f0f2f6 !important; } a { color: #79c0ff !important; } a:hover { color: #58a6ff !important; } section[data-testid="stSidebar"] { background-color: #161c25 !important; border-right: 1px solid #30363d; } section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stRadio > label > div { color: #c9d1d9 !important; } section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f0f2f6 !important; } section[data-testid="stSidebar"] div[data-testid="stRadio"] > label { padding: 0.5rem 0.75rem !important; border-radius: 6px !important; margin-bottom: 0.25rem !important; border: 1px solid transparent !important; transition: background-color 0.2s ease, border-color 0.2s ease; } section[data-testid="stSidebar"] div[data-testid="stRadio"] > label:hover { background-color: #2a3038 !important; border-color: #4a5058 !important; } section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"]:checked + div + div > label { background-color: #094C87 !important; border-color: #58a6ff !important; } section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"]:checked + div + div > label span{ color: #FFFFFF !important; } section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:first-child { background-color: #21262d !important; border: 1px solid #30363d !important; border-radius: 6px !important; } section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:first-child:hover { border-color: #58a6ff !important; } div[data-baseweb="popover"] ul[role="listbox"] li div { background-color: #161c25 !important; color: #c9d1d9 !important; } div[data-baseweb="popover"] ul[role="listbox"] li:hover div { background-color: #2a3038 !important; color: #f0f2f6 !important; } div[data-baseweb="popover"] ul[role="listbox"] li[aria-selected="true"] div { background-color: #094C87 !important; color: #FFFFFF !important; } section[data-testid="stSidebar"] div[data-testid="stExpander"] { background-color: transparent !important; border: none !important; margin-bottom: 0.5rem; } section[data-testid="stSidebar"] div[data-testid="stExpander"] summary { padding: 0.5rem 0.75rem !important; border-radius: 6px !important; border: 1px solid #30363d !important; background-color: #21262d !important; color: #c9d1d9 !important; /* Explicitly set dark mode sidebar expander text color */ } section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover { background-color: #2a3038 !important; border-color: #58a6ff !important; color: #f0f2f6 !important; /* Dark mode hover text */ } section[data-testid="stSidebar"] div[data-testid="stExpander"][aria-expanded="true"] summary { background-color: #2a3038 !important; border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important; border-bottom-color: transparent !important; } section[data-testid="stSidebar"] div[data-testid="stExpander"] [data-testid="stVerticalBlock"] { background-color: #1c222b !important; padding: 0.5rem !important; border: 1px solid #30363d !important; border-top: none !important; border-bottom-left-radius: 6px !important; border-bottom-right-radius: 6px !important; } div[data-baseweb="tab-list"] { background-color: #0e1117 !important; padding-bottom: 0px !important; border-bottom: 1px solid #30363d !important; } div[data-baseweb="tab"]:not([aria-selected="true"]) { background-color: transparent !important; color: #8b949e !important; padding: 0.75rem 1rem !important; border-bottom: 3px solid transparent !important; transition: color 0.2s ease, border-color 0.2s ease; } div[data-baseweb="tab"]:not([aria-selected="true"]):hover { background-color: #161c25 !important; color: #c9d1d9 !important; } div[data-baseweb="tab"][aria-selected="true"] { background-color: transparent !important; color: #f0f2f6 !important; border-bottom: 3px solid #58a6ff !important; font-weight: 600 !important; } div[data-testid="stTabs"] [data-testid="stVerticalBlock"] { padding-top: 1rem !important; } div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; } div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus { border-color: #58a6ff; box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.25); } div[data-testid="stButton"] button { background-color: #238636; color: white; border: 1px solid #2ea043; border-radius: 6px; padding: 0.5rem 1rem; } div[data-testid="stButton"] button:hover { background-color: #2ea043; border-color: #3fb950; } div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; background-color: #161c25; } div[data-testid="stDataFrame"] table th { background-color: #21262d; color: #f0f2f6; border-bottom: 1px solid #30363d; } div[data-testid="stDataFrame"] table td { color: #c9d1d9; border-bottom: 1px solid #21262d; } div[data-testid="stMetric"] { background-color: #161c25; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; } div[data-testid="stMetric"] label { color: #8b949e !important; } div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f0f2f6 !important; } div[data-testid="stExpander"]:not(section[data-testid="stSidebar"] div[data-testid="stExpander"]) { background-color: #161c25; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 1rem; } div[data-testid="stExpander"]:not(section[data-testid="stSidebar"] div[data-testid="stExpander"]) summary { color: #c9d1d9 !important; padding: 0.75rem; } div[data-testid="stInfo"] { background-color: rgba(56, 139, 253, 0.1); border: 1px solid rgba(56, 139, 253, 0.4); color: #79c0ff; border-radius: 6px; } div[data-testid="stWarning"] { background-color: rgba(187, 128, 9, 0.1); border: 1px solid rgba(187, 128, 9, 0.4); color: #d29922; border-radius: 6px; } div[data-testid="stError"] { background-color: rgba(248, 81, 73, 0.1); border: 1px solid rgba(248, 81, 73, 0.4); color: #f85149; border-radius: 6px; } div[data-testid="stSuccess"] { background-color: rgba(35, 134, 54, 0.1); border: 1px solid rgba(35, 134, 54, 0.4); color: #56d364; border-radius: 6px; } iframe { background-color: #0e1117 !important; border-radius: 6px; }
        </style>
        """, unsafe_allow_html=True)

    # ... (rest of your main function, including plotly_template setting, data loading, filters, and dashboard layout) ...
    # Make sure that the plotly_template is set AFTER the st.session_state.selected_theme is determined
    plotly_template = 'plotly_dark' if st.session_state.selected_theme == "Dark" else 'plotly_white'
    
    # Application title and a brief introduction
    st.title("🇮🇳 Startup Investment Analysis in India")
    st.markdown("Interactive dashboard for exploring funding trends (2015-2020). *Data courtesy of Kaggle.*")

    # Load and clean the dataset
    data_file_path = '/home/shubhayu/Desktop/startup-investment-analysis/data/cleaned_startup_funding.csv'
    df = load_and_clean_data(data_file_path)
    # If data loading fails or results in an empty DataFrame, show a warning and stop
    if df.empty: 
        st.warning("Dataset could not be loaded or is empty. Please check the file path and data integrity.")
        return 

    # --- Sidebar: Filters for Data Exploration ---
    st.sidebar.header("🔍 Filters")
    # Year Filter: Includes an "All Years" option for overview analysis
    if not ('year' in df.columns and not df['year'].isnull().all()): # Check if 'year' column is valid
        st.sidebar.error("Year data is missing or invalid in the dataset. Cannot proceed.")
        return 
    
    unique_years = sorted(df['year'].dropna().unique().astype(int)) # Get unique years from the data
    year_options = ["All Years"] + unique_years # Prepend "All Years" to the list
    selected_year_option = st.sidebar.selectbox(
        "Select Year", 
        options=year_options, 
        index=0 # Default selection is "All Years"
    ) 

    # Determine the base DataFrame for subsequent filtering based on the year selection
    if selected_year_option == "All Years":
        time_basis_df = df.copy() # Use the entire dataset if "All Years" is selected
        current_time_period_title = "Overall (2015-2020)" # Title reflects the full period
    else:
        time_basis_df = df[df['year'] == selected_year_option].copy() # Filter by the specific year
        current_time_period_title = str(selected_year_option) # Title reflects the selected year

    # City Filter: Options dynamically populated based on the 'time_basis_df'
    selected_city = "All" # Default
    if 'city_location' in time_basis_df.columns and not time_basis_df['city_location'].isnull().all():
        # Get unique cities that have funding data from the current time_basis_df
        cities_with_data = ["All"] + sorted(time_basis_df[time_basis_df['amount_in_usd'] > 0]['city_location'].dropna().unique())
        selected_city = st.sidebar.selectbox("Select City", options=cities_with_data)
    else: # If city data is not available for the current year selection
        st.sidebar.info("City data not available for the current year selection.")

    # Investment Type Filter: Options dynamically populated
    selected_investment = "All" # Default
    if 'investment_type' in time_basis_df.columns and not time_basis_df['investment_type'].isnull().all():
        investment_type_options = ["All"] + sorted(time_basis_df['investment_type'].dropna().unique())
        selected_investment = st.sidebar.selectbox("Select Investment Type", options=investment_type_options)
    else: # If investment type data is not available
        st.sidebar.info("Investment type data not available for the current year selection.")
    
    # Apply the City and Investment Type filters to the 'time_basis_df' to get the final 'filtered_df'
    filtered_df = time_basis_df.copy()
    if selected_city != "All": 
        filtered_df = filtered_df[filtered_df['city_location'] == selected_city]
    if selected_investment != "All": 
        filtered_df = filtered_df[filtered_df['investment_type'] == selected_investment]

    # --- Main Dashboard Display Area ---
    if filtered_df.empty: # If no data matches the current filters
        st.info("No data matches the selected filters. Please try broadening your criteria.")
    else:
        # Display a dynamic subheader reflecting the current filter context
        st.subheader(f"📊 Dashboard for {current_time_period_title}" + 
                     (f" in {selected_city}" if selected_city != "All" else "") + 
                     (f" ({selected_investment})" if selected_investment != "All" else ""))
        
        # Display Key Metrics in columns
        m_col1, m_col2, m_col3 = st.columns(3)
        total_inv = filtered_df['amount_in_usd'].sum()
        num_startups = filtered_df['startup_name'].nunique()
        with m_col1: st.metric("💰 Total Investment (USD)", f"${total_inv:,.0f}") # Formatted currency
        with m_col2: st.metric("🚀 Startups Funded", f"{num_startups}")
        with m_col3: 
            avg_inv = (total_inv / num_startups) if total_inv > 0 and num_startups > 0 else 0
            st.metric("🏦 Avg. Investment (USD)", f"${avg_inv:,.0f}" if avg_inv > 0 else "N/A") # Handle division by zero
        st.markdown("---") # Visual separator

        # Tabs for organizing different categories of analysis
        tab_titles = ["  Key Trends & Overview ", " Sector & Type Analysis ", " Investor Activity ", " Geographical Insights "]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        # --- Tab 1: Key Trends & Overview ---
        # This tab focuses on overall funding flows, either yearly or monthly based on selection.
        with tab1:
            st.markdown(f"#### Funding Flows for {current_time_period_title}")
            t1_col1, t1_col2 = st.columns(2) # Layout in two columns
            with t1_col1:
                # Display Yearly Funding Trend if "All Years" is selected
                if selected_year_option == "All Years":
                    st.markdown("###### Yearly Funding Trend (Total Amount)")
                    yearly_funding = filtered_df.groupby('year')['amount_in_usd'].sum().reset_index()
                    # Create bar chart for yearly funding, title is None as markdown provides it
                    fig = create_plotly_chart(px.bar, yearly_funding, 'year', 'amount_in_usd', 
                                              title=None, x_label='Year', y_label='Total Funding (USD)', 
                                              template=plotly_template, text_auto='.2s') # '.2s' for smart number formatting
                    if fig: st.plotly_chart(fig, use_container_width=True)
                else: # Display Monthly Funding Trend for a specific selected year
                    st.markdown(f"###### Monthly Funding Trend ({selected_year_option})")
                    monthly_funding = filtered_df.groupby('month_name')['amount_in_usd'].sum().reset_index()
                    # Ensure months are sorted chronologically, not alphabetically
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    monthly_funding['month_name'] = pd.Categorical(monthly_funding['month_name'], categories=month_order, ordered=True)
                    monthly_funding = monthly_funding.sort_values('month_name')
                    fig = create_plotly_chart(px.line, monthly_funding, 'month_name', 'amount_in_usd', 
                                              title=None, x_label='Month', y_label='Total Funding (USD)', 
                                              template=plotly_template)
                    if fig: fig.update_traces(mode='lines+markers'); st.plotly_chart(fig, use_container_width=True)
            with t1_col2:
                # Display Yearly Funding Count (Number of Deals) if "All Years" is selected
                if selected_year_option == "All Years":
                    st.markdown("###### Yearly Funding Count (Number of Deals)")
                    yearly_counts = filtered_df.groupby('year')['startup_name'].nunique().reset_index()
                    fig = create_plotly_chart(px.bar, yearly_counts, 'year', 'startup_name', 
                                              title=None, x_label='Year', y_label='Number of Deals', 
                                              template=plotly_template, text_auto=True) # Show exact deal numbers on bars
                    if fig: st.plotly_chart(fig, use_container_width=True)
                else: # Display Monthly Funding Count for a specific selected year
                    st.markdown(f"###### Monthly Funding Count ({selected_year_option})")
                    monthly_counts = filtered_df.groupby('month_name')['startup_name'].nunique().reset_index()
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 
                    monthly_counts['month_name'] = pd.Categorical(monthly_counts['month_name'], categories=month_order, ordered=True)
                    monthly_counts = monthly_counts.sort_values('month_name')
                    fig = create_plotly_chart(px.line, monthly_counts, 'month_name', 'startup_name', 
                                              title=None, x_label='Month', y_label='Number of Deals', 
                                              template=plotly_template)
                    if fig: fig.update_traces(mode='lines+markers'); st.plotly_chart(fig, use_container_width=True)
        
        # --- Tab 2: Sector & Investment Type Analysis ---
        # This tab breaks down funding by industry and investment type, and shows funding amount distribution.
        with tab2:
            st.markdown(f"#### Sector & Investment Type Analysis for {current_time_period_title}")
            t2_col1, t2_col2 = st.columns(2) # Layout in two columns
            with t2_col1: # Top Funded Industries
                # Check if industry data exists and is not all 'Unknown'
                if 'industry_vertical' in filtered_df.columns and not filtered_df['industry_vertical'].isin(['Unknown', pd.NA]).all():
                    st.markdown("###### Top Funded Industries")
                    industry_f = filtered_df.groupby('industry_vertical')['amount_in_usd'].sum().nlargest(10).sort_values(ascending=True) # Sort for horizontal bar
                    fig = create_plotly_chart(px.bar, industry_f.reset_index(), 'amount_in_usd', 'industry_vertical', 
                                              title=None, x_label='Total Funding (USD)', y_label='Industry', 
                                              template=plotly_template, orientation='h', text_auto='.2s')
                    if fig: st.plotly_chart(fig, use_container_width=True)
            with t2_col2: # Funding by Investment Type
                # Check if investment type data exists and is not all 'Unknown'
                if 'investment_type' in filtered_df.columns and not filtered_df['investment_type'].isin(['Unknown', pd.NA]).all():
                    st.markdown("###### Funding by Investment Type")
                    inv_type_f = filtered_df.groupby('investment_type')['amount_in_usd'].sum().nlargest(10).sort_values(ascending=False)
                    fig = create_plotly_chart(px.bar, inv_type_f.reset_index(), 'investment_type', 'amount_in_usd', 
                                              title=None, x_label="Investment Type", y_label="Total Funding (USD)", 
                                              template=plotly_template, text_auto='.2s')
                    if fig: st.plotly_chart(fig, use_container_width=True)

            st.markdown("---") # Separator
            # Distribution of Funding Amounts (Histogram)
            st.markdown("###### Distribution of Funding Amounts (Log Scale for X-axis)")
            # Filter for amounts > $1000 for a more meaningful log-scale histogram
            funding_for_hist = filtered_df[filtered_df['amount_in_usd'] > 1000]['amount_in_usd'] 
            if not funding_for_hist.empty:
                fig_hist = create_plotly_chart(px.histogram, funding_for_hist.to_frame(), x_col='amount_in_usd', y_col=None, # y_col=None for 1D histogram
                                               title=None, x_label="Funding Amount (USD)", y_label="Number of Deals",
                                               template=plotly_template, log_x=True, nbinsx_hist=50) # log_x for skewed data, suggest 50 bins
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
            else: 
                st.info("Not enough data points (with funding > $1000) to display funding amount distribution for the current filters.")
        
        # --- Tab 3: Investor Activity ---
        # This tab shows top funded startups and top investors for the current filters.
        with tab3:
            st.markdown(f"#### Investor Activity for {current_time_period_title}")
            if 'investors_name' in filtered_df.columns and not filtered_df['investors_name'].isin(['Unknown',pd.NA]).all():
                # Top Funded Startups (based on current filters)
                st.markdown("###### Top Funded Startups (Current Filters)")
                top_s_filt = filtered_df.groupby('startup_name')['amount_in_usd'].sum().nlargest(10).reset_index()
                top_s_filt['Amount (USD)'] = top_s_filt['amount_in_usd'].map('${:,.0f}'.format) # Format amount as currency string
                top_s_filt.index = np.arange(1, len(top_s_filt) + 1) # Create a 1-based index for ranking
                if not top_s_filt.empty: 
                    # Display DataFrame of top startups
                    st.dataframe(top_s_filt[['startup_name', 'Amount (USD)']].rename(columns={'startup_name': 'Startup Name'}), use_container_width=True)
                else: 
                    st.info("No startup data available for the current filters.")
                st.markdown("---") # Separator
                
                # Top Investors by Number of Deals
                st.markdown("###### Top Investors by Number of Deals (Current Filters)")
                valid_investors = filtered_df['investors_name'].dropna().astype(str) # Ensure strings and no NaNs
                if not valid_investors.empty:
                    # Process investor names: split by comma/semicolon, explode into separate rows, strip whitespace, remove empty/NaNs
                    inv_counts = valid_investors.str.split(r',|;\s*').explode().str.strip().replace('', np.nan).dropna()
                    # Filter out generic or uninformative investor names (case-insensitive)
                    inv_counts = inv_counts[~inv_counts.str.lower().isin(['undisclosed investors', 'undisclosed', 'others', 'nan', '', 'na'])]
                    inv_counts = inv_counts.value_counts().nlargest(15) # Get the top 15 investors by deal count
                    if not inv_counts.empty:
                        df_ic = inv_counts.reset_index(); df_ic.columns = ['investor', 'deals'] # Prepare DataFrame for chart
                        # Create horizontal bar chart for top investors (better for long names)
                        fig = create_plotly_chart(px.bar, df_ic, 'deals', 'investor', 
                                                  title=None, x_label="Number of Deals", y_label="Investor Name", 
                                                  template=plotly_template, orientation='h', text_auto=True)
                        if fig: 
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort investors by deal count
                            st.plotly_chart(fig, use_container_width=True)
                    else: 
                        st.info("No specific investor data found for current filters after cleaning generic terms.")
                else: 
                    st.info("No investor names present in the filtered data.")
            else: 
                st.info("Investor name column not available or is empty for the current filters.")

        # --- Tab 4: Geographical Insights ---
        # This tab shows funding distribution across cities.
        with tab4:
            st.markdown(f"#### Geographical Insights for {current_time_period_title}")
            if 'city_location' in filtered_df.columns and not filtered_df['city_location'].isin(['Unknown',pd.NA]).all():
                # Aggregate funding data by city: total funding, deal count, average deal size
                city_analysis = filtered_df.groupby('city_location')['amount_in_usd'].agg(['sum', 'count', 'mean']).reset_index()
                city_analysis = city_analysis.rename(columns={'sum': 'Total Funding', 'count': 'Deal Count', 'mean': 'Average Deal Size'})
                
                st.markdown("###### Top Cities by Total Funding")
                top_cities_funding = city_analysis.sort_values(by='Total Funding', ascending=False).nlargest(10, 'Total Funding')
                fig_cf = create_plotly_chart(px.bar, top_cities_funding, 'city_location', 'Total Funding', 
                                             title=None, x_label='City', y_label='Total Funding (USD)', 
                                             template=plotly_template, text_auto='.2s')
                if fig_cf: st.plotly_chart(fig_cf, use_container_width=True)

                st.markdown("###### Top Cities by Number of Deals")
                top_cities_deals = city_analysis.sort_values(by='Deal Count', ascending=False).nlargest(10, 'Deal Count')
                fig_cd = create_plotly_chart(px.bar, top_cities_deals, 'city_location', 'Deal Count', 
                                             title=None, x_label='City', y_label='Number of Deals', 
                                             template=plotly_template, text_auto=True)
                if fig_cd: st.plotly_chart(fig_cd, use_container_width=True)
            else: 
                st.info(f"City location data not available or is empty for the current view.")
        
        # --- Data Export Section ---
        st.markdown("---") # Visual separator
        st.subheader("📥 Export Filtered Data")
        if not filtered_df.empty:
            @st.cache_data # Cache the CSV conversion function for better performance on repeated calls with same data
            def convert_df_to_csv(df_to_c): 
                return df_to_c.to_csv(index=False).encode('utf-8') # Convert DataFrame to CSV string, then encode to bytes
            csv_d = convert_df_to_csv(filtered_df)
            # Generate a dynamic filename based on current filters for the downloaded CSV
            f_name_period = "AllYears" if selected_year_option == "All Years" else str(selected_year_option)
            st.download_button(
                "Download Data as CSV", 
                csv_d, 
                f"startup_investments_{f_name_period}_{selected_city}_{selected_investment}.csv".replace('__','_').replace('All','any').replace('/','_').replace(' ','_'), # Sanitize filename
                'text/csv' # MIME type for CSV files
            )

    # --- Advanced Analysis Tools in Sidebar (using Expanders for a cleaner look) ---
    st.sidebar.markdown("---") # Separator in sidebar
    st.sidebar.header("🛠️ Advanced Tools")
    
    # Data Profiling Tool Expander
    profile_sb = st.sidebar.expander("📊 Data Profile (Full Dataset)")
    with profile_sb:
        if st.button("Generate Profile", key="ydata_sb_button"): # Unique key for this button
            if not df.empty: # Ensure there's data to profile
                with st.spinner("Generating data profile report... This may take a moment."): # Show progress
                    try:
                        report_f = generate_profile(df, "Full_Dataset_Profile") # Generate the report
                        st.success(f"✅ Profile report '{report_f}' generated!")
                        st.session_state['profile_to_show'] = report_f # Store filename in session state to display in main area
                    except Exception as e: 
                        st.error(f"Error generating profile: {e}")
            else: 
                st.warning("Original dataset is empty, cannot generate profile.")
    
    # AI Summary for Full Dataset Expander
    ai_full_sb = st.sidebar.expander("🤖 AI Summary (Full Dataset)")
    with ai_full_sb:
        if st.button("Summarize Full Data", key="ai_full_sb_button"): # Unique key
            if not df.empty:
                if not os.getenv('OPENROUTER_API_KEY'): # Check for API key
                    st.warning("OPENROUTER_API_KEY is not set. Cannot generate AI summary.")
                else:
                    with st.spinner("AI is summarizing the full dataset..."): # Show progress
                        summary = generate_deepseek_summary(df, "the full dataset") 
                        st.session_state['ai_summary_to_show'] = summary # Store summary to display
                        st.success("Full dataset AI summary is ready!")
            else: 
                st.warning("Original dataset is empty.")

    # AI Summary for Filtered Dataset Expander
    ai_filtered_sb = st.sidebar.expander("🤖 AI Summary (Filtered Data)")
    with ai_filtered_sb:
        if filtered_df.empty: # Check if there's any filtered data to summarize
            st.info("No data is currently filtered to summarize.")
        elif st.button("Summarize Filtered Data", key="ai_filtered_sb_button"): # Unique key
            if not os.getenv('OPENROUTER_API_KEY'): 
                st.warning("OPENROUTER_API_KEY is not set.")
            else:
                with st.spinner("AI is summarizing the filtered data..."):
                    # Create a descriptive context string for the AI based on current filters
                    filter_desc = f"data for {current_time_period_title}"
                    if selected_city != "All": filter_desc += f", city: {selected_city}"
                    if selected_investment != "All": filter_desc += f", investment type: {selected_investment}"
                    summary = generate_deepseek_summary(filtered_df, f"the filtered dataset ({filter_desc})")
                    st.session_state['ai_summary_to_show'] = summary # Store summary to display
                    st.success("Filtered data AI summary is ready!")

    # --- Display Area in Main Content for Reports/Summaries generated from Sidebar Tools ---
    # This allows showing the generated content without navigating away or complex layouts.
    
    # Display the generated data profile report if its filename is in session state
    if 'profile_to_show' in st.session_state and st.session_state['profile_to_show']:
        st.markdown("---") # Separator
        st.subheader("📋 Generated Data Profile Report")
        try:
            with open(st.session_state['profile_to_show'], 'r', encoding='utf-8') as f: 
                html_c = f.read() # Read HTML content from the generated file
            components.html(html_c, height=1000, scrolling=True) # Embed HTML report in the app
            # Button to clear the displayed profile report from view
            if st.button("Clear Profile View", key="clear_profile_view_button"): 
                st.session_state['profile_to_show'] = None # Remove from session state
                st.rerun() # Rerun to update the view
        except Exception as e: 
            st.error(f"Could not display profile report: {e}")
            st.session_state['profile_to_show'] = None # Clear on error to prevent repeated attempts

    # Display the generated AI summary if it's in session state
    if 'ai_summary_to_show' in st.session_state and st.session_state['ai_summary_to_show']:
        st.markdown("---") # Separator
        st.subheader("🧠 AI Generated Summary")
        st.markdown(st.session_state['ai_summary_to_show']) # Display the summary text
        # Button to clear the displayed AI summary from view
        if st.button("Clear AI Summary View", key="clear_ai_summary_view_button"): 
            st.session_state['ai_summary_to_show'] = None # Remove from session state
            st.rerun() # Rerun to update the view

# Entry point for the Streamlit application: this runs when the script is executed
if __name__ == "__main__":
    try: 
        main() # Call the main function to start the app
    except Exception as e: 
        # Catch any critical, unhandled errors in the main app flow and display them
        st.error(f"A critical error occurred in the application: {e}")
        st.exception(e) # Display the full traceback for easier debugging