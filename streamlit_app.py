import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import sys
from ydata_profiling import ProfileReport
import requests
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()
def generate_prompt_from_dataframe(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    sample_rows = df.head(3).to_string(index=False)

    prompt = f"""You are a data analyst. Analyze the following dataset and provide a concise summary of:
- Key trends and patterns
- Outliers or anomalies
- Summary in bullet points

The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
Numeric columns: {', '.join(num_cols) if num_cols else 'None'}
Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}

Here are the first few rows:
{sample_rows}
"""
    return prompt


def generate_deepseek_summary(data):
    
    API_KEY = os.getenv('OPENROUTER_API_KEY')
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Update for Streamlit sharing
        "X-Title": "Data Summary App"
    }

    messages = [
        {
            "role": "user",
            "content": f"Give a concise summary of this dataset:\n{data.head(10).to_csv(index=False)}"
        }
    ]

    payload = {
        "model": "deepseek/deepseek-r1:free",  # Or correct model ID from OpenRouter
        "messages": messages
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        print("Status Code:", response.status_code)
        print("Raw Response Text:", response.text)

        if response.status_code == 200:
            try:
                result = response.json()
                return result['choices'][0]['message']['content']
            except ValueError:
                return f"Invalid JSON: {response.text}"
        else:
            return f"Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Error occurred while calling DeepSeek via OpenRouter API: {e}"


def generate_profile(df):
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, title="Startup Data Summary", explorative=True)
    profile.to_file("summary_report.html")


#Adding a dark/light theme toggle in streamlit

import streamlit as st

# Theme toggle
theme = st.radio("🌗 Choose Theme", ["Light", "Dark"], horizontal=True, key="theme_toggle")

# Apply dark theme only if selected
if theme == "Dark":
    st.markdown("""
    <style>
        /* Overall background */
        .stApp {
            background-color: #0e1117;
            color: #f5f5f5;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }

        /* Widget containers */
        .css-1v0mbdj, .css-1d391kg, .css-1cpxqw2, .stSelectbox, .stRadio {
            background-color: #1f2937 !important;
            color: #f5f5f5 !important;
        }

        /* All headings and paragraph text */
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #f5f5f5 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #30363d;
            color: white;
            border: none;
        }

        /* Charts and plot backgrounds (like Plotly/Altair) */
        .element-container iframe {
            background-color: #0e1117 !important;
        }

        /* Dropdown arrows and text */
        .stSelectbox>div>div {
            background-color: #1f2937 !important;
            color: #f5f5f5 !important;
        }

        /* Fix for metric, expander, etc. */
        .css-1cpxqw2, .css-qrbaxs {
            background-color: #1f1f1f !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)


def load_and_clean_data():
    """
    Load and clean startup funding data with comprehensive error handling
    """
    try:
        # Use the full path to the CSV file
        file_path = '/home/shubhayu/Desktop/startup-investment-analysis/data/cleaned_startup_funding.csv'
        
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='latin1')

        # Rename columns to be more Python-friendly
        df = df.rename(columns={
            'Sr no': 'sr_no',
            'Date': 'date',
            'Startup name': 'startup_name',
            'City Location': 'city_location',
            'Investment Type': 'investment_type',
            'Investment Name': 'investors_name'
        })

        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Clean and ensure numeric amount
        df['amount_in_usd'] = pd.to_numeric(df['amount_in_usd'], errors='coerce').fillna(0)

        # Clean city names
        df['city_location'] = df['city_location'].astype(str).str.strip().str.title().replace({
            'Delhi': 'New Delhi', 
            'Bangalore': 'Bengaluru',
            'Mumbai / Pune': 'Mumbai'
        })

        # Fill NAs in important columns
        df['startup_name'] = df['startup_name'].fillna('Unknown')
        df['industry_vertical'] = df['industry_vertical'].fillna('Miscellaneous')

        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        # Log the full error for debugging
        st.error(f"Error details: {sys.exc_info()}")
        return pd.DataFrame()

def create_bar_chart(data, x_col, y_col, title, x_label, y_label):
    """
    Create a Plotly bar chart with error handling
    """
    try:
        # Convert data to DataFrame if it's a Series
        if isinstance(data, pd.Series):
            data = data.reset_index()
        
        # Check if data is empty
        if len(data) == 0:
            st.warning("No data available for the chart")
            return None
        
        # Create bar chart
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col,
            labels={x_col: x_label, y_col: y_label},
            title=title
        )
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart: {e}")
        return None

def create_line_chart(data, x_col, y_col, title, x_label, y_label):
    """
    Create a Plotly line chart with error handling
    """
    try:
        # Convert data to DataFrame if it's a Series
        if isinstance(data, pd.Series):
            data = data.reset_index()
        
        # Check if data is empty
        if len(data) == 0:
            st.warning("No data available for the chart")
            return None
        
        # Create line chart
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            labels={x_col: x_label, y_col: y_label},
            title=title
        )
        return fig
    except Exception as e:
        st.error(f"Error creating line chart: {e}")
        return None

import streamlit.components.v1 as components

# def generate_profile():
#     from ydata_profiling import ProfileReport
#     profile = ProfileReport(df, title="Data Summary", explorative=True)
#     profile.to_file("summary_report.html")

def show_profile():
    with open("summary_report.html", 'r', encoding='utf-8') as f:
        html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)


def main():
    try:
        # App title and description
        st.title("Startup Investment Analysis in India")
        st.markdown("Exploring startup funding trends from 2015 to 2020")

        # Load the data
        df = load_and_clean_data()

        if df.empty:
            st.error("Could not load the dataset. Please check the data source.")
            return

        # Sidebar filters
        st.sidebar.header("Filters")

        # Year filter
        years = sorted(df['year'].dropna().unique())
        selected_year = st.sidebar.selectbox("Select Year", options=years)

        # Dynamically filter cities with data for the selected year
        year_df = df[df['year'] == selected_year]
        cities_with_data = sorted(year_df.groupby('city_location')
                                   .filter(lambda x: x['amount_in_usd'].sum() > 0)['city_location']
                                   .unique())

        # City filter with only cities that have data
        selected_city = st.sidebar.selectbox("Select City", options=["All"] + list(cities_with_data))
        # Investment Type filter
        investment_types = df['investment_type'].dropna().unique()
        selected_investment = st.sidebar.selectbox("Select Investment Type", options=["All"] + list(investment_types))

        # Apply investment filter
        if selected_investment != "All":
            filtered_df = filtered_df[filtered_df['investment_type'] == selected_investment]


        # Apply filters
        filtered_df = df[df['year'] == selected_year]
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df['city_location'] == selected_city]

        # Main Dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Investment", f"${filtered_df['amount_in_usd'].sum():,.0f}")
        
        with col2:
            st.metric("Number of Startups", filtered_df['startup_name'].nunique())

        with col3:
            top_funded = filtered_df.groupby('startup_name')['amount_in_usd'].sum().idxmax()
            st.metric("Top Funded Startup", top_funded)


        # Top Funded Industries
        st.subheader("Top Funded Industries")
        industry_funding = (
            filtered_df.groupby('industry_vertical')['amount_in_usd']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        
        # Create and display industry funding chart
        industry_fig = create_bar_chart(
            industry_funding.reset_index(), 
            'industry_vertical', 
            'amount_in_usd', 
            'Top 10 Industries by Funding', 
            'Industry', 
            'Total Funding (USD)'
        )
        if industry_fig:
            st.plotly_chart(industry_fig)

        # Top Funded Startups
        st.subheader("Top Funded Startups")
        top_startups = (
            filtered_df.groupby('startup_name')['amount_in_usd']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        # Add a 'Ranking' column starting from 1
        top_startups = top_startups.reset_index()
        top_startups['Ranking'] = top_startups.index + 1

        # Reorder the columns to display 'Ranking' first and reset the index to avoid the default index column
        top_startups = top_startups[['Ranking', 'startup_name', 'amount_in_usd']]

        # Set 'Ranking' as the index to remove the default index column
        top_startups.set_index('Ranking', inplace=True)

        st.dataframe(top_startups, use_container_width=True)


        # Monthly Funding Trend
        st.subheader("Monthly Funding Trend")
        monthly_funding = (
            filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['amount_in_usd']
            .sum()
        )
        monthly_funding.index = monthly_funding.index.astype(str)
        # Funding Count Trend
        st.subheader("Monthly Funding Count")
        monthly_funding_count = (
            filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['startup_name']
            .count()
        )
        monthly_funding_count.index = monthly_funding_count.index.astype(str)

        monthly_count_fig = create_line_chart(
            monthly_funding_count.reset_index(),
            'date',
            'startup_name',
            'Monthly Funding Count',
            'Month',
            'Number of Fundings'
        )
        if monthly_count_fig:
            st.plotly_chart(monthly_count_fig)

        st.subheader("City-wise Funding Comparison")

        city_funding = (
            df[df['year'] == selected_year]
            .groupby('city_location')['amount_in_usd']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        city_fig = create_bar_chart(
            city_funding.reset_index(),
            'city_location',
            'amount_in_usd',
            'Top 10 Cities by Funding',
            'City',
            'Total Funding (USD)'
        )
        if city_fig:
            st.plotly_chart(city_fig)


        # Create and display monthly funding trend chart
        monthly_fig = create_line_chart(
            monthly_funding.reset_index(), 
            'date', 
            'amount_in_usd', 
            'Monthly Funding Trend', 
            'Month', 
            'Total Funding (USD)'
        )
        if monthly_fig:
            st.plotly_chart(monthly_fig)

        st.subheader("📥 Export Filtered Data")

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_df)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"startup_filtered_{selected_year}.csv",
            mime='text/csv',
        )

        top_investors = (
            filtered_df['investors_name'].dropna()
            .str.split(', ')
            .explode()
            .value_counts()
            .head(10)
        )

        # Reset the index and rename columns
        top_investors = top_investors.reset_index()
        top_investors.columns = ['investor_name', 'investment_count']

        # Create the bar chart
        create_bar_chart(top_investors, 'investor_name', 'investment_count', 'Top 10 Investors', 'Investor', 'Investment Count')



        st.markdown("## 📊 Auto-generated Profile Report")

        if st.button("Generate Data Summary"):
            generate_profile(df)
            st.success("✅ Summary report generated!")

        if os.path.exists("summary_report.html"):
            with open("summary_report.html", 'r', encoding='utf-8') as f:
                html_content = f.read()
                components.html(html_content, height=1000, scrolling=True)

        if st.button("Generate Data Summary Using DeepSeek"):
            summary = generate_deepseek_summary(df)
            st.success("✅ Summary generated!")
            st.write(summary)


    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {e}")
        # Log the full error for debugging
        st.error(f"Error details: {sys.exc_info()}")

if __name__ == "__main__":
    main()