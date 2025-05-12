import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

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
            'Investment Type': 'investment_type'
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

def main():
    try:
        # App title and description
        st.title("Startup Investment Analysis in India")
        st.markdown("Exploring startup funding trends from 2001 to 2024")

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

        # Apply filters
        filtered_df = df[df['year'] == selected_year]
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df['city_location'] == selected_city]

        # Main Dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Investment", f"${filtered_df['amount_in_usd'].sum():,.0f}")
        
        with col2:
            st.metric("Number of Startups", filtered_df['startup_name'].nunique())

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
        st.dataframe(top_startups.reset_index(), use_container_width=True)

        # Monthly Funding Trend
        st.subheader("Monthly Funding Trend")
        monthly_funding = (
            filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['amount_in_usd']
            .sum()
        )
        monthly_funding.index = monthly_funding.index.astype(str)
        
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

    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {e}")
        # Log the full error for debugging
        st.error(f"Error details: {sys.exc_info()}")

if __name__ == "__main__":
    main()