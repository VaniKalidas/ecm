import streamlit as st
from datetime import datetime
import pandas as pd
from scipy import stats
import numpy as np
import holidays
import itertools
import matplotlib.pyplot as plt
from prophet import Prophet 
from prophet.diagnostics import performance_metrics, cross_validation


class RetailForecastApp:
    """Main application class for retail sales forecasting."""
    
    def __init__(self):
        self.df = None
        self.pivot = None
        self.metrics = {}
        
    def load_data(self, dataset):
        """Load data from CSV file."""
        try:
            df = pd.read_csv(f'datasets/{dataset}')
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            return df
        except FileNotFoundError:
            st.error(f"Dataset '{dataset}' not found in datasets folder.")
            return None
    
    def get_missing_data(self, df):
        """Calculate missing data statistics."""
        total_null_sum = df.isnull().sum()
        total_null_count = df.isnull().count()
        null_percent = total_null_sum / total_null_count * 100
        null_table = pd.concat([total_null_sum, null_percent], axis=1, 
                              keys=['Total', 'Percent'])
        return pd.DataFrame(null_table)
    
    def calculate_mape(self, actual_val, predict_val):
        """Calculate Mean Absolute Percentage Error."""
        actual_val = np.array(actual_val)
        predict_val = np.array(predict_val)
        mape = np.mean(np.abs((actual_val - predict_val) / actual_val)) * 100
        return mape
    
    def setup_sidebar(self):
        """Setup sidebar navigation with buttons."""
        st.sidebar.header("DrivenSales")
        st.sidebar.write()
        st.sidebar.write("Navigation")
        # Use buttons for navigation
        if st.sidebar.button("📊 Raw Data", use_container_width=True):
            st.session_state.current_page = "Raw Data"
        if st.sidebar.button("🔍 Data Insights", use_container_width=True):
            st.session_state.current_page = "Data Insights"
        if st.sidebar.button("📈 Forecasting", use_container_width=True):
            st.session_state.current_page = "Forecasting"
        
        # Initialize session state if not exists
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Raw Data"
            
        return st.session_state.current_page
    
    def show_raw_data(self, categories):
        """Display raw data page."""
        st.title("Raw Data")
        
        # Category selector
        selected_category = st.selectbox("Select Category", categories)
        st.write("")
        st.subheader("Time Series Data")
        
        if selected_category == "ALL":
            st.write("Sample Sales Data")
            st.write(self.pivot.head(15))
            st.write("")
            st.write("All Sales By Category")
            st.line_chart(self.pivot, x_label="Date of Sales", y_label="Number of Sales")
        else:
            if selected_category in self.pivot.columns:
                st.write(f"Sample Sales Data For {selected_category}")
                series = self.pivot[[selected_category]]
                st.write(series.head(15))
                st.write(f"All Sales For {selected_category}")
                st.line_chart(series, x_label="Date of Sales", y_label="Number of Sales")
            else:
                st.warning("Selected category not found in the data")
    
    def show_data_insights(self):
        """Display data insights page."""
        st.title("Data Insights")
        st.subheader("Dates in Dataset")
        st.write(f"Data is from {self.df['date'].min()} to {self.df['date'].max()}.")
        
        # Limit date range to dates with more data
        end_date = datetime.strptime(self.df['date'].max(), "%Y-%m-%d")
        selected_years = st.slider("Select number of years to include in analysis", 
                                  1, 5, 2)
        start_date = end_date.replace(year=end_date.year - selected_years)
        st.write(f"To ensure sufficient data for analysis, the date range will be limited to {start_date.date()} to {end_date.date()}.")
        
        self.pivot = self.pivot[
            (self.pivot.index >= str(start_date.date())) & 
            (self.pivot.index <= str(end_date.date()))
        ]        
        st.write("")

        # Missing data analysis
        st.subheader("Missing Data Analysis")
        st.write("The table below shows the total number and percentage of missing values for each category.")
        st.write(self.get_missing_data(self.pivot).head(33))
        
        if self.get_missing_data(self.pivot).isnull().sum().sum() == 0:
            st.write("No missing data found in the dataset.")
        else:
            st.write("Missing data found in the dataset. Values will be filled with 0 for analysis purposes.")
            self.df = self.df.fillna(0)
        st.write("")

        # Low sales categories
        total_num_categories = self.pivot.shape[1]
        low_sales_categories = self.pivot.mean()[self.pivot.mean() < 1000]
        
        st.subheader("Low Sales Categories")
        st.write(f"There are {len(low_sales_categories)} out of {total_num_categories} categories with average daily sales less than 1000 units.")
        
        for category in low_sales_categories.index:
            st.write(f"{category}: Average Daily Sales = {low_sales_categories[category]:.2f} units")
        
        st.write("These categories are being dropped due to insufficient sales volume for reliable forecasting.")
        self.pivot = self.pivot.drop(columns=low_sales_categories.index)
        st.write("")

        # Extended zero sales periods
        st.subheader("Categories with Extended Zero Sales Periods")
        st.write("The table below shows the percentage of rows with zero sales for each category.")
        st.write((self.pivot == 0).astype(int).sum(axis=0) / len(self.pivot.sum(axis=0)))
        
        long_zero_sales_categories = (self.pivot == 0).astype(int).sum(axis=0) / len(self.pivot.sum(axis=0))
        long_zero_sales_categories = long_zero_sales_categories[long_zero_sales_categories > 0.5]
        
        st.write(f"There are {len(long_zero_sales_categories)} out of {total_num_categories} categories with more than 50% of days having zero sales.")
        
        for category in long_zero_sales_categories.index:
            st.write(f"{category}: {long_zero_sales_categories[category]*100:.2f}% days with zero sales")
        
        st.write("These categories are being dropped due to extended periods of zero sales affecting forecasting accuracy.")
        self.pivot = self.pivot.drop(columns=long_zero_sales_categories.index)
        st.write("")

        # Outlier removal
        z_scores = np.abs(stats.zscore(self.pivot, nan_policy='omit'))
        outlier_rows = np.where(z_scores > 2.7)[0]
        
        st.subheader("Outlier Removal")
        st.write("Number of outlier rows being removed by category:")
        
        for i, category in enumerate(self.pivot.columns):
            num_outliers = np.sum(z_scores[:, i] > 2.7)
            st.write(f"{category}: {num_outliers} outlier rows")
        
        st.write(f"Removing {len(outlier_rows)} rows with outliers based on z-score method.")
        st.write("The remaining rows are 3 standard deviations from the mean with 99.7% confidence.")
        self.pivot = self.pivot.drop(self.pivot.index[outlier_rows])
        st.write("")

        # Store cleaned data in session state for forecasting page
        st.session_state.cleaned_pivot = self.pivot
        st.success("Data preprocessing completed! Ready for forecasting.")
    
    def show_forecasting(self, categories):
        """Display forecasting page."""
        st.title("Retail Store Inventory and Demand Forecasting")
        
        # Use cleaned data from data insights if available
        if hasattr(st.session_state, 'cleaned_pivot'):
            forecasting_pivot = st.session_state.cleaned_pivot
            st.success("Using preprocessed data from Data Insights page")
        else:
            forecasting_pivot = self.pivot
            st.warning("Using raw data. For better results, preprocess data in Data Insights page first.")
        
        # Category selector
        selected_category = st.selectbox("Select Category", categories)
        st.write("")
        st.subheader("Sales Forecasting")
        
        if selected_category == "ALL":
            st.warning("Forecasting for all categories is inaccurate. Please select a specific category.")
            return
        
        # if selected_category is available in database
        if selected_category in forecasting_pivot.columns:
            st.write(f"Sales Forecasting For {selected_category}")
            series = forecasting_pivot[[selected_category]].reset_index()
            series.columns = ['ds', 'y'] # time = y axis, sales = x axis
            
            # Train Prophet model
            model = Prophet(
                yearly_seasonality=True,  
                weekly_seasonality=True
            )
            model.fit(series)
            
            # Create future dataframe
            selected_period = st.slider(
                "Select number of days to forecast", 
                5, 120, 90, step=5
            )
            
            future = model.make_future_dataframe(periods=selected_period)
            forecast = model.predict(future)
            
            # Plot forecast
            fig = model.plot(forecast)
            plt.ylabel("Sales")
            plt.xlabel("Date")
            plt.title(f"Sales Forecast for {selected_category}")
            st.pyplot(fig)
            
            # Show forecast components
            with st.expander("Show Forecast Components"):
                comp_fig = model.plot_components(forecast)
                st.pyplot(comp_fig)
                
        else:
            st.warning("Selected category not available for forecasting. Please choose from available categories.")
    
    def run(self):
        """Main application runner."""
        st.set_page_config(layout="wide", page_title="Retail Sales Forecast")
        
        # Load data
        self.df = self.load_data("train.csv")
        if self.df is None:
            st.error("Failed to load data. Please check if the dataset exists.")
            return
        
        # Prepare data
        sales_by_date = self.df.groupby(['date', 'family'])['sales'].sum().reset_index()
        self.pivot = sales_by_date.pivot(index='date', columns='family', values='sales')
        categories = np.insert(self.df['family'].unique(), 0, "ALL")
        
        # Setup navigation
        current_page = self.setup_sidebar()
        
        # Display appropriate page
        if current_page == "Raw Data":
            self.show_raw_data(categories)
        elif current_page == "Data Insights":
            self.show_data_insights()
        elif current_page == "Forecasting":
            self.show_forecasting(categories)


if __name__ == "__main__":
    app = RetailForecastApp()
    app.run()
