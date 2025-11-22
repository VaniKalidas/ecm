import streamlit as st
from datetime import datetime
import pandas as pd
from prophet import Prophet 
from scipy import stats 
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import ollama
import tempfile
import base64
import os

# Load data from dataset
def loadData(dataset):
    df = pd.read_csv(f'datasets/{dataset}')
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

# Clean data in dataset
def getMissingData(df):
    totalNullSum = df.isnull().sum()
    totalNullCount = df.isnull().count()
    nullPercent = totalNullSum / totalNullCount * 100
    nullTable = pd.concat([totalNullSum, nullPercent], axis = 1, keys = ['Total', 'Percent'])
    return pd.DataFrame(nullTable)

# Train data using train dataset


# Test data using test dataset
def calculateMAPE(actualVal, predictVal):
    actualVal = np.array(actualVal)
    predictVal = np.array(predictVal)
    MAPE = np.mean(np.abs((actualVal - predictVal) / actualVal)) * 100
    return MAPE

if __name__ == "__main__":
    st.set_page_config(layout="wide") # use full width of page
    st.sidebar.header("Pages")
    setting = st.sidebar.selectbox("Setting", ("Raw Data", "Data Insights", "Forecasting"))
    # side bar menu options with buttons for raw data, data insights, forecasting
    #selectedSetting = "Raw Data"
    #if selectedSetting == "Raw Data":
    #    setting = "Raw Data"
    #if selectedSetting == "Data Insights":
    #    setting = "Data Insights"
    #if selectedSetting == "Forecasting":
    #    setting = "Forecasting"


    df = loadData("train.csv")
    categories = np.insert(df['family'].unique(), 0, "ALL")
    # Aggregate sales per date and family, then pivot to wide format
    salesByDate = df.groupby(['date', 'family'])['sales'].sum().reset_index()
    pivot = salesByDate.pivot(index='date', columns='family', values='sales')

    if setting == "Raw Data":
        st.title("Raw Data")
        # Category selector
        selectedCategory = st.selectbox("Select Category", categories)
        st.write("")
        st.subheader("Time Series Data")
        if selectedCategory == "ALL": # if all data is selected
            st.write("Sample Sales Data")
            st.write(pivot.head(15))
            st.write("")
            st.write("")
            st.write("All Sales By Category")
            st.line_chart(pivot, x_label="Date of Sales", y_label="Number of Sales")
        else: # if a category is slected
            if selectedCategory in pivot.columns:
                st.write(f"Sample Sales Data For {selectedCategory}")
                series = pivot[[selectedCategory]]
                st.write(series.head(15))
                st.write(f"All Sales For {selectedCategory}")
                st.line_chart(series, x_label="Date of Sales", y_label="Number of Sales")
            else:
                st.warning("Selected category not found in the data")

    if setting == "Data Insights":
        st.title("Data Insights")
        st.subheader("Dates in Dataset")
        st.write(f"Data is from {df['date'].min()} to {df['date'].max()}.")
        # limit date range to dates with more data
        # get number of years in dataset from end date
        endDate = datetime.strptime(df['date'].max(), "%Y-%m-%d")
        selectedYears = st.slider("Select number of years to include in analysis", 1, 5, 3)
        startDate = endDate.replace(year=endDate.year - selectedYears)
        st.write(f"To ensure sufficient data for analysis, the date range will be limited to {startDate.date()} to {endDate.date()}.")
        pivot = pivot[(pivot.index >= str(startDate.date())) & (pivot.index <= str(endDate.date()))]        
        st.write("")

        # how much data is N/A
        st.subheader("Missing Data Analysis")
        st.write("The table below shows the total number and percentage of missing values for each category.")
        st.write(getMissingData(pivot).head(33))
        if getMissingData(pivot).isnull().sum().sum() == 0:
            st.write("No missing data found in the dataset.")
        else:
            st.write("Missing data found in the dataset. Values will be filled with 0 for analysis purposes.")
            df = df.fillna(0)
        st.write("")

        # categories that have average daily sales less than $1000
        totalNumCategories = pivot.shape[1]
        lowSalesCategories = pivot.mean()[pivot.mean() < 1000]
        st.subheader("Low Sales Categories")
        st.write(f"There are {len(lowSalesCategories)} out of {totalNumCategories} categories with average daily sales less than 1000 units.")
        for category in lowSalesCategories.index:
            st.write(f"{category}: Average Daily Sales = {lowSalesCategories[category]:.2f} units")
        st.write("These categories are being dropped due to insufficient sales volume for reliable forecasting.")
        pivot = pivot.drop(columns=lowSalesCategories.index)
        st.write("")

        # categories that have long periods of zero sales
        st.subheader("Categories with Extended Zero Sales Periods")
        st.write("The table below shows the percentage of rows with zero sales for each category.")
        st.write((pivot == 0).astype(int).sum(axis=0) / len(pivot.sum(axis=0)))
        longZeroSalesCategories = (pivot == 0).astype(int).sum(axis=0) / len(pivot.sum(axis=0))
        longZeroSalesCategories = longZeroSalesCategories[longZeroSalesCategories > 0.5]
        st.write(f"There are {len(longZeroSalesCategories)} out of {totalNumCategories} categories with more than 50% of days having zero sales.")
        for category in longZeroSalesCategories.index:
            st.write(f"{category}: {longZeroSalesCategories[category]*100:.2f}% days with zero sales")
        st.write("These categories are being dropped due to extended periods of zero sales affecting forecasting accuracy.")
        pivot = pivot.drop(columns=longZeroSalesCategories.index)
        st.write("")

        # remove outliers using z-score
        # remove rows where any category has z-score > 2.7 or < -2.7 (99.7% confidence interval)
        from scipy import stats 
        zScores = np.abs(stats.zscore(pivot, nan_policy='omit'))
        outlierRows = np.where(zScores > 2.7)[0]
        st.subheader("Outlier Removal")
        # display number of outlier rows being removed by category
        st.write("Number of outlier rows being removed by category:")
        for i, category in enumerate(pivot.columns):
            numOutliers = np.sum(zScores[:, i] > 2.7)
            st.write(f"{category}: {numOutliers} outlier rows")
        st.write(f"Removing {len(outlierRows)} rows with outliers based on z-score method.")
        st.write("The remaining rows are 3 standard deviations from the mean with 99.7% confidence.")
        pivot = pivot.drop(pivot.index[outlierRows])
        st.write("")

    if setting == "Forecasting":
        st.title("Retail Store Inventory and Demand Forecasting")
        # Category selector
        selectedCategory = st.selectbox("Select Category", categories)
        st.write("")
        st.subheader("Sales Forecasting")
        if selectedCategory == "ALL": # if all data is selected
            allCategoryForecast = st.toggle("Show All Sales Forecasting", False)
            if allCategoryForecast == False:
                st.warning("Forecasting for all categories is innacurate, please select a category.")
            else:
                st.write("Sales Forecasting For All Categories")
                series = pivot.reset_index()
                series = series.melt(id_vars=['date'], var_name='family', value_name='sales')
                series.columns = ['ds', 'family', 'y']
                # Train Prophet model
                model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
                model.fit(series)
                # Create future dataframe for next 90 days
                selectedPeriod = st.slider("Select number of days to include in forecasting", 5, 120, 90, step=5)
                future = model.make_future_dataframe(periods=selectedPeriod)
                forecast = model.predict(future)
                # Plot forecast
                figure = model.plot(forecast)
                # change y and x labels using matplotlib
                plt.ylabel("Sales")
                plt.xlabel("Date")
                st.pyplot(figure)
        else: # if a category is slected
            allCategoryForecast = st.toggle("Show All Sales Forecasting Categories", True)
            # show categories that have not been dropped from pivot
            # list categories not dropped
            if allCategoryForecast:
                st.write("Categories Available For Forecasting:")
                for category in pivot.columns:
                    st.write(f"- {category}")
                st.write("")
            if selectedCategory in pivot.columns:
                st.write(f"Sales Forecasting For {selectedCategory}")
                series = pivot[[selectedCategory]].reset_index()
                series.columns = ['ds', 'y']
                # Train Prophet model
                model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
                model.fit(series)
                # Create future dataframe for next 90 days
                selectedPeriod = st.slider("Select number of days to include in forecasting", 5, 120, 90, step=5)
                future = model.make_future_dataframe(periods=selectedPeriod)
                forecast = model.predict(future)
                # Plot forecast
                figure = model.plot(forecast)
                st.pyplot(figure)
            else:
                st.warning("Selected category not found in the data")
        st.write("")

        if (selectedCategory == "ALL" and allCategoryForecast == True) or (selectedCategory != "ALL" and selectedCategory in pivot.columns):
            st.subheader("Analysis")
            if st.button("Run Analysis"):
                with st.spinner("Analysing forecasting sales data..."):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as chart:
                        # download plot as image to temporary file
                        figure.savefig(chart.name)
                        chartPath = chart.name

                    with open(chartPath, 'rb') as chartImage:
                        image = base64.b64encode(chartImage.read()).decode('utf-8')

                    analysisMessage = [{
                        'role': 'user',
                        'content': """You are an inventory analyst specialising in technical analysis at a chain store.
                            Analyse the inventory in the chart and provide a recommendation on the amount of inventory that should be purchased.
                            Based on your recomendation only on the sales chart provided. 
                            First give your recommendation and then give a detailed justification.
                            """,
                        'images': [image]
                    }]
                    response = ollama.chat(model='llama3.2-version', messages=analysisMessage)

                st.write(response['message']['content'])
                os.remove(chartPath)
