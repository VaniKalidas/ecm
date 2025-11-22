import streamlit as st
from datetime import date, datetime
import pandas as pd
from prophet import Prophet 
from prophet.plot import plot_plotly 
from plotly import graph_objs as go
import numpy as np
import ollama
import tempfile
import base64
import os

# Load data from dataset
def loadData(dataset):
    df = pd.read_csv(f'datasets/{dataset}')
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

# Plot data
def plotData(df):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=df['date'], y=df['family']))
    figure.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

# Clean data in dataset
def getMissingData(df):
    totalNullSum = df.isnull().sum()
    totalNullCount = df.isnull().count()
    nullPercent = totalNullSum / totalNullCount * 100
    nullTable = pd.concat([totalNullSum, nullPercent], axis = 1, keys = ['Total', 'Percent'])
    return pd.Dataframe(nullTable)

# Train data using train dataset


# Test data using test dataset
def calculateMAPE(actualVal, predictVal):
    actualVal = np.array(actualVal)
    predictVal = np.array(predictVal)
    MAPE = np.mean(np.abs((actualVal - predictVal) / actualVal)) * 100
    return MAPE

if __name__ == "__main__":
    st.set_page_config(layout="wide") # use full width of page
    st.sidebar.header("Settings")
    setting = st.sidebar.selectbox("Setting", ("Raw Data", "Data Insights", "Forecasting"))
    df = loadData("train.csv")
    
    categories = np.insert(df['family'].unique(), 0, "ALL")
    # Aggregate sales per date and family, then pivot to wide format
    salesByDate = df.groupby(['date', 'family'])['sales'].sum().reset_index()
    pivot = salesByDate.pivot(index='date', columns='family', values='sales').fillna(0)

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


    if setting == "Forecasting":
        st.title("Retail Store Inventory and Demand Forecasting")


        '''st.subheader("Analysis")
        if st.button("Run Analysis"):
            with st.spinner("Analysing forecasting sales data..."):
                with chart.NamedTemporaryFile(suffix=".png", delete=False)as chart:
                    figure.write_image(chart.name)
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
            os.remove(chartPath)'''
