import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Dataset Visualization and Analysis Tool')

# Sidebar for uploading dataset and selecting visualization
st.sidebar.title('Options')

# File upload in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=['csv'])

def remove_outliers_iqr(data, column):
    "Remove outliers using the IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

# Visualization type selection in the sidebar
visualization_type = st.sidebar.selectbox('Select Visualization Type', [
    'Histogram', 'Scatter Plot', 'Box Plot', 'Heatmap', 
    'Line Plot', 'Bar Plot', 'Pair Plot', 'Pie Chart'
])

# Check if a file is uploaded
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Show columns only if the dataset is loaded
    columns = df.columns.tolist()
    
    if visualization_type == 'Histogram':
        column = st.sidebar.selectbox('Select Column for Histogram', columns)
        bins = st.sidebar.slider('Number of Bins', min_value=5, max_value=50, value=20)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        st.pyplot(plt)
    
    elif visualization_type == 'Scatter Plot':
        x_col = st.sidebar.selectbox('Select X-axis', columns)
        y_col = st.sidebar.selectbox('Select Y-axis', columns)
        marker_size = st.sidebar.slider('Marker Size', min_value=50, max_value=300, value=100)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, s=marker_size)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        st.pyplot(plt)
    
    elif visualization_type == 'Box Plot':
        column = st.sidebar.selectbox('Select Column for Box Plot', columns)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=column)
        plt.title(f'Box Plot of {column}')
        st.pyplot(plt)
    
    elif visualization_type == 'Heatmap':
        st.write("Heatmap of correlation matrix:")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)
    
    elif visualization_type == 'Line Plot':
        x_col = st.sidebar.selectbox('Select X-axis', columns)
        y_col = st.sidebar.selectbox('Select Y-axis', columns)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=x_col, y=y_col)
        plt.title(f'Line Plot: {x_col} vs {y_col}')
        st.pyplot(plt)
    
    elif visualization_type == 'Bar Plot':
        column = st.sidebar.selectbox('Select Column for Bar Plot', columns)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column)
        plt.title(f'Bar Plot of {column}')
        st.pyplot(plt)
    
    elif visualization_type == 'Pair Plot':
        st.write("Pair Plot for numerical columns:")
        plt.figure(figsize=(10, 6))
        sns.pairplot(df.select_dtypes(include=['number']))
        st.pyplot(plt)
    
    elif visualization_type == 'Pie Chart':
        column = st.sidebar.selectbox('Select Column for Pie Chart', columns)
        pie_data = df[column].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Pie Chart of {column}')
        st.pyplot(plt)

else:
    st.write("Please upload a dataset to start visualizing.")
