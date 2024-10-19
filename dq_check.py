import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

columns = []

# Streamlit app title
st.title('Dataset Visualization and Analysis Tool')

# Sidebar for uploading dataset and selecting options
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
# Check if a file is uploaded
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())


    
    # EDA Checkboxes
    st.sidebar.write("### EDA Options")
    show_info = st.sidebar.checkbox("Show Dataset Info")
    show_shape = st.sidebar.checkbox("Show Shape of the dataset")
    show_missing = st.sidebar.checkbox("Show Missing Values")
    show_dtypes = st.sidebar.checkbox("Show Data Types")


    # Display EDA results based on checkboxes
    if show_info:
        st.write("### Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)
    
    if show_shape:
        st.write("### Dataset Shape")
        st.write(df.shape)
    
    if show_missing:
        st.write("### Missing Values")
        missing_values = df.isnull().sum()
        missing_percentage = df.isnull().mean() * 100
        st.write(pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage}))
    
    if show_dtypes:
        st.write("### Data Types")
        st.write(df.dtypes)

    # Dropdown for Visualization Types with checkbox to enable visualizations
    st.sidebar.write("### Visualization Options")
    enable_visualization = st.sidebar.checkbox("Enable Data Visualization")

    if enable_visualization:
        visualization_type = st.sidebar.selectbox('Select Visualization Type', [
            'Histogram', 'Scatter Plot', 'Box Plot', 'Heatmap', 
            'Line Plot', 'Bar Plot', 'Pair Plot', 'Pie Chart'
        ])

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

    # Correlation analysis and handling outliers checkboxes
    st.sidebar.write("### Advanced Analysis")
    show_corr = st.sidebar.checkbox("Show Correlation Matrix")
    show_outliers = st.sidebar.checkbox("Handle Outliers")
    remove_outliers = st.sidebar.checkbox("Remove Outliers")
    # Correlation analysis
    if show_corr:
        st.write("### Correlation Matrix")
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.write("No numeric columns available for correlation analysis.")
        else:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

    # Handle outliers (Box Plots for numerical columns)
    if show_outliers:
        st.write("### Handling Outliers (Box Plots)")
        num_columns = df.select_dtypes(include=['number']).columns
        for col in num_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col])
            plt.title(f'Box Plot of {col}')
            st.pyplot(plt)
    if remove_outliers:
        columns_to_check = []
        columns = df.columns.tolist()
        column_to_check = st.sidebar.selectbox('Select Column for Outlier Removal', columns)
        if columns_to_check:
            df = remove_outliers_iqr(df, column_to_check) 
        st.write(f"Outliers removed based on {column_to_check}. New shape: {df.shape}")

else:
    st.write("Please upload a dataset to start visualizing.")
