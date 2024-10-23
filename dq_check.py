import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Streamlit app title
st.title('Dataset Visualization and Analysis Tool')

# Sidebar for uploading dataset and selecting options
st.sidebar.title('Options')

# File upload in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=['csv'])

def remove_outliers_iqr(data, column):
    """Remove outliers using the IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

# Initialize session state for DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Check if a file is uploaded
if uploaded_file:
    # Load the dataset
    st.session_state.df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(st.session_state.df.head())

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
        st.session_state.df.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)
    
    if show_shape:
        st.write("### Dataset Shape")
        st.write(st.session_state.df.shape)
    
    if show_missing:
        st.write("### Missing Values")
        missing_values = st.session_state.df.isnull().sum()
        missing_percentage = st.session_state.df.isnull().mean() * 100
        st.write(pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage}))
    
    if show_dtypes:
        st.write("### Data Types")
        st.write(st.session_state.df.dtypes)

    # Dropdown for Visualization Types with checkbox to enable visualizations
    st.sidebar.write("### Visualization Options")
    enable_visualization = st.sidebar.checkbox("Enable Data Visualization")

    if enable_visualization:
        visualization_type = st.sidebar.selectbox('Select Visualization Type', [
            'Histogram', 'Scatter Plot', 'Box Plot', 'Heatmap', 
            'Line Plot', 'Bar Plot', 'Pair Plot', 'Pie Chart'
        ])

        columns = st.session_state.df.columns.tolist()

        if visualization_type == 'Histogram':
            column = st.sidebar.selectbox('Select Column for Histogram', columns)
            bins = st.sidebar.slider('Number of Bins', min_value=5, max_value=50, value=20)
            plt.figure(figsize=(10, 6))
            sns.histplot(st.session_state.df[column], bins=bins, kde=True)
            plt.title(f'Histogram of {column}')
            st.pyplot(plt)

        elif visualization_type == 'Scatter Plot':
            x_col = st.sidebar.selectbox('Select X-axis', columns)
            y_col = st.sidebar.selectbox('Select Y-axis', columns)
            marker_size = st.sidebar.slider('Marker Size', min_value=50, max_value=300, value=100)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=st.session_state.df, x=x_col, y=y_col, s=marker_size)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            st.pyplot(plt)

        elif visualization_type == 'Box Plot':
            column = st.sidebar.selectbox('Select Column for Box Plot', columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=st.session_state.df, x=column)
            plt.title(f'Box Plot of {column}')
            st.pyplot(plt)

        elif visualization_type == 'Heatmap':
            st.write("Heatmap of correlation matrix:")
            plt.figure(figsize=(10, 6))
            sns.heatmap(st.session_state.df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        elif visualization_type == 'Line Plot':
            x_col = st.sidebar.selectbox('Select X-axis', columns)
            y_col = st.sidebar.selectbox('Select Y-axis', columns)
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=st.session_state.df, x=x_col, y=y_col)
            plt.title(f'Line Plot: {x_col} vs {y_col}')
            st.pyplot(plt)

        elif visualization_type == 'Bar Plot':
            column = st.sidebar.selectbox('Select Column for Bar Plot', columns)
            plt.figure(figsize=(10, 6))
            sns.countplot(data=st.session_state.df, x=column)
            plt.title(f'Bar Plot of {column}')
            st.pyplot(plt)

        elif visualization_type == 'Pair Plot':
            st.write("Pair Plot for numerical columns:")
            plt.figure(figsize=(10, 6))
            sns.pairplot(st.session_state.df.select_dtypes(include=['number']))
            st.pyplot(plt)

        elif visualization_type == 'Pie Chart':
            column = st.sidebar.selectbox('Select Column for Pie Chart', columns)
            pie_data = st.session_state.df[column].value_counts()
            plt.figure(figsize=(10, 6))
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            plt.title(f'Pie Chart of {column}')
            st.pyplot(plt)

    # Correlation analysis and handling outliers checkboxes
    st.sidebar.write("### Advanced Analysis")
    show_corr = st.sidebar.checkbox("Show Correlation Matrix")
    show_outliers = st.sidebar.checkbox("Handle Outliers")
    remove_outliers = st.sidebar.checkbox("Remove Outliers")
    fill_missing = st.sidebar.checkbox("Fill Missing Values")
    drop_missing = st.sidebar.checkbox("Drop Missing Values")

    # Correlation analysis
    if show_corr:
        st.write("### Correlation Matrix")
        numeric_df = st.session_state.df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.write("No numeric columns available for correlation analysis.")
        else:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

    # Handle outliers (Box Plots for numerical columns)
    if show_outliers:
        st.write("### Handling Outliers (Box Plots)")
        num_columns = st.session_state.df.select_dtypes(include=['number']).columns
        for col in num_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=st.session_state.df[col])
            plt.title(f'Box Plot of {col}')
            st.pyplot(plt)

    # Remove outliers
    if remove_outliers:
        columns = st.session_state.df.columns.tolist()
        column_to_check = st.sidebar.selectbox('Select Column for Outlier Removal', columns)

        # Perform outlier removal if a column is selected
        if column_to_check:
            new_df = remove_outliers_iqr(st.session_state.df, column_to_check)
            st.session_state.df = new_df  # Update the session state with the new DataFrame
            st.write(f"Outliers removed based on {column_to_check}. New shape: {st.session_state.df.shape}")

    # Handle missing values
    if drop_missing:
        st.session_state.df = st.session_state.df.dropna()
        st.write("Missing values dropped. New shape:", st.session_state.df.shape)

    if fill_missing:
        fill_option = st.sidebar.selectbox("Select fill method", ["Mean", "Standard Deviation"])
        if fill_option == "Mean":
            st.session_state.df = st.session_state.df.fillna(st.session_state.df.mean())
            st.write("Missing values filled with mean.")
        elif fill_option == "Standard Deviation":
            st.session_state.df = st.session_state.df.fillna(st.session_state.df.std())
            st.write("Missing values filled with standard deviation.")

    # Normalization and Standardization options
    st.sidebar.write("### Scaling Options")
    scaling_option = st.sidebar.selectbox("Select Scaling Method", ["None", "Normalization", "Standardization"])
    
    if scaling_option == "Normalization":
        num_columns = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        if num_columns:
            scaler = MinMaxScaler()
            st.session_state.df[num_columns] = scaler.fit_transform(st.session_state.df[num_columns])
            st.write("Normalization applied to numerical columns.")
        else:
            st.write("No numerical columns available for normalization.")
    
    elif scaling_option == "Standardization":
        num_columns = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        if num_columns:
            scaler = StandardScaler()
            st.session_state.df[num_columns] = scaler.fit_transform(st.session_state.df[num_columns])
            st.write("Standardization applied to numerical columns.")
    st.sidebar.write("### Encoding Options")
    encoding_option = st.sidebar.selectbox("Select Encoding Method", ["None", "Label Encoding", "One-Hot Encoding"])
    if encoding_option == "Label Encoding":
        label_cols = st.sidebar.multiselect("Select Columns for Label Encoding", st.session_state.df.select_dtypes(include=['object']).columns)
        for col in label_cols:
            le = LabelEncoder()
            st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
        st.write("Label Encoding applied to selected columns.")
    
    elif encoding_option == "One-Hot Encoding":
        st.session_state.df = pd.get_dummies(st.session_state.df)
        st.write("One-Hot Encoding applied to the dataset.")
        

else:
    st.write("Please upload a dataset to start visualizing.")

