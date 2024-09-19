# Dataset Visualization and Analysis Tool

This is a Streamlit-based web application for visualizing and analyzing datasets. Users can upload a CSV file and generate various types of visualizations to gain insights into their data.

## Features

- **Upload CSV File:** Load your dataset for analysis.
- **Data Preview:** View the first few rows of the uploaded dataset.
- **Visualizations:** Generate different types of plots including:
  - Histogram
  - Scatter Plot
  - Box Plot
  - Heatmap
  - Line Plot
  - Bar Plot
  - Pair Plot
  - Pie Chart

## How to Use

1. Upload a CSV file using the sidebar.
2. Select the type of visualization you want to generate.
3. Customize the plot using the available options (e.g., select columns, adjust bins).
4. The generated plot will be displayed on the main screen.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- Matplotlib
- Seaborn

## Installation

1. Clone the repository.
2. Install the required packages: `pip install streamlit pandas matplotlib seaborn`
3. Run the app: `streamlit run dq_xi.py`
