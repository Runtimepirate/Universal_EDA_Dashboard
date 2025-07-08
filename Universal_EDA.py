import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_summary(df):
    summary = {}
    summary['Shape'] = df.shape
    summary['Columns'] = df.columns.tolist()
    summary['Data Types'] = df.dtypes.astype(str).to_dict()
    summary['Missing Values'] = df.isnull().sum().to_dict()
    summary['Missing Percentage'] = (df.isnull().mean() * 100).round(2).to_dict()
    summary['Duplicates'] = df.duplicated().sum()
    numeric_summary = df.select_dtypes(include=np.number).describe().T
    summary['Numeric Summary'] = numeric_summary
    cat_summary = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        cat_summary[col] = {
            'Unique Values': df[col].nunique(),
            'Top Values': df[col].value_counts().head(5).to_dict()
        }
    summary['Categorical Summary'] = cat_summary
    return summary

def plot_missing_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap')
    st.pyplot(fig)

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

def plot_univariate(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    st.subheader('Univariate Distributions')
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)
    for col in cat_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Chart of {col}')
        st.pyplot(fig)

def plot_bivariate(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 2:
        st.subheader('Bivariate Scatter Plots')
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[j]], ax=ax)
                ax.set_title(f'Scatter Plot: {numeric_cols[i]} vs {numeric_cols[j]}')
                st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for scatter plots.")



with st.sidebar:
    st.image("https://raw.githubusercontent.com/Runtimepirate/About_me/main/Profile_pic.jpg", width=200)

    st.markdown("## **Mr. Aditya Katariya [[Resume](https://drive.google.com/file/d/1Vq9-H1dl5Kky2ugXPIbnPvJ72EEkTROY/view?usp=drive_link)]**")
    st.markdown(" *College - Noida Institute of Engineering and Technology, U.P*")

    st.markdown("----")

    st.markdown("## Contact Details:-")
    st.markdown("📫 *[Prasaritation@gmail.com](mailto:Prasaritation@gmail.com)*")
    st.markdown("💼 *[LinkedIn](https://www.linkedin.com/in/adityakatariya/)*")
    st.markdown("💻 *[GitHub](https://github.com/Runtimepirate)*")

    st.markdown("----")

    st.markdown("**AI & ML Enthusiast**")
    st.markdown(
        """
        Passionate about solving real-world problems using data science and customer analytics. Always learning and building smart, scalable AI solutions.
        """
    )

st.title('Universal EDA Dashboard')
st.markdown("""
### Overview

**Universal EDA Dashboard** — A tool that provides comprehensive exploratory data analysis on any uploaded CSV or Excel file to :-

- Instantly view dataset shape, columns, data types, and missing values  
- Visualize numeric and categorical features  
- Generate correlation heatmaps and univariate/bivariate plots  
- Designed for **data scientists**, **students**, and **analysts** to speed up initial data understanding  
""")

uploaded_file = st.file_uploader('Upload your dataset (CSV or Excel)', type=['csv', 'xlsx'])


if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success('Dataset loaded successfully!')

        st.subheader('Data Overview')
        st.write(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')
        st.write('Columns and Data Types:')
        dtypes_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes.astype(str)})
        st.dataframe(dtypes_df)

        st.subheader('Data Preview')
        st.dataframe(df.head())

        summary = generate_summary(df)

        st.subheader('Missing Values')
        missing_df = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().mean()*100).round(2)
        })
        st.dataframe(missing_df)
        plot_missing_heatmap(df)

        st.subheader('Duplicates')
        st.write(f'Number of duplicate rows: {summary["Duplicates"]}')

        st.subheader('Summary Statistics')
        st.write('Numeric Columns:')
        st.dataframe(summary['Numeric Summary'])

        st.write('Categorical Columns:')
        for col, stats in summary['Categorical Summary'].items():
            st.write(f'**{col}**')
            st.write(f'Unique Values: {stats["Unique Values"]}')
            st.write('Top Values:')
            st.write(stats['Top Values'])

        plot_univariate(df)
        plot_correlation_heatmap(df)
        plot_bivariate(df)

    except Exception as e:
        st.error(f'Error loading or processing file: {e}')
