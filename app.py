import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import io
from google.colab import files
import plotly.express as px

# To Upload the file
#uploaded_file = files.upload()

# Get the filename
#filename = next(iter(uploaded_file))

# Reading the File via Pandas
data = pd.read_csv("Daily Household Transactions.csv")

# Task 1

# Data Cleaning and Preprocessing
# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Fill missing values in 'Subcategory' with a placeholder
data['Subcategory'] = data['Subcategory'].fillna('Unknown')

# Fill missing values in 'Note' with an empty string
data['Note'] = data['Note'].fillna('')
data['Amount'] = data['Amount'].astype(float)

# Handle outliers in the 'Amount' column
Q1 = data['Amount'].quantile(0.25)
Q3 = data['Amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['Amount'] < (Q1 - 1.5 * IQR)) | (data['Amount'] > (Q3 + 1.5 * IQR))]
data = data[~data.index.isin(outliers.index)]

# Create new features from the Date column
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Task 2

# Exploratory Data Analysis (EDA)
# Setting the style via Seaborn
sns.set(style="whitegrid")

# Distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount (INR)')
plt.ylabel('Frequency')
plt.show()

# Distribution of transactions over time
plt.figure(figsize=(10, 6))
data['Date'].dt.to_period('M').value_counts().sort_index().plot(kind='bar')
plt.title('Number of Transactions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.show()

# Breakdown of transactions by Category
plt.figure(figsize=(10, 6))
data['Category'].value_counts().plot(kind='bar')
plt.title('Number of Transactions by Category')
plt.xlabel('Category')
plt.ylabel('Number of Transactions')
plt.show()

# Breakdown of transactions by Subcategory
plt.figure(figsize=(10, 6))
data['Subcategory'].value_counts().nlargest(20).plot(kind='bar')
plt.title('Top 20 Subcategories by Number of Transactions')
plt.xlabel('Subcategory')
plt.ylabel('Number of Transactions')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Task 3
# Clustering Algorithms (e.g., K-means)
agg_data = data.pivot_table(index='Mode', columns='Category', values='Amount', aggfunc='sum', fill_value=0)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(agg_data)

# Elbow Method to determine the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(scaled_data)

# Add cluster labels to the original data
agg_data['Cluster'] = kmeans.labels_

# Task 4
# Visualization of Customer Segments

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
agg_data['PCA1'] = pca_data[:, 0]
agg_data['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=agg_data, palette='viridis')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Interactive scatter plot for PCA components
fig = px.scatter(agg_data, x='PCA1', y='PCA2', color='Cluster', title='Customer Segmentation using K-Means Clustering')
fig.show()