# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

# Reading data from the CSV file
data = pd.read_csv("Customer_Data.csv")
df = pd.DataFrame(data)

# Converting first characters of each word to uppercase and remaining to lowercase for column names
df.columns = [i.title() for i in df.columns]

# Basic Exploratory Data Analysis (EDA)
print("Shape of data: {}".format(df.shape))
print("Number of rows: {}".format(df.shape[0]))
print("Number of columns: {}".format(df.shape[1]))

# Displaying information about the dataset
#df.info()

# Display summary statistics
print(df.describe(include="all").T)

# Checking for duplicate values
print("Duplicate values in df are:", df.duplicated().sum())

# Checking for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Data Cleaning
df1 = df.copy()

# Handling missing values in 'Minimum_Payments' column
df1["Minimum_Payments"] = df1["Minimum_Payments"].fillna(0)
df1.dropna(inplace=True)

# Dropping 'Cust_Id' as it's not needed for clustering
df1.drop(['Cust_Id'], axis=1, inplace=True)

# Confirm no missing values remain
print(df1.isnull().sum())

# Visualizing Tenure column distribution
plt.figure(figsize=(5, 4))
sns.countplot(df1['Tenure'])
plt.title("Tenure Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=[20, 15], dpi=100)
plt.title("Correlation Graph", fontsize=16)
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', cbar=False)
plt.show()

# Distribution plots for each column
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        sns.distplot(df1[df1.columns[4 * i + j]], color="LimeGreen", ax=axs[i, j])
plt.show()

# Scatter plot of 'Installments_Purchases' vs 'Purchases'
plt.scatter(df1["Installments_Purchases"], df1["Purchases"], color="LimeGreen", s=50, alpha=0.5)
plt.title("Installments_Purchases VS Purchases", weight="bold")
plt.xlabel("Installments_Purchases")
plt.ylabel("Purchases")
plt.grid(True)
plt.show()

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df1)

# Finding the number of clusters using the elbow method
kmeans_set = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
inertia_list = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    kmeans.fit(scaled_features)
    inertia_list.append(kmeans.inertia_)

# Elbow method plot
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), inertia_list, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.axvline(x=3, color="red", linestyle="--")
plt.show()

# Silhouette scores for different cluster numbers
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

# Silhouette score plot
plt.plot(range(2, 11), silhouette_coefficients)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Coefficients for Different Clusters")
plt.show()

# Applying KMeans Clustering with 3 clusters
kmeans = KMeans(n_clusters=3, **kmeans_set)
df1["Labels"] = kmeans.fit_predict(scaled_features)

# Visualizing clusters based on 'Installments_Purchases' and 'Purchases'
plt.scatter(df1["Installments_Purchases"], df1["Purchases"], c=df1["Labels"].astype(float), s=50, alpha=0.5)
plt.title("Installments_Purchases VS Purchases with Cluster Labels", weight="bold")
plt.xlabel("Installments_Purchases")
plt.ylabel("Purchases")
plt.grid(True)
plt.show()

# KMeans Clustering performance evaluation
print("Silhouette score:", silhouette_score(scaled_features, df1["Labels"]))
print("Calinski-Harabasz score:", metrics.calinski_harabasz_score(scaled_features, df1["Labels"]))
print("Davies-Bouldin score:", metrics.davies_bouldin_score(scaled_features, df1["Labels"]))

# PCA for dimensionality reduction
pca = PCA(n_components=8)
pca_transformed = pca.fit_transform(scaled_features)

# Visualizing explained variance ratio for PCA components
plt.bar(range(8), (pca.explained_variance_ratio_) * 100, tick_label=["PC" + str(x) for x in range(1, 9)])
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Explained Variance Ratio')
plt.title("Explained Variance by Principal Components")
plt.show()

# PCA with 3D Visualization for the top 3 components
pca_3 = PCA(n_components=3).fit_transform(scaled_features)
pca_3_df = pd.DataFrame(pca_3, columns=["PC1", "PC2", "PC3"])
pca_3_df["Labels"] = df1["Labels"]

# 3D scatter plot for PCA components
fig = px.scatter_3d(pca_3_df, x='PC1', y='PC2', z='PC3', color=pca_3_df['Labels'].astype(str))
fig.show()

# PCA with 2D Visualization for the top 2 components
pca_2 = PCA(n_components=2).fit_transform(scaled_features)
plt.scatter(pca_2[:, 0], pca_2[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.title("PCA with 2 Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Cumulative variance explained by the number of PCA components
plt.plot(np.cumsum(pca.explained_variance_ratio_), color="blue")
plt.vlines(x=8, ymax=1, ymin=0, colors="red", linestyles="--")
plt.hlines(y=0.9, xmax=17, xmin=0, colors="green", linestyles=":")
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title("Cumulative Explained Variance by PCA Components")
plt.grid(True)
plt.show()

# PCA with 8 components (using cumulative variance threshold)
pca_8 = PCA(n_components=8).fit_transform(scaled_features)
pca_8_df = pd.DataFrame(pca_8, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"])
pca_8_df["Cluster"] = df1["Labels"]
print(pca_8_df.head())

# Pair plot to visualize the relationships between the first 8 principal components
sns.pairplot(pca_8_df, hue="Cluster", diag_kind="kde", palette="Set2", corner=True)
plt.suptitle("Pairwise Scatter Plot of the First 8 Principal Components", y=1.02)
plt.show()