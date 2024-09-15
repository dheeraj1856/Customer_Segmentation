# Customer Segmentation using KMeans and PCA

This project implements a customer segmentation analysis using **KMeans clustering** and **Principal Component Analysis (PCA)**. The dataset contains customer data, and the goal is to group customers into clusters based on their purchase and payment behavior.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Performance Evaluation](#performance-evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#Conclusion)


## Overview
This project involves applying unsupervised machine learning techniques to cluster customers into segments based on their financial behavior. We use **KMeans Clustering** to create the segments and **Principal Component Analysis (PCA)** to reduce dimensionality and help visualize the clusters.

## Dataset
The dataset consists of the following columns:

- **Cust_Id**: Customer identification (removed during preprocessing)
- **Balance**: The balance maintained by the customer
- **Purchases**: Total amount of purchases made by the customer
- **One_Off_Purchases**: Purchases made in one transaction
- **Installments_Purchases**: Purchases made in installments
- **Cash_Advance**: Total cash advance taken by the customer
- **Purchases_Frequency**: Frequency of purchases made
- **One_Off_Purchases_Frequency**: Frequency of one-off purchases
- **Purchases_Installments_Frequency**: Frequency of installment-based purchases
- **Cash_Advance_Frequency**: Frequency of cash advances
- **Purchases_Trx**: Number of purchase transactions
- **Credit_Limit**: Credit limit assigned to the customer
- **Payments**: Total amount of payments made by the customer
- **Minimum_Payments**: Minimum payments made by the customer
- **Prc_Full_Payment**: Percentage of full payments made by the customer
- **Tenure**: Customer tenure in months

There are 8,950 rows and 18 columns in the dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (KMeans, PCA, Metrics)
- Plotly (for 3D visualizations)

## Methodology
1. **Data Preprocessing**:
    - Handled missing values for columns like `Minimum_Payments` and `Credit_Limit`.
    - Dropped the `Cust_Id` column as it's not needed for clustering.
  
2. **Exploratory Data Analysis (EDA)**:
    - Descriptive statistics and visualizations (heatmaps, distributions) were created to understand the dataset.

3. **Feature Scaling**:
    - Standardized the data using `StandardScaler` to ensure all features are on a similar scale.

4. **KMeans Clustering**:
    - Applied KMeans to group customers into clusters.

5. **PCA**:
    - Used Principal Component Analysis to reduce dimensionality for visualization and interpretation.

6. **Performance Evaluation**:
    - Evaluated the clustering using:
        - **Silhouette Score**
        - **Calinski-Harabasz Score**
        - **Davies-Bouldin Score**

## Performance Evaluation
After applying KMeans clustering, the performance metrics for the model are as follows:

- **Silhouette Score**: `0.2504` - This measures how similar an object is to its own cluster compared to other clusters.
- **Calinski-Harabasz Score**: `1605.19` - This ratio measures the dispersion of clusters.
- **Davies-Bouldin Score**: `1.5961` - Lower values indicate better-defined clusters.

## Visualizations
Several visualizations were created to help understand the dataset and the clustering results:

- Correlation Heatmap: Visualizes the relationships between variables.
- Distribution Plots: Shows the distribution of various features in the dataset.
- Cluster Visualization: Uses scatter plots to show the clusters formed by KMeans in both 2D and 3D using PCA.
- Elbow Method: Helps determine the optimal number of clusters by plotting inertia values.

## Conclusion

The KMeans clustering and PCA model successfully segmented customers based on financial behavior. Key performance metrics include:

- **Silhouette Score**: `0.2504` – Moderate cluster separation.
- **Calinski-Harabasz Score**: `1605.19` – High cluster compactness and separation.
- **Davies-Bouldin Score**: `1.5961` – Reasonable cluster definition.

PCA reduced 18 features to 8 principal components, retaining significant data variance. The clusters provide actionable insights for personalized marketing and customer management strategies. Further improvements can be made by exploring different clustering techniques or optimizing parameters.


