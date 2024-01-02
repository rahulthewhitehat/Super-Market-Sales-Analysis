
# Importing Google Drive to attach the dataset!
from google.colab import drive
drive.mount('/content/drive')

# Attaching the Dataset!
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/superstore_data.csv', encoding='windows-1252') # replace with your data csv file

# To check, if its working or not, displaying first five data!
print(df.head())

# Identify Discrete/Continuos for each attribute! (1)
df_cols = ['Customer ID', 'Customer Name', 'City', 'State',  'Country', 'Postal Code', 'Category', 'Quantity', 'Sales', 'Discount', 'Profit']
for column in df_cols:
    unique_values = df[column].unique()

    # Determine if the data is discrete or continuous based on the number of unique values
    data_type = 'Discrete' if len(unique_values) < len(df) / 10 else 'Continuous'

    # Print the column name and its nature
    print(f'{column}: {data_type}')

# Identifying Target Variables - Chosen Two for our analysis (2)

# Display basic statistics for Sales and Profit columns
print("Sales Statistics:")
print(df['Sales'].describe())

print("\nProfit Statistics:")
print(df['Profit'].describe())

# Description Statistics

import seaborn as sns
import matplotlib.pyplot as plt

# Summary statistics for numerical columns
numeric_columns = ['Sales', 'Profit']

# Distribution of numerical columns
for column in numeric_columns:
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Discrete Data - Classification -- Using Random Forest Classifier (3)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# List of discrete columns
discrete_columns = [
    'Customer ID', 'Customer Name', 'City', 'State',  'Country', 'Postal Code', 'Category', 'Quantity']

# Loop through each discrete column for classification
for discrete_column in discrete_columns:
    # Split the data into features (X) and the target variable (y)
    X = df.drop(discrete_column, axis=1)
    y = df[discrete_column]

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nColumn: {discrete_column}")
    print(f"Accuracy: {accuracy:.2f}")

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Continous Data - Regression - Perform Linear Regression on Continous Data (4)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# List of continuous columns
continuous_columns = ['Sales', 'Profit']

# Loop through each continuous column for regression
for continuous_column in continuous_columns:
    # Split the data into features (X) and the target variable (y)
    X = df.drop(continuous_column, axis=1)
    y = df[continuous_column]

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Linear Regression model
    regressor = LinearRegression()

    # Train the model
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nColumn: {continuous_column}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

# Principal Component Analysis - To reduce the dimension of the data, by finding the cumulative variance (80%) - (5)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select the features for PCA (excluding non-numeric columns)
numeric_columns =  df.select_dtypes(include=[np.number]).columns
X_pca = df[numeric_columns]

# Standardize the data
X_pca_standardized = StandardScaler().fit_transform(X_pca)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(X_pca_standardized)

# Display the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print("Cummulative Variance: ")
print(cumulative_explained_variance*100) # to make it look like a percentage

# Determine the number of components to retain for 80% variance
n_components = np.argmax(cumulative_explained_variance >= 0.8) + 1
print("No of Components to be retained: ", n_components, "\n")

# Apply PCA with the selected number of components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_pca_standardized)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# Display the DataFrame with PCA components
print("Principal Components: \n")
print(pca_df)

# Display the DataFrame with PCA components
print("Principal Components: \n")
print(pca_df)

# Check if 'PC1' is present in the DataFrame
if 'PC1' in pca_df.columns:
    # Scatter plot of the first five principal components
    plt.figure(figsize=(10, 6))
    for i in range(1, 6):
        plt.scatter(range(len(df)), pca_df[f'PC{i}'], label=f'PC{i}')

    plt.xlabel('Observation Index')
    plt.ylabel('Principal Component Value')
    plt.title('Scatter Plot of the First Five Principal Components')
    plt.legend()
    plt.show()
else:
    print("'PC1' not found in DataFrame.")

# K Means Clustering using the retained principal components - (6 - 1)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming you've retained the first five principal components
X_cluster = pca_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (k)
k = 2

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
pca_df['Cluster_KMeans'] = kmeans.fit_predict(X_cluster)

# Display the DataFrame with the KMeans cluster assignments
print(pca_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'Cluster_KMeans']])

# Hierarchical Clustering using the retained principal components (Single, Complete and Average Linkages) - (6 - 1)

from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

# Plot dendrograms for each linkage method
for method in linkage_methods:
    # Apply hierarchical clustering
    linked = linkage(X_cluster, method)

    # Plot the hierarchical clustering dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title(f'Hierarchical Clustering Dendrogram ({method}.capitalize() linkage)')
    plt.show()

# Finding CoVariance Matrix, Eigen Values and Vectors for the numneric columns in the dataset - (7)

import numpy as np

# Select numeric columns for eigen decomposition
numeric_columns = df.select_dtypes(include=[np.number])

# Standardize the numeric columns
numeric_columns_standardized = StandardScaler().fit_transform(numeric_columns)

# Calculate the covariance matrix
covariance_matrix = np.cov(numeric_columns_standardized, rowvar=False)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Display the results
print("Covariance Matrix: ")
print(covariance_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

import matplotlib.pyplot as plt
import seaborn as sns
# Visualize eigenvalues
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues)
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues')
plt.show()

# Visualize eigenvectors (heatmap)
sns.heatmap(eigenvectors, cmap='coolwarm')
plt.title('Eigenvectors')
plt.show()

# Finding Estimated Factor Loading Matrix, Communalties, and Specific Variances

from sklearn.decomposition import FactorAnalysis

# Assuming numeric_columns_standardized contains the standardized numeric columns
factor_analysis = FactorAnalysis(n_components=5)  # Choose the number of latent factors
factor_analysis.fit(numeric_columns_standardized)

# Estimated Factor Loading Matrix
factor_loading_matrix = factor_analysis.components_.T

# Communalities
communalities = 1 - factor_analysis.noise_variance_

# Specific Variances
specific_variances = factor_analysis.noise_variance_

# Display the results
print("Estimated Factor Loading Matrix:")
print(factor_loading_matrix)
print("\nCommunalities:")
print(communalities)
print("\nSpecific Variances:")
print(specific_variances)

# Split the dataset into features (X) and target variable (y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = numeric_columns_standardized
y = df['Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predictions
y_pred = lda.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Anderson Classification
anderson_classification = lda.decision_function(X_test)

# Fisher's Method
fisher_method = lda.transform(X_test)

# Maximum Distance Ratio
max_distance_ratio = lda.decision_function(X_test)


# Display the results
print("\nAnderson Classification:")
print(anderson_classification)
print("\nFisher's Method:")
print(fisher_method)
print("\nMaximum Distance Ratio:")
print(max_distance_ratio)

# Classification Results

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming lda is your trained LinearDiscriminantAnalysis model
y_pred = lda.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=lda.classes_, yticklabels=lda.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Market and Sales Analysis for the given dataset

# Sales and profit trends over time
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)

plt.plot(df['Sales'], label='Sales')
plt.plot(df['Profit'], label='Profit')
plt.xlabel('Order Date')
plt.ylabel('Amount')
plt.title('Sales and Profit Trends over Time')
plt.legend()
plt.show()

# Distribution of sales and profit by category or sub-category
sns.barplot(x='Category', y='Sales', data=df)
plt.title('Sales Distribution by Category')
plt.show()

sns.barplot(x='Sub-Category', y='Profit', data=df)
plt.title('Profit Distribution by Sub-Category')
plt.show()

# Market and Sales Analysis for the given dataset

# Sales and profit trends over time
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df.set_index('Ship Date', inplace=True)

plt.plot(df['Sales'], label='Sales')
plt.plot(df['Profit'], label='Profit')
plt.xlabel('Order Date')
plt.ylabel('Amount')
plt.title('Sales and Profit Trends over Time')
plt.legend()
plt.show()

# Distribution of sales and profit by category or sub-category
sns.barplot(x='Category', y='Sales', data=df)
plt.title('Sales Distribution by Category')
plt.show()

sns.barplot(x='Sub-Category', y='Profit', data=df)
plt.title('Profit Distribution by Sub-Category')
plt.show()