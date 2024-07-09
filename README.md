# Shopping Customer Segmentation

### Unsupervised Machine Learning Project

---

**Problem Statement:** Understand the target customers for the marketing team to plan a strategy.

**Context:**

- Identify the most important shopping groups based on income, age, and the mall shopping score.
- The number of groups with a label for each.

**Objective Market Segmentation:**

Divide a target market into approachable groups. Create subsets of a market based on demographic and behavioral criteria to better understand the target for marketing activities.

---

## The Approach

1. Perform EDA
2. Use KMEANS Clustering Algorithm to create our segments
3. Use Summary Statistic on the clusters
4. Visualization

## Jupyter Notebook Details

### Import Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
```

### Load Dataset

```python
# Load the dataset from a CSV file
df = pd.read_csv("Mall_Customers.csv")
df.head()
```

### Univariate Analysis

Generate descriptive statistics and visualize distributions of Age, Annual Income, and Spending Score.

```python
df.describe()
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])
```

### Bivariate Analysis

Visualize distributions and relationships between features by gender using KDE plots and box plots.

```python
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for column in columns:
    plt.figure()
    sns.kdeplot(data=df, x=column, hue='Gender', fill=True)
    plt.title(f'Distribution of {column} by Gender')
    plt.show()

for column in columns:
    plt.figure()
    sns.boxplot(data=df, y=df[column], x='Gender')
    plt.title(f'Distribution of {column} by Gender')
    plt.show()
```

### Correlation Analysis

Calculate and visualize the correlation matrix.

```python
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### Clustering

#### Univariate Clustering

Perform KMeans clustering on Annual Income.

```python
Clustering1 = KMeans(n_clusters=3)
Clustering1.fit(df[['Annual Income (k$)']])
df['Income Cluster'] = Clustering1.labels_
df.head()
df[['Income Cluster']].value_counts()
df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

#### Bivariate Clustering

Perform KMeans clustering on Annual Income and Spending Score.

```python
Clustering2 = KMeans(n_clusters=5)
Clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = Clustering2.labels_
df.head()
Centers = pd.DataFrame(Clustering2.cluster_centers_)
Centers.columns = ['x', 'y']
plt.figure(figsize=(10, 8))
plt.scatter(x=Centers['x'], y=Centers['y'], s=100, c='black', marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Spending and Income Cluster', palette='tab10')
```

#### Multivariate Clustering

Perform KMeans clustering on scaled features including Age, Annual Income, Spending Score, and Gender.

```python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
dff = pd.get_dummies(df, drop_first=True).astype(int)
dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff = pd.DataFrame(scale.fit_transform(dff))

inertia_score3 = []
for i in range(1, 11):
    kmeans3 = KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_score3.append(kmeans3.inertia_)
plt.plot(range(1, 11), inertia_score3)
```

### Cluster Analysis

Analyze clusters to identify target groups for marketing strategies.

```python
pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize='index')
df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

**Target Cluster:**

- Target group would be cluster 1 which has a high spending score and high income.
- 60% of cluster 1 shoppers are women. Marketing campaigns should focus on popular items in this cluster.
- Cluster 4 presents an interesting opportunity for sales events targeting the customers.

---

## Usage

1. **Exploratory Data Analysis (EDA)**: Understand the dataset and visualize distributions.
2. **Clustering**: Apply KMeans clustering algorithm to segment customers.
3. **Visualization**: Plot the clusters to interpret the segments.
4. **Analysis**: Analyze the clusters to derive insights for marketing strategies.

---

## Requirements

- pandas
- seaborn
- matplotlib
- scikit-learn

```sh
pip install pandas seaborn matplotlib scikit-learn
```

---

## Authors

- Mohammadreza Delvari

## License

This project is licensed under the MIT License.
