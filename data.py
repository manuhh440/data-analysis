

# Load the Iris dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Explore the structure: data types and missing values
print("\nInformation about the dataset:")
print(iris_df.info())

print("\nNumber of missing values per column:")
print(iris_df.isnull().sum())

# Clean the dataset (in this case, there are no missing values, but if there were...)
# For demonstration, let's imagine we wanted to fill missing numerical values with the mean
# and drop rows with missing categorical values (though not needed here)
# for col in iris_df.columns:
#     if iris_df[col].dtype in ['int64', 'float64']:
#         iris_df[col].fillna(iris_df[col].mean(), inplace=True)
#     elif iris_df[col].dtype == 'object':
#         iris_df.dropna(subset=[col], inplace=True)

print("\nDataset after handling missing values (no changes in this case):")
print(iris_df.head())
import matplotlib.pyplot as plt

# 1. Line chart (not directly applicable to this dataset in a typical time-series sense)
# We can create a line plot of the index vs. one of the features to see the distribution order
plt.figure(figsize=(10, 6))
plt.plot(iris_df.index, iris_df['sepal_length'], marker='o', linestyle='-')
plt.title('Sepal Length Distribution')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.show()

# 2. Bar chart: Average petal length per species
average_petal_length = iris_df.groupby('species')['petal_length'].mean()
plt.figure(figsize=(8, 6))
average_petal_length.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(8, 6))
plt.hist(iris_df['sepal_width'], bins=10, color='lightsalmon', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: Sepal length vs. petal length
plt.figure(figsize=(8, 6))
plt.scatter(iris_df['sepal_length'], iris_df['petal_length'], color='gold', alpha=0.7)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.show()
