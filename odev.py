import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("/Users/velikalkan/Desktop/yz_deneme_odevi/veri_seti.csv", sep=",")

# Drop NaN values
df = df.dropna()

# Select relevant columns
selected_columns = ['Post Hour', 'Lifetime Post Total Impressions',
                    'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                    'Lifetime Post Impressions by people who have liked your Page',
                    'Lifetime Post reach by people who like your Page',
                    'Lifetime People who have liked your Page and engaged with your post', 'Total Interactions']

df = df[selected_columns]

# Boxplot before removing outliers
df.boxplot()
plt.show()

# Remove outliers using Z-score
def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df))
    filtered_data = df[(z_scores < threshold).all(axis=1)]
    return filtered_data

# Detect and remove outliers
df = remove_outliers_zscore(df)

# Boxplot after removing outliers
df.boxplot()
plt.show()

# Independent variables (X) and dependent variable (y)
X = df.drop('Total Interactions', axis=1)
y = df['Total Interactions']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Uncomment the following lines if you want to use SGDRegressor
# model = linear_model.SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, random_state=42)
# model.fit(X_train, y_train.ravel())

# Create and train a Linear Regression model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate errors and scores
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Print errors and scores
print(f'Training Error: {train_error}, Score: {train_score}')
print(f'Testing Error: {test_error}, Score: {test_score}')

# Scatter plot for training set
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_train, y=y_train_pred, color='blue', label='Training Data')
plt.title('Actual vs Predicted Values for Training Set')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Scatter plot for testing set
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_test_pred, color='red', label='Testing Data')
plt.title('Actual vs Predicted Values for Testing Set')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Regression curve
plt.figure(figsize=(12, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Training Data')
plt.scatter(y_test, y_test_pred, color='red', label='Testing Data')
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='green', linewidth=2, label='Ideal Regression Line')
plt.title('Actual vs Predicted Values with Regression Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
