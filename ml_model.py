import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('UrbanEdgeApparel.csv')

# Data cleaning and preprocessing
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data.dropna(subset=['Order Date'], inplace=True)  # Drop rows where Order Date is NaN
data = data[data['Order Status'] == 'Completed']  # Consider only completed orders

# Feature Engineering
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day

# Dropping unused columns
data.drop(['Order ID', 'Order Status', 'Order Date', 'Customer ID', 'Product Variant ID',
           'Shipment ID', 'Shipment Number', 'Shipping Address Type', 'Shipping City', 'Shipping State',
           'Shipping Postal Code', 'Shipping Country', 'Payment Status'], axis=1, inplace=True)

# Handling categorical data
le = LabelEncoder()
data['Company ID'] = le.fit_transform(data['Company ID'].astype(str))  # Encode Company ID
data['Product ID'] = le.fit_transform(data['Product ID'].astype(str))  # Encode Product ID

# Scaling numerical features
scaler = StandardScaler()
data[['Product Unit Selling Price', 'Product Quantity']] = scaler.fit_transform(data[['Product Unit Selling Price', 'Product Quantity']])

# Prepare features and target variable
X = data.drop(['Total Selling Price'], axis=1)
y = data['Total Selling Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
