import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("AB_NYC_2019.csv")
print(df.head())

# Information about dataset
print(df.info())

# Missing data check
print("\nMissing Data:\n", df.isnull().sum())

# Statistics about price column
print('\nPrice distribution:\n', df["price"].describe())

# Remove unnecessary columns for price prediction
df = df.drop(["id", "name","host_name","last_review","neighbourhood"], axis=1)
df = df.dropna()

# Disable outlier values
df = df[df["price"] > 0]
df = df[df["price"] < 500]

# Make log transform
df["price"] = np.log(df["price"])

print("New data size", df.shape)

# Digitizing textual data with one-hot encoding
df_encoded = pd.get_dummies(df,columns=["neighbourhood_group","room_type"],drop_first=True)
print(df_encoded.head())

# Dependent variable (price)
y = df_encoded["price"]

# Independent variable (all columns except price)
X= df_encoded.drop("price", axis = 1)

# Split data train and test (%80 train / %20 test)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

# Create Model
model = LinearRegression()

#Model Training
model.fit(X_train,y_train)

#Prediction with test data
y_pred = model.predict(X_test)

print("Mean Squared Error", mean_squared_error(y_test,y_pred))
print("R2 Score:", r2_score(y_test,y_pred))

# Transform log values to real values
actual_prices = np.exp(y_test)
predicted_prices = np.exp(y_pred)

for real,pred in list(zip(actual_prices, predicted_prices))[:10]:
    print(f"Actual Price: {real:.2f} USD, Predict: {pred: .2f} USD")

