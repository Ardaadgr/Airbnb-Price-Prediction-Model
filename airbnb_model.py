import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Visualization settings
plt.style.use("ggplot")
sns.set_theme()

df = pd.read_csv("AB_NYC_2019.csv")

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

# Digitizing textual data with one-hot encoding
df_encoded = pd.get_dummies(df,columns=["neighbourhood_group","room_type"],drop_first=True)

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

# Visualizations

# Price Distribution: before and after log transform
df_raw =pd.read_csv("AB_NYC_2019.csv")
df_raw = df_raw[df_raw["price"] > 0]
df_raw = df_raw[df_raw["price"] < 500]

plt.figure(figsize = (14,6))

plt.subplot(1,2,1)
sns.histplot(df_raw["price"], bins =50,kde=True,color = "blue")
plt.title("Price Distribution (Before Log Transform)")

plt.subplot(1,2,2)
sns.histplot(np.log(df_raw["price"]),bins=50,kde=True,color = "green")
plt.title("Price Distribution (After Log Transform)")

plt.tight_layout()
plt.show()

# Price Distribution by Room Types
plt.figure(figsize=(8,5))
sns.boxplot(x="room_type",y="price",data = df_raw)
plt.ylim(0,500)
plt.title("Price Distribution by Room Types")
plt.show()

# Price Distribution by Neighbourhood
plt.figure(figsize= (8,5))
sns.boxplot(x="neighbourhood_group", y="price",data=df_raw)
plt.ylim(0,500)
plt.title("Price Distribution by Neighbourhood ")
plt.show()

# Correlation Matrix Between Features
plt.figure(figsize=(10,8))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True,fmt=".2f",cmap="coolwarm")
plt.title("Correlation Matrix Between Features")
plt.show()

# Real vs Prediction Price Graphics
plt.figure(figsize=(6,6))
sns.scatterplot(x=actual_prices, y=predicted_prices,alpha=0.5)
plt.plot([0,500],[0,500], color = "red",linestyle="--")
plt.xlabel("Real Prices ($)")
plt.ylabel("Prediction Price ($)")
plt.xlim(0,500)
plt.ylim(0,500)
plt.show()