# Airbnb-Price-Prediction-Model
Airbnb price prediction model with kaggle dataset 
dataset url = https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

First step
Import database 

```python
# Read database with pandas 
df = pd.read_csv("AB_NYC_2019.csv")
```

Second Step
Data Preprocessing 

a- Clean Unnecessary Columns

```python
df = df.drop(["id", "name", "host_name", "last_review", "neighbourhood"], axis=1)
```

**In summary:** These columns are either uninformative (`id`), too complex due to textual content (`name`, `host_name`), or too diverse to handle easily at the beginner level (`neighbourhood`). Therefore, dropping them is a reasonable step.

b- Clean missing data

```python
df = df.dropna()
```

Model can not train with missing (null,none) data. We can fill missing data with fillna() method but this is my first project and I choose basic way.

c- Extreme Outlier Removal

```python
df = df[”price”] > 0]
df = df[”price”] < 500]
```

Some Airbnb’s price is zero and this is technically meaningless

Some prices are more than 10.000$ these prices misleads the model because these values are rare but enough to manipulate training process.

d - Apply Log Transformation to Prices

```python
df[”price”] = np.log(df[”price”])
```

The price data is highly skewed: most prices are between $100–$200, but some go well beyond $1000. This skewness can mislead the model.

When we apply `np.log()`, the differences between prices are compressed, allowing the model to generalize better.

Example prices:

`[50, 100, 300, 1000]`

After log transformation:

`[3.91, 4.60, 5.70, 6.91]`

**Advantage:** It reduces extreme gaps between price values → making learning easier for the model.

e- Digitization of Categorical Data (One-hot encoding)

```python
df_encoded = pd.get_dummies(df, columns =["neighbourhood_group", "room_type"], drop_first = True)

```

ML Models only work with numerical data.
neighbourhood_group: Manhattan,Brooklyn, etc.(text)
room_type: Entire home, Private room, vs. (text)

get_dummies() change texts like 0-1 numbers

drop_first=True removes first category and prevent multicollinearity

Third Step

Start prepare train and test data 

a- X and Y definition (Input and Target definition)

```python
y = df_encoded["price"]  
X = df_encoded.drop("price", axis=1)  
```

To provide the model to learn:

**X** → Independent variables (location, room type, number of rooms, etc.)

**y** → Dependent variable (log-transformed price)

b- Splitting Data Into Training and Test 

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
```

Split the data in two pieces
%80 Train data percentage for model (train)
%20 Test data percentage for model (test)

c- Create, Train and Test the Model

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

LinearRegression() → Most basic prediction model “Price is directli proportional to the number of rooms and location ”

fit() → Training process (**The relationship between x_train and y_train is learned**)
predict() → Apply the learning to test data

Final Step 

Turning Back the Predicted Data to Actual Prices

```python
actual_prices = np.exp(y_test)
predicted_prices = np.exp(y_pred)
for real, pred in list(zip(actual_prices, predicted_prices))[:10]:
print(f"Gerçek: {real:.2f} USD, Tahmin: {pred:.2f} USD")
```

Why we need to do this? 
Because we apply log transform to data. 
If you want to see the predicted results in actual dollar values, you need to convert them back using `np.exp()`.

Metrics:

**1. Mean Squared Error (MSE)**

 *It is the average of the squares of the differences between the predicted values and the actual values.*

---

 **Mathematical Formula:**

![MSE](https://github.com/user-attachments/assets/197c101e-ca25-471f-ba7b-365a1d2891dc)


---

 **Explanation:**

- Errors are calculated as (actual − predicted).
- Each error is squared (to eliminate negatives).
- The squared errors are averaged → this gives the *mean squared error*.

---

**Interpretation:**

- A **lower MSE** indicates a **better-performing model**.
- Its unit is the **square of the original unit** (e.g., dollars²), which makes it harder to interpret directly in terms of the target variable.

---

**Advantage:**

- **Penalizes large errors more heavily**, since errors are squared.

2.

**R² Score (Coefficient of Determination)**

 I*ndicates how much of the variance in the target variable is explained by the model.*

---

**Mathematical Formula:**

![R2_Score_Small](https://github.com/user-attachments/assets/518d2bc4-8cb4-446c-8684-b75b314c704e)



Where:

- SSres: Sum of squared residuals (model error)
- SStot: Total variance in the data (differences from the mean)

---

**Meaning:**

- **R² = 1**: Perfect prediction (no error)
- **R² = 0**: The model learned nothing (no better than predicting the mean)
- **R² < 0**: The model performs worse than just predicting the average → warning sign

---

E**xample:**

- **R² = 0.80**: The model explains 80% of the variation in prices.
- **R² = 0.20**: The model explains only 20% of the variation → weak model.

**MSE** → Answers the question: *"How wrong are the predictions?"*

**R²** → Answers the question: *"How well does the model explain the data?"*

This setup:

- Removes outliers
- Normalizes prices using log transformation
- Converts text data into numerical format
- Splits data into training and test sets
- Trains and validates the model
