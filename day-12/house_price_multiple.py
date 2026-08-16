from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# -----------------------------
# 1. Training data
# -----------------------------

# Features:
# [house size, bedrooms, age]

X = [
    [500, 1, 20],
    [600, 2, 15],
    [700, 2, 10],
    [800, 3, 12],
    [900, 3, 8],
    [1000, 3, 5],
    [1100, 4, 10],
    [1200, 4, 5],
    [1300, 4, 3],
    [1400, 5, 2]
]

# Target: price in lakhs

y = [
    80,
    105,
    130,
    150,
    175,
    200,
    220,
    245,
    270,
    300
]


# -----------------------------
# 2. Split the data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 3. Create the model
# -----------------------------

model = LinearRegression()


# -----------------------------
# 4. Train the model
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 5. Test the model
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)


# -----------------------------
# 6. Display model performance
# -----------------------------

print("\n--- Model Performance ---")

print("Actual prices:", y_test)

print("Predicted prices:", predictions)

print("MAE:", mae)

print("Weights:", model.coef_)

print("Bias:", model.intercept_)


# -----------------------------
# 7. Take input from user
# -----------------------------

print("\n--- House Price Predictor ---")

size = float(input("Enter house size (sq ft): "))

bedrooms = float(input("Enter number of bedrooms: "))

age = float(input("Enter house age (years): "))


# -----------------------------
# 8. Make prediction
# -----------------------------

prediction = model.predict([
    [size, bedrooms, age]
])


# -----------------------------
# 9. Display prediction
# -----------------------------

print("\nPredicted house price:", prediction[0], "lakhs")