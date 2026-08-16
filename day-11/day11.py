from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = [
    [500],
    [600],
    [700],
    [800],
    [900],
    [1000],
    [1100],
    [1200],
    [1300],
    [1400]
]

y = [
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Training data:", X_train)
print("Test data:", X_test)

print("Actual prices:", y_test)
print("Predicted prices:", predictions)

print("MAE:", mae)

print("Weight:", model.coef_)
print("Bias:", model.intercept_)