from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)


# --------------------------------
# 1. Training data
# --------------------------------

# X = hours studied

X = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
]

# y = result
# 0 = Fail
# 1 = Pass

y = [
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1
]


# --------------------------------
# 2. Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 3. Create model
# --------------------------------

model = LogisticRegression()


# --------------------------------
# 4. Train model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 5. Predict test data
# --------------------------------

predictions = model.predict(X_test)


# --------------------------------
# 6. Calculate metrics
# --------------------------------

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)


# --------------------------------
# 7. Display results
# --------------------------------

print("\n--- Model Performance ---")

print("Actual:", y_test)

print("Predicted:", predictions)

print("\nAccuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(cm)

print("\nPrecision:", precision * 100, "%")

print("Recall:", recall * 100, "%")


# --------------------------------
# 8. Take user input
# --------------------------------

print("\n--- Student Pass Predictor ---")

hours = float(input("Enter hours studied: "))


# --------------------------------
# 9. Predict user's result
# --------------------------------

prediction = model.predict([[hours]])

probability = model.predict_proba([[hours]])


# --------------------------------
# 10. Display user's result
# --------------------------------

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")


print(
    "Probability of Fail:",
    probability[0][0] * 100,
    "%"
)

print(
    "Probability of Pass:",
    probability[0][1] * 100,
    "%"
)