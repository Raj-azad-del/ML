from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
# 3. Create the model
# --------------------------------

model = LogisticRegression()


# --------------------------------
# 4. Train the model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 5. Test the model
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print("\n--- Model Performance ---")

print("Actual:", y_test)

print("Predicted:", predictions)

print("Accuracy:", accuracy * 100, "%")


# --------------------------------
# 6. Take user input
# --------------------------------

print("\n--- Student Pass Predictor ---")

hours = float(input("Enter hours studied: "))


# --------------------------------
# 7. Make prediction
# --------------------------------

prediction = model.predict([[hours]])


# --------------------------------
# 8. Get probability
# --------------------------------

probability = model.predict_proba([[hours]])


# --------------------------------
# 9. Display result
# --------------------------------

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")


print("Probability of Fail:", probability[0][0] * 100, "%")
print("Probability of Pass:", probability[0][1] * 100, "%")