from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[6]])

print("Prediction:", prediction)
print("Weight:", model.coef_)
print("Bias:", model.intercept_)