# ESERCIZIO 1

# 1a
summer_X_train = summer_train[["temp"]]
summer_y_train = summer_train["demand"]
summer_X_test = summer_test[["temp"]]
summer_y_test = summer_test["demand"]

# 1b
lrm = LinearRegression()
lrm.fit(summer_X_train, summer_y_train)

# 1c
print_eval(summer_X_train, summer_y_train, lrm)

# 1d
print_eval(summer_X_test, summer_y_test, lrm)

# 1e
plot_model_on_data(summer_X_train, summer_y_train, lrm)

# 1f
plot_model_on_data(summer_X_test, summer_y_test, lrm)


# ESERCIZIO 2

# 2a
lrm = LinearRegression()
lrm.fit(X_train, y_train)

# 2b
print_eval(X_train, y_train, lrm)

# 2c
print_eval(X_test, y_test, lrm)


# ESERCIZIO 3

# 3a
X_train_d3 = np.c_[X_train, X_train ** 2, X_train ** 3]

# 3b
X_test_d3 = np.c_[X_test, X_test ** 2, X_test ** 3]

# 3c
prm = LinearRegression()
prm.fit(X_train_d3, y_train)

# 3d
print_eval(X_train_d3, y_train, prm)

print_eval(X_test_d3, y_test, prm)


# ESERCIZIO 4

# 4a
prm = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("linreg", LinearRegression())
])

# 4b
prm.fit(X_train, y_train)

# 4c
print_eval(X_train, y_train, prm)

print_eval(X_test, y_test, prm)

# 4d
plot_model_on_data(X_test, y_test, prm)


# ESERCIZIO 5

# 5a
prm = Pipeline([
    ("poly",   PolynomialFeatures(degree=15, include_bias=False)),
    ("linreg", LinearRegression())
])
prm.fit(X_train, y_train)
print_eval(X_test, y_test, prm)

# 5b
prm = Pipeline([
    ("poly",   PolynomialFeatures(degree=15, include_bias=False)),
    ("scale",  StandardScaler()),
    ("linreg", LinearRegression())
])
prm.fit(X_train, y_train)
print_eval(X_test, y_test, prm)

# 5c
prm = Pipeline([
    ("poly",   PolynomialFeatures(degree=15, include_bias=False)),
    ("scale",  MinMaxScaler()),
    ("linreg", LinearRegression())
])
prm.fit(X_train, y_train)
print_eval(X_test, y_test, prm)