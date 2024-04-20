# ESERCIZIO 1

# 1a
model_a = LinearRegression()
model_a.fit(X_train, y_train)
print_eval(X_test, y_test, model_a)

# 1b
model_b = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("linreg", LinearRegression())
])
model_b.fit(X_train, y_train)
print_eval(X_test, y_test, model_b)

# 1c
model_c = Pipeline([
    ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
    ("scale",  StandardScaler()),
    ("linreg", LinearRegression())
])
model_c.fit(X_train, y_train)
print_eval(X_test, y_test, model_c)


# ESERCIZIO 2

# 2a
def test_regression(degree, alpha):
    rrm = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])
    rrm.fit(X_train, y_train)
    return rrm.score(X_test, y_test)

# 2b
res_degree = np.arange(3, 31)
res_low_reg = np.array([test_regression(d, 0.01) for d in res_degree])

# 2c
res_high_reg = np.array([test_regression(d, 10) for d in res_degree])

# 2d
plt.figure(figsize=(10, 6))
plt.plot(res_degree, res_low_reg, "ro-")
plt.plot(res_degree, res_high_reg, "bo-")
plt.grid()
plt.xlabel("Grado regr. polinomiale")
plt.ylabel("Score R²")
# aggiungiamo una legenda al grafico
plt.legend(["α = 0.01", "α = 10"], loc="lower right");


# ESERCIZIO 3

# 3a
model_a = LinearRegression()
model_a.fit(X_train, y_train)
print_eval(X_test, y_test, model_a)

# 3b
model_b = Ridge(alpha=10)
model_b.fit(X_train, y_train)
print_eval(X_test, y_test, model_b)

# 3c
model_c = Pipeline([
    ("scale", StandardScaler()),
    ("lr", LinearRegression())
])
model_c.fit(X_train, y_train)
print_eval(X_test, y_test, model_c)


# ESERCIZIO 4

# 4a
def elastic_net_with_alphas(alpha_l2, alpha_l1):
    alpha = alpha_l1 + alpha_l2
    l1_ratio = alpha_l1 / alpha
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# 4b
model = Pipeline([
    ("scale", StandardScaler()),
    ("regr",  elastic_net_with_alphas(1, 0.1))
])
model.fit(X_train, y_train)
print_eval(X_test, y_test, model)


# ESERCIZIO 5

# 5a
model = poly_std_elasticnet(2)
%time model.fit(X_train, y_train)
print_eval(X_test, y_test, model)

# 5b
model = poly_std_elasticnet(5)
%time model.fit(Xsub_train, y_train)
print_eval(Xsub_test, y_test, model)

# 5c
model = poly_std_elasticnet(5)
%time model.fit(X_train, y_train)
print_eval(X_test, y_test, model)


# ESERCIZIO 6

# 6a
model = Pipeline([
    ("scale", StandardScaler()),
    ("regr",  KernelRidge(alpha=10, kernel="poly", degree=3))
])

# 6b
cv_results = cross_validate(model, X, y, cv=kfold_5)

# 6c
cv_scores = cv_results["test_score"]
cv_scores.mean(), cv_scores.std()


# ESERCIZIO 7

# 7a
model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("regr", ElasticNet())
])
grid = {
    "poly__degree": [2, 3],
    "regr__alpha": [0.1, 1, 10],
    "regr__l1_ratio": [0.1, 0.25, 0.5]
}
gs = GridSearchCV(model, grid, cv=kfold_5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print_eval(X_test, y_test, gs)

# 7b
model = Pipeline([
    ("scale", StandardScaler()),
    ("regr", KernelRidge(kernel="poly"))
])
grid = {
    "regr__degree": range(2, 7),
    "regr__alpha": [0.01, 0.1, 1, 10],
}
gs = GridSearchCV(model, grid, cv=kfold_5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print_eval(X_test, y_test, gs)


# ESERCIZIO 8

# 8a
def nested_cv(model, grid):
    results = []
    for train_indices, val_indices in outer_cv.split(X, y):
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
        gs = GridSearchCV(model, grid, cv=inner_cv)
        gs.fit(X_train, y_train)
        score = gs.score(X_val, y_val)
        results.append(score)
    return results

# 8b
model = Pipeline([
    ("scale", StandardScaler()),
    ("regr", KernelRidge(kernel="poly"))
])
grid = {
    "regr__degree": range(2, 7),
    "regr__alpha": [0.01, 0.1, 1, 10],
}
nested_cv(model, grid)