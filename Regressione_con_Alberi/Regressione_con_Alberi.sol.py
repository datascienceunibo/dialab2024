# ESERCIZIO 1

# 1a
data_open = data_sales.loc[data_sales["Open"] == 1].drop(columns=["Open"])

# 1b
data_stores = pd.read_csv("rossmann-stores.csv")

# 1c
data = pd.merge(data_open, data_stores, left_on=["Store"], right_on=["Store"])
# abbreviabile in:
# data_merged = pd.merge(data_open, data_stores, on=["Store"])


# ESERCIZIO 2

# 2a
data["CompetitionOpen"] = (
    data["CompetitionOpenSinceYear"].isna()
    | (data["Date"].dt.year > data["CompetitionOpenSinceYear"])
    | (
        (data["Date"].dt.year == data["CompetitionOpenSinceYear"])
        & (data["Date"].dt.month >= data["CompetitionOpenSinceMonth"])
    )
)


# ESERCIZIO 3

# 3a
X_train_num = data_train[numeric_vars + binary_vars]
X_val_num = data_val[numeric_vars + binary_vars]

# 3b
model = Ridge()
model.fit(X_train_num, y_train)
model.score(X_val_num, y_val)

# 3c
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", Ridge())
])
model.fit(X_train_num, y_train)
model.score(X_val_num, y_val)


# ESERCIZIO 4

model = Pipeline([
    ("encoder", OneHotEncoder()),
    ("regr",    Ridge())
])
model.fit(X_train_cat, y_train)
model.score(X_val_cat, y_val)


# ESERCIZIO 5

# 5a
model = Pipeline([
    ("preproc", ColumnTransformer([
        ("numeric", PolynomialFeatures(include_bias=False), numeric_vars + binary_vars),
        ("categorical", OneHotEncoder(), categorical_vars)
    ])),
    ("regr" , Ridge())
])
grid = {
    "preproc__numeric__degree": [1, 2, 3],
    "regr__alpha": [0.01, 1]
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(data_train_sample, y_train_sample)
gs.best_params_

gs.score(data_val, y_val)

# 3b
model = Pipeline([
    ("preproc", ColumnTransformer([
        ("numeric", Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False))
        ]), numeric_vars + binary_vars),
        ("categorical", OneHotEncoder(), categorical_vars)
    ])),
    ("regr" , Ridge())
])
grid = {
    "preproc__numeric__poly__degree": [1, 2, 3],
    "regr__alpha": [0.01, 1]
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(data_train_sample, y_train_sample)
gs.best_params_

gs.score(data_val, y_val)


# ESERCIZIO 6

# 6a
model = DecisionTreeRegressor(random_state=42)
grid = {
    "max_depth": range(4, 9),
    "min_samples_split": [.05, .1, .15],
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(X_train, y_train)

# 6b
pd.DataFrame(gs.cv_results_).pivot_table(values="mean_test_score", index="param_max_depth", columns="param_min_samples_split")