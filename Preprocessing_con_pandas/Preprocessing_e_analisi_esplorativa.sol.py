# ESERCIZIO 1

# 1a
(sales["Open"] == 0).sum()

# 1b
sales.loc[sales["Open"] == 1, "Customers"].mean()

# 1c
sales.loc[sales["Store"] == 123, "Sales"].sum()

# 1d
sales.sort_values("Sales", ascending=False).head(5)


# ESERCIZIO 2

# 2a
stores.isna().sum()

# 2b
stores.loc[stores["CompetitionDistance"].isna()]

# 2c
stores["CompetitionDistance"].fillna(1000).mean()


# ESERCIZIO 3

# 3a
(sales["Date"].dt.year == 2015).all() and (sales["Date"].dt.month == 7).all()

# 3b
sales["DayOfWeek"].equals(sales["Date"].dt.weekday + 1)


# ESERCIZIO 4

# 4a
per_customer_avg = sales_open["Sales"] / sales_open["Customers"]

# 4b
per_customer_avg.describe()

# 4c
pd.cut(per_customer_avg, 5).value_counts()


# ESERCIZIO 5

# 5a
pd.cut(sales_open["Customers"], 3).value_counts().plot.pie();

# 5b
stores.loc[stores["StoreType"] == "a", "Assortment"].value_counts().plot.pie();

# 5c
(sales_open["Sales"] / sales_open["Customers"]).plot.hist();

# 5d
(sales_open["Sales"] / sales_open["Customers"]).plot.box();

# 5e
sales_open.plot.scatter("CompetitionDistance", "Sales");

# 5f
sales_open.boxplot(column="Customers", by="StoreType", showmeans=True);