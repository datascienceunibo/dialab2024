# ESERCIZIO 1

# 1a
sales.loc[sales["Customers"] >= 4000]

# 1b
sales.loc[sales["DayOfWeek"] == "Sun", "Sales"].mean()

# 1c
sales["CompetitionDistance"].isna().sum()

# 1d
sales["Date"].value_counts()


# ESERCIZIO 2

# 2a
sales.loc[("2015-07-01", 42), "Customers"]

# 2b
sales.loc["2015-07-02", "Sales"].sum()

# 2c
sales.loc[("2015-07-03", [2, 15, 18]), "Sales"].sum()


# ESERCIZIO 3

# 3a
sales.groupby(["DayOfWeek", "Promo"])["Customers"].mean()

# 3b
sales.groupby(sales["DayOfWeek"].isin(["Sat", "Sun"]))[["Sales", "Customers"]].mean()

# 3c
sales.groupby("Date")["Sales"].sum().idxmax()

# 3d
(
    # ordino il frame per ricavi discendenti
    sales.sort_values("Sales", ascending=False)
    # partiziono il frame per date
    .groupby("Date")
    # per ogni gruppo prendo la prima riga
    .head(1)
    # dal frame risultante prendo i valori del livello Store dell'indice
    .index.get_level_values("Store")
    # conto le occorrenze di ciascun valore
    .value_counts()
)


# ESERCIZIO 4

# 4a
sales.groupby(["DayOfWeek", "Promo"])["Customers"].mean().unstack()

# 4b
sales.pivot_table(
    values=["Customers"],
    index=["DayOfWeek"],
    columns=["Promo"],
    aggfunc=["mean"],
)

# 4c
sales["Sales"].unstack()

# 4d
sales["Sales"].unstack().idxmax(axis=1).value_counts()