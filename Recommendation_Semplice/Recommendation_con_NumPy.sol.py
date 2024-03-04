# ESERCIZIO 1

for uid, iid in purchases_set:
    row = user_indices[uid]
    col = item_indices[iid]
    purchases[row, col] = 1


# ESERCIZIO 2

# 2a
# per ogni coppia ID, nome
for uid, name in users.items():
    # ottengo l'indice del vettore
    i = user_indices[uid]
    # inserisco il nome nel vettore
    user_names[i] = name

# 2b
item_names = np.empty(n_items, dtype=object)
for iid, name in items.items():
    item_names[item_indices[iid]] = name


# ESERCIZIO 3

# 3a
user_purchases.mean()

# 3b
item_purchases.max()

# 3c
user_names[user_purchases.argmax()]

# 3d
(user_purchases >= 50).sum()

# 3e
item_names[item_purchases >= 35]


# ESERCIZIO 4

# 4a
similarity.max()

# 4b
user_names[similarity[user_indices[7661]].argmax()]


# ESERCIZIO 5

# 5a
interest = similarity @ purchases

# 5b
interest.shape == purchases.shape

# 5c
interest[purchases_bool] = 0


# ESERCIZIO 6

# 6a
interest_ranking_user_0 = (-interest[0]).argsort().argsort()

# 6b
suggestions_for_user_0 = interest_ranking_user_0 < N

# 6c
interest_ranking = (-interest).argsort(1).argsort(1)
suggestions = interest_ranking < N

# 6d
item_names[suggestions[0]]


# ESERCIZIO 7

with open("purchases-2014.csv", "r") as f:
    for uid, iid in csv.reader(f, delimiter=";"):
        purchases_updated[user_indices[int(uid)], item_indices[int(iid)]] = 1


# ESERCIZIO 8

new_purchases = (purchases_updated - purchases).astype(bool)


# ESERCIZIO 9

hits = suggestions & new_purchases


# ESERCIZIO 10

# 10a
satisfied_users = hits.any(1)

# 10b
satisfied_users.sum()

# 10c
satisfied_users.mean()


# ESERCIZIO EXTRA

# a
random_interest = np.random.random((n_users, n_items))

# b
random_interest[purchases_bool] = 0

# c
random_interest_ranking = (-random_interest).argsort(1).argsort(1)
random_suggestions = random_interest_ranking < N

# d
random_hits = random_suggestions & new_purchases
randomly_satisfied_users = random_hits.any(1)

# e
randomly_satisfied_users.mean()