import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

DATASET_PATH = "datasets/cakes_train.csv"

df = pd.read_csv(DATASET_PATH)
df["eggs"] = df["eggs"] * 63

print("\nFirst 5 rows:\n", df.head())

print("\nLast 5 rows:\n", df.tail())

print("\nInfo:")
df.info()

print("\nDescribe:\n", df.describe())

features = ["flour", "eggs", "sugar", "milk", "butter", "baking_powder"]

for feature in features:
    plt.figure(figsize=(6, 4))

    m_mask = df["type"].str.lower() == "muffin"
    c_mask = df["type"].str.lower() == "cupcake"

    plt.scatter(df.loc[m_mask, feature], np.zeros(m_mask.sum()), alpha=0.7, label="Muffin")
    plt.scatter(df.loc[c_mask, feature], np.ones(c_mask.sum()), alpha=0.7, label="Cupcake")

    plt.yticks([0, 1], ["Muffin", "Cupcake"])
    plt.xlabel(feature); plt.ylabel("type"); plt.title(f"{feature} vs type")

    plt.legend()
    plt.tight_layout()
    plt.show()

corr = df[features].corr()

plt.figure(figsize=(6, 5))
im = plt.imshow(corr, interpolation="nearest")

plt.title("Correlation Matrix")
plt.colorbar(im, fraction=0.046, pad=0.04)

tick_pos = range(len(corr.columns))

plt.xticks(tick_pos, corr.columns, rotation=45, ha="right")
plt.yticks(tick_pos, corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
plt.tight_layout()
plt.show()

X = df[features].copy()
y = df["type"].copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
acc_train = clf.score(X_train, y_train)
acc_val = accuracy_score(y_val, y_pred)
err_val = 1.0 - acc_val

print(f"\nAccuracy (train): {acc_train:.4f}")
print(f"Accuracy (val): {acc_val:.4f}")
print(f"Error (1 - Accuracy): {err_val:.4f}")

plt.figure(figsize=(16, 9))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, impurity=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()