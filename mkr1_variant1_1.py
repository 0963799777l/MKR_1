import math
from collections import Counter

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression


def entropy(probs):
    return -sum(p * math.log(p, 2) for p in probs if p > 0)


def compute_gain_for_smoking():
    data = [
        ("Так", "Так"),
        ("Так", "Так"),
        ("Так", "Ні"),
        ("Так", "Так"),
        ("Так", "Ні"),
        ("Так", "Ні"),
        ("Так", "Ні"),
        ("Так", "Ні"),
        ("Так", "Ні"),
        ("Так", "Ні"),
        ("Ні", "Так"),
        ("Ні", "Так"),
        ("Ні", "Так"),
        ("Ні", "Так"),
        ("Ні", "Ні"),
        ("Ні", "Ні"),
        ("Ні", "Так"),
        ("Ні", "Ні"),
        ("Ні", "Ні"),
        ("Ні", "Ні"),
    ]

    N = len(data)

    bronchitis_counts = Counter(b for _, b in data)
    p_yes = bronchitis_counts["Так"] / N
    p_no = bronchitis_counts["Ні"] / N

    H_S = entropy([p_yes, p_no])

    groups = {"Так": [], "Ні": []}
    for smoking, bronch in data:
        groups[smoking].append(bronch)

    H_cond = 0.0
    split_probs = []

    for smoking_value, bronchs in groups.items():
        Nj = len(bronchs)
        split_probs.append(Nj / N)
        counts_j = Counter(bronchs)
        p_yes_j = counts_j["Так"] / Nj
        p_no_j = counts_j["Ні"] / Nj
        H_j = entropy([p_yes_j, p_no_j])
        H_cond += (Nj / N) * H_j

        print(f'H(S | Куріння = "{smoking_value}") = {H_j:.6f} біт')

    Gain = H_S - H_cond

    split_info = entropy(split_probs)
    gain_ratio = Gain / split_info if split_info > 0 else 0.0

    print("РЕЗУЛЬТАТИ ДЛЯ ОЗНАКИ 'Куріння'")
    print(f"H(S)               = {H_S:.6f} біт")
    print(f"H(S | X)           = {H_cond:.6f} біт")
    print(f"Gain               = {Gain:.6f} біт")
    print(f"SplitInfo          = {split_info:.6f} біт")
    print(f"Gain-ratio         = {gain_ratio:.66f}")

    return {
        "H_S": H_S,
        "H_cond": H_cond,
        "Gain": Gain,
        "SplitInfo": split_info,
        "GainRatio": gain_ratio,
    }


def demo_kfold_logreg():

    rng = np.random.RandomState(0)
    X_class0 = rng.normal(loc=-1.0, scale=1.0, size=(50, 2))
    X_class1 = rng.normal(loc=+1.0, scale=1.0, size=(50, 2))
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 50 + [1] * 50)

    model = LogisticRegression(max_iter=1000)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv)

    print("ПЕРЕХРЕСНА ПЕРЕВІРКА (5-Fold) ДЛЯ ЛОГІСТИЧНОЇ РЕГРЕСІЇ")
    print("Окремі значення accuracy:", scores)
    print("Середня точність        :", scores.mean())
    print("Стандартне відхилення   :", scores.std())


if __name__ == "__main__":
    compute_gain_for_smoking()

    demo_kfold_logreg()
