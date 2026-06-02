# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def simulate_once(n=100, trials=1_000_000, batch_size=10_000, seed=42):
    rng = np.random.default_rng(seed)

    success = np.zeros(n, dtype=np.int64)
    total = np.zeros(n, dtype=np.int64)

    finished = 0

    while finished < trials:
        current_batch = min(batch_size, trials - finished)
        finished += current_batch

        lives = np.argsort(
            rng.random((current_batch, n)),
            axis=1
        ).astype(np.int16) + 1

        row_idx = np.arange(current_batch)

        for k in range(1, n):
            standard = np.max(lives[:, :k], axis=1)
            tail = lives[:, k:]

            better = tail > standard[:, None]
            valid = np.any(better, axis=1)

            first_better = np.argmax(better, axis=1)
            selected_pos = k + first_better
            selected_score = lives[row_idx, selected_pos]

            success[k] += np.sum(valid & (selected_score == n))
            total[k] += current_batch

    rates = np.divide(
        success,
        total,
        out=np.zeros_like(success, dtype=float),
        where=total != 0
    )

    valid_k = np.arange(1, n)
    best_k = valid_k[np.argmax(rates[1:n])]
    best_rate = rates[best_k]

    return int(best_k), best_k / n, float(best_rate)


def repeat_experiment(
    n=100,
    trials=1_000_000,
    repeats=100,
    batch_size=10_000,
    seed=2024
):
    results = []

    for i in range(repeats):
        print(f"===== 第 {i + 1} / {repeats} 次重复实验 =====")

        best_k, best_ratio, best_rate = simulate_once(
            n=n,
            trials=trials,
            batch_size=batch_size,
            seed=seed + i
        )

        results.append({
            "重复编号": i + 1,
            "模拟次数": trials,
            "最优跳过人数": best_k,
            "最优跳过比例": best_ratio,
            "最高成功率": best_rate
        })

        print(
            f"最优 k={best_k}, "
            f"比例={best_ratio:.2%}, "
            f"成功率={best_rate:.4%}"
        )

        pd.DataFrame(results).to_csv(
            "实验_100万次重复100组_实时结果.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return pd.DataFrame(results)


def plot_result(df):
    counts = df["最优跳过人数"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("最优跳过人数 k")
    plt.ylabel("出现次数")
    plt.title("100 次重复实验中，最优 k 的分布")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_100次重复实验_最优k分布.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.hist(df["最高成功率"], bins=12)
    plt.xlabel("最高成功率")
    plt.ylabel("出现次数")
    plt.title("100 次重复实验中，最高成功率的分布")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_100次重复实验_最高成功率分布.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n = 100
    trials = 1_000_000
    repeats = 100
    batch_size = 10_000

    df = repeat_experiment(
        n=n,
        trials=trials,
        repeats=repeats,
        batch_size=batch_size,
        seed=2024
    )

    df.to_csv(
        "实验_100万次重复100组_最终结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(df.describe())
    plot_result(df)
