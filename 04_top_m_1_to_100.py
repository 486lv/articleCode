# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def experiment_top_m_1_to_100(
    n=100,
    trials=1_000_000,
    batch_size=20_000,
    seed=123
):
    rng = np.random.default_rng(seed)

    rank_counts = np.zeros((n, n + 1), dtype=np.int64)

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
            threshold = np.max(lives[:, :k], axis=1)
            tail = lives[:, k:]
            better = tail > threshold[:, None]

            valid = np.any(better, axis=1)
            first_better = np.argmax(better, axis=1)
            selected_pos = k + first_better
            selected_score = lives[row_idx, selected_pos]

            selected_rank = n - selected_score + 1
            ranks = selected_rank[valid]

            counts = np.bincount(ranks, minlength=n + 1)
            rank_counts[k] += counts

        print(f"已完成 {finished:,} / {trials:,}")

    results = []

    for top_m in range(1, n + 1):
        best_k = None
        best_rate = -1

        for k in range(1, n):
            success_count = rank_counts[k, 1:top_m + 1].sum()
            rate = success_count / trials

            if rate > best_rate:
                best_rate = rate
                best_k = k

        results.append({
            "top_m": top_m,
            "成功标准": f"选中前{top_m}名",
            "最优跳过人数": int(best_k),
            "最优跳过比例": best_k / n,
            "最高成功率": float(best_rate)
        })

    return pd.DataFrame(results)


def add_marginal_gain(df):
    df = df.copy()
    df["边际成功率提升"] = df["最高成功率"].diff()
    df.loc[df["top_m"] == 1, "边际成功率提升"] = np.nan
    return df


def plot_top_m(df):
    plt.figure(figsize=(9, 5))
    plt.plot(df["top_m"], df["最优跳过比例"], linewidth=2)
    plt.xlabel("成功标准：选中前 m 名")
    plt.ylabel("最优观察比例")
    plt.title("成功标准放宽后，最优观察比例如何变化")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_top_m与最优观察比例.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(df["top_m"], df["最高成功率"], linewidth=2)
    plt.xlabel("成功标准：选中前 m 名")
    plt.ylabel("最高成功率")
    plt.title("成功标准放宽后，最高成功率如何变化")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_top_m与最高成功率.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(df["top_m"], df["边际成功率提升"], linewidth=2)
    plt.axhline(0.01, linestyle="--", linewidth=1)
    plt.xlabel("成功标准：选中前 m 名")
    plt.ylabel("边际成功率提升")
    plt.title("每多接受一个名次，能多换来多少成功率？")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_top_m边际收益.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = experiment_top_m_1_to_100(
        n=100,
        trials=1_000_000,
        batch_size=20_000,
        seed=123
    )

    df = add_marginal_gain(df)

    df.to_csv(
        "实验_top_m从1到100_最优结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(df.head(20))
    plot_top_m(df)
