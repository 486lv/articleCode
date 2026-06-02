# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def run_simulation(n=100, trials=100000, batch_size=10000, top_m=5, seed=42):
    rng = np.random.default_rng(seed)

    success_best = np.zeros(n + 1, dtype=np.int64)
    success_top_m = np.zeros(n + 1, dtype=np.int64)
    total_count = np.zeros(n + 1, dtype=np.int64)

    finished = 0

    while finished < trials:
        current_batch = min(batch_size, trials - finished)
        finished += current_batch

        lives = np.array([
            rng.permutation(np.arange(1, n + 1))
            for _ in range(current_batch)
        ])

        row_idx = np.arange(current_batch)
        top_threshold = n - top_m + 1

        for k in range(1, n):
            threshold = np.max(lives[:, :k], axis=1)
            tail = lives[:, k:]
            better = tail > threshold[:, None]

            valid = np.any(better, axis=1)
            first_better = np.argmax(better, axis=1)
            selected_pos = k + first_better
            selected_score = lives[row_idx, selected_pos]

            success_best[k] += np.sum(valid & (selected_score == n))
            success_top_m[k] += np.sum(valid & (selected_score >= top_threshold))
            total_count[k] += current_batch

        print(f"已完成 {finished:,} / {trials:,} 次模拟")

    k_values = np.arange(1, n)

    result = pd.DataFrame({
        "跳过人数": k_values,
        "跳过比例": k_values / n,
        "选中最合适对象的概率": success_best[1:n] / total_count[1:n],
        f"选中前{top_m}名对象的概率": success_top_m[1:n] / total_count[1:n],
    })

    return result


def plot_result(result, n=100, top_m=5, start_age=18, end_age=40):
    result = result.copy()
    result["开始认真选择的年龄"] = (
        start_age + result["跳过比例"] * (end_age - start_age)
    )

    best_row = result.loc[result["选中最合适对象的概率"].idxmax()]
    print("\n===== 实验结果 =====")
    print(f"最优跳过人数 k = {int(best_row['跳过人数'])}")
    print(f"最优跳过比例 = {best_row['跳过比例']:.2%}")
    print(f"对应年龄 ≈ {best_row['开始认真选择的年龄']:.2f} 岁")
    print(f"成功率 = {best_row['选中最合适对象的概率']:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        result["开始认真选择的年龄"],
        result["选中最合适对象的概率"],
        linewidth=2
    )
    plt.xlabel("开始认真选择的年龄", fontsize=13)
    plt.ylabel("选中最合适对象的概率", fontsize=13)
    plt.title("你这一生在什么时候选择对的人最合适？", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图1_什么时候选择对的人最合适.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(
        result["开始认真选择的年龄"],
        result[f"选中前{top_m}名对象的概率"],
        linewidth=2
    )
    plt.xlabel("开始认真选择的年龄", fontsize=13)
    plt.ylabel(f"选中前{top_m}名对象的概率", fontsize=13)
    plt.title(f"不追求唯一最好时，选中前 {top_m} 名对象的概率", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"图2_选中前{top_m}名对象的概率.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n = 100
    trials = 100000
    batch_size = 10000
    top_m = 5

    start_age = 18
    end_age = 40

    result = run_simulation(
        n=n,
        trials=trials,
        batch_size=batch_size,
        top_m=top_m,
        seed=42
    )

    result.to_csv(
        "基础实验_模拟结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_result(
        result=result,
        n=n,
        top_m=top_m,
        start_age=start_age,
        end_age=end_age
    )
