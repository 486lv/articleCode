# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def experiment_trials_checkpoints(
    n=100,
    checkpoints=None,
    batch_size=2_000_000,
    seed=42
):
    """
    模拟次数稳定性实验。
    这里用于“只看选中第 1 名”的情况，采用数学等价的快速统计方法。
    对于 10 亿次这种规模，严格生成完整排列会非常慢。
    """

    if checkpoints is None:
        checkpoints = [
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000
        ]

    checkpoints = sorted(checkpoints)
    max_trials = checkpoints[-1]
    rng = np.random.default_rng(seed)

    diff = np.zeros(n + 2, dtype=np.int64)
    results = []

    finished = 0
    checkpoint_index = 0

    while finished < max_trials:
        target = checkpoints[checkpoint_index]
        current_batch = min(batch_size, target - finished)

        best_pos = rng.integers(0, n, size=current_batch)

        # best_pos >= 1 时才可能成功
        mask = best_pos >= 1
        pos = best_pos[mask]

        if len(pos) > 0:
            # 最优对象前面的人里，最大者的位置
            # 如果 best_pos=1，那么前面只有位置0，prev_best_pos=0
            prev_best_pos = np.array([
                rng.integers(0, p) if p > 0 else 0
                for p in pos
            ])

            # 成功条件：prev_best_pos < k <= best_pos
            start = prev_best_pos + 1
            end = pos + 1

            np.add.at(diff, start, 1)
            np.add.at(diff, end, -1)

        finished += current_batch

        if finished == target:
            success = np.cumsum(diff)
            k_values = np.arange(1, n)
            rates = success[k_values] / finished

            best_idx = np.argmax(rates)
            best_k = int(k_values[best_idx])
            best_rate = float(rates[best_idx])

            results.append({
                "模拟次数": finished,
                "最优跳过人数": best_k,
                "最优跳过比例": best_k / n,
                "最高成功率": best_rate
            })

            print(
                f"trials={finished:,}, "
                f"best_k={best_k}, "
                f"ratio={best_k/n:.2%}, "
                f"rate={best_rate:.4%}"
            )

            checkpoint_index += 1
            if checkpoint_index >= len(checkpoints):
                break

    return pd.DataFrame(results)


def plot_trials(df):
    plt.figure(figsize=(9, 5))
    plt.plot(df["模拟次数"], df["最优跳过比例"], marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("模拟次数")
    plt.ylabel("最优跳过比例")
    plt.title("模拟次数增加后，最优观察比例是否稳定")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_模拟次数增加后的最优观察比例.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = experiment_trials_checkpoints(
        n=100,
        checkpoints=[
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000
        ],
        batch_size=2_000_000,
        seed=42
    )

    df.to_csv(
        "实验_不同模拟次数_最优结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(df)
    plot_trials(df)
