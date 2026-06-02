# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def generate_two_way_scores(batch_size, n, rho, rng):
    x = rng.normal(size=(batch_size, n))
    z = rng.normal(size=(batch_size, n))
    y = rho * x + np.sqrt(1 - rho ** 2) * z

    my_score = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.int16) + 1
    their_score = np.argsort(np.argsort(y, axis=1), axis=1).astype(np.int16) + 1

    return my_score, their_score


def run_two_way_simulation(
    n=100,
    trials=100000,
    batch_size=10000,
    accept_threshold=60,
    rho=0.3,
    start_age=18,
    end_age=40,
    seed=42
):
    rng = np.random.default_rng(seed)

    match_count = np.zeros(n, dtype=np.int64)
    success_global_best = np.zeros(n, dtype=np.int64)
    success_best_feasible = np.zeros(n, dtype=np.int64)

    selected_score_sum = np.zeros(n, dtype=np.float64)
    selected_age_sum = np.zeros(n, dtype=np.float64)

    total = np.zeros(n, dtype=np.int64)

    finished = 0

    while finished < trials:
        current_batch = min(batch_size, trials - finished)
        finished += current_batch

        my_score, their_score = generate_two_way_scores(
            batch_size=current_batch,
            n=n,
            rho=rho,
            rng=rng
        )

        row_idx = np.arange(current_batch)
        willing = their_score >= accept_threshold

        feasible_score = np.where(willing, my_score, -1)
        feasible_exists = np.any(willing, axis=1)
        best_feasible_pos = np.argmax(feasible_score, axis=1)

        for k in range(1, n):
            standard = np.max(my_score[:, :k], axis=1)

            tail_my = my_score[:, k:]
            tail_willing = willing[:, k:]

            can_match = (tail_my > standard[:, None]) & tail_willing
            valid = np.any(can_match, axis=1)

            first_match = np.argmax(can_match, axis=1)
            selected_pos = k + first_match
            selected_my_score = my_score[row_idx, selected_pos]
            selected_age = start_age + (selected_pos / (n - 1)) * (end_age - start_age)

            match_count[k] += np.sum(valid)
            success_global_best[k] += np.sum(valid & (selected_my_score == n))
            success_best_feasible[k] += np.sum(
                valid & feasible_exists & (selected_pos == best_feasible_pos)
            )

            selected_score_sum[k] += np.sum(selected_my_score[valid])
            selected_age_sum[k] += np.sum(selected_age[valid])
            total[k] += current_batch

        print(f"已完成 {finished:,} / {trials:,} 次模拟")

    k_values = np.arange(1, n)

    result = pd.DataFrame({
        "跳过人数": k_values,
        "跳过比例": k_values / n,
        "开始认真选择的年龄": start_age + (k_values / n) * (end_age - start_age),
        "匹配成功率": match_count[1:n] / total[1:n],
        "选中全局最合适且对方愿意的概率":
            success_global_best[1:n] / total[1:n],
        "选中最佳可成关系的概率":
            success_best_feasible[1:n] / total[1:n],
        "成功匹配时的平均满意度": np.divide(
            selected_score_sum[1:n],
            match_count[1:n],
            out=np.zeros_like(selected_score_sum[1:n], dtype=float),
            where=match_count[1:n] != 0
        ),
        "平均真正匹配年龄": np.divide(
            selected_age_sum[1:n],
            match_count[1:n],
            out=np.zeros_like(selected_age_sum[1:n], dtype=float),
            where=match_count[1:n] != 0
        )
    })

    return result


def summarize_best_results(result):
    metrics = [
        "匹配成功率",
        "选中全局最合适且对方愿意的概率",
        "选中最佳可成关系的概率",
        "成功匹配时的平均满意度"
    ]

    rows = []

    for metric in metrics:
        best_row = result.loc[result[metric].idxmax()]

        rows.append({
            "指标": metric,
            "最优跳过人数": int(best_row["跳过人数"]),
            "最优跳过比例": best_row["跳过比例"],
            "开始认真选择的年龄": best_row["开始认真选择的年龄"],
            "最高值": best_row[metric],
            "平均真正匹配年龄": best_row["平均真正匹配年龄"]
        })

    return pd.DataFrame(rows)


def plot_single_metric(result, metric, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(
        result["开始认真选择的年龄"],
        result[metric],
        linewidth=2
    )
    plt.xlabel("开始认真选择的年龄")
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all_results(result):
    plot_single_metric(result, "匹配成功率", "图_双向选择_匹配成功率.png")
    plot_single_metric(result, "选中全局最合适且对方愿意的概率", "图_双向选择_选中全局最合适且对方愿意.png")
    plot_single_metric(result, "选中最佳可成关系的概率", "图_双向选择_选中最佳可成关系.png")
    plot_single_metric(result, "成功匹配时的平均满意度", "图_双向选择_成功匹配时的平均满意度.png")


def threshold_experiment(
    n=100,
    trials=100000,
    batch_size=10000,
    thresholds=None,
    rho=0.3,
    start_age=18,
    end_age=40,
    seed=100
):
    if thresholds is None:
        thresholds = [40, 50, 60, 70, 80, 90]

    rows = []

    for threshold in thresholds:
        print(f"===== 对方愿意门槛 = {threshold} =====")

        result = run_two_way_simulation(
            n=n,
            trials=trials,
            batch_size=batch_size,
            accept_threshold=threshold,
            rho=rho,
            start_age=start_age,
            end_age=end_age,
            seed=seed + threshold
        )

        summary = summarize_best_results(result)
        summary["对方愿意门槛"] = threshold
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def plot_threshold_summary(summary):
    df = summary[summary["指标"] == "选中最佳可成关系的概率"].copy()

    plt.figure(figsize=(9, 5))
    plt.plot(
        df["对方愿意门槛"],
        df["开始认真选择的年龄"],
        marker="o",
        linewidth=2
    )
    plt.xlabel("对方愿意门槛")
    plt.ylabel("最佳开始认真选择年龄")
    plt.title("对方越挑剔，最佳开始年龄如何变化？")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_双向选择_不同门槛下最佳开始年龄.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n = 100
    trials = 100000
    batch_size = 10000

    start_age = 18
    end_age = 40
    accept_threshold = 60
    rho = 0.3

    result = run_two_way_simulation(
        n=n,
        trials=trials,
        batch_size=batch_size,
        accept_threshold=accept_threshold,
        rho=rho,
        start_age=start_age,
        end_age=end_age,
        seed=42
    )

    result.to_csv(
        "双向选择模型_18到40岁_主实验结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = summarize_best_results(result)
    summary.to_csv(
        "双向选择模型_18到40岁_最优结果汇总.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(summary)
    plot_all_results(result)

    threshold_summary = threshold_experiment(
        n=n,
        trials=trials,
        batch_size=batch_size,
        thresholds=[40, 50, 60, 70, 80, 90],
        rho=rho,
        start_age=start_age,
        end_age=end_age,
        seed=100
    )

    threshold_summary.to_csv(
        "双向选择模型_18到40岁_不同对方门槛汇总.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(threshold_summary)
    plot_threshold_summary(threshold_summary)
