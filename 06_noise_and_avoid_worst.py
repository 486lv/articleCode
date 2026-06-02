# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def simulate_noise_and_risk(
    n=100,
    trials=100000,
    batch_size=10000,
    sigmas=None,
    top_m=5,
    bad_m=20,
    start_age=18,
    end_age=40,
    seed=42
):
    if sigmas is None:
        sigmas = [0, 2, 5, 10, 15, 20, 30]

    rng = np.random.default_rng(seed)

    detail_rows = []
    summary_rows = []

    for sigma in sigmas:
        print(f"===== 判断误差 sigma = {sigma} =====")

        success_best = np.zeros(n, dtype=np.int64)
        success_top_m = np.zeros(n, dtype=np.int64)
        bad_count = np.zeros(n, dtype=np.int64)
        selected_score_sum = np.zeros(n, dtype=np.float64)
        regret_sum = np.zeros(n, dtype=np.float64)
        total = np.zeros(n, dtype=np.int64)

        finished = 0

        while finished < trials:
            current_batch = min(batch_size, trials - finished)
            finished += current_batch

            true_score = np.array([
                rng.permutation(np.arange(1, n + 1))
                for _ in range(current_batch)
            ])

            noise = rng.normal(loc=0, scale=sigma, size=(current_batch, n))
            observed_score = true_score + noise

            row_idx = np.arange(current_batch)

            for k in range(1, n):
                standard = np.max(observed_score[:, :k], axis=1)
                tail_observed = observed_score[:, k:]
                better = tail_observed > standard[:, None]

                valid = np.any(better, axis=1)
                first_better = np.argmax(better, axis=1)

                # 如果后面没人超过标准，强制选择最后一个人
                selected_pos = np.where(valid, k + first_better, n - 1)
                selected_true_score = true_score[row_idx, selected_pos]

                success_best[k] += np.sum(selected_true_score == n)
                success_top_m[k] += np.sum(selected_true_score >= n - top_m + 1)
                bad_count[k] += np.sum(selected_true_score <= bad_m)
                selected_score_sum[k] += np.sum(selected_true_score)
                regret_sum[k] += np.sum(n - selected_true_score)
                total[k] += current_batch

            print(f"已完成 {finished:,} / {trials:,}")

        k_values = np.arange(1, n)
        age_values = start_age + (k_values / n) * (end_age - start_age)

        best_rate = success_best[1:n] / total[1:n]
        top_m_rate = success_top_m[1:n] / total[1:n]
        bad_rate = bad_count[1:n] / total[1:n]
        avoid_bad_rate = 1 - bad_rate
        avg_score = selected_score_sum[1:n] / total[1:n]
        avg_regret = regret_sum[1:n] / total[1:n]

        for i, k in enumerate(k_values):
            detail_rows.append({
                "判断误差sigma": sigma,
                "跳过人数": int(k),
                "跳过比例": k / n,
                "开始认真选择年龄": age_values[i],
                "选中最合适对象的概率": best_rate[i],
                f"选中前{top_m}名对象的概率": top_m_rate[i],
                f"选到后{bad_m}名对象的概率": bad_rate[i],
                f"避开后{bad_m}名对象的概率": avoid_bad_rate[i],
                "平均真实满意度": avg_score[i],
                "平均后悔值": avg_regret[i]
            })

        idx_best = np.argmax(best_rate)
        idx_top = np.argmax(top_m_rate)
        idx_avoid_bad = np.argmin(bad_rate)
        idx_avg_score = np.argmax(avg_score)

        summary_rows.append({
            "判断误差sigma": sigma,

            "选中最合适_最优跳过人数": int(k_values[idx_best]),
            "选中最合适_最优比例": k_values[idx_best] / n,
            "选中最合适_最佳年龄": age_values[idx_best],
            "选中最合适_最高成功率": best_rate[idx_best],

            f"选中前{top_m}名_最优跳过人数": int(k_values[idx_top]),
            f"选中前{top_m}名_最优比例": k_values[idx_top] / n,
            f"选中前{top_m}名_最佳年龄": age_values[idx_top],
            f"选中前{top_m}名_最高成功率": top_m_rate[idx_top],

            f"避开后{bad_m}名_最优跳过人数": int(k_values[idx_avoid_bad]),
            f"避开后{bad_m}名_最优比例": k_values[idx_avoid_bad] / n,
            f"避开后{bad_m}名_最佳年龄": age_values[idx_avoid_bad],
            f"避开后{bad_m}名_最低踩坑率": bad_rate[idx_avoid_bad],
            f"避开后{bad_m}名_最高避坑率": avoid_bad_rate[idx_avoid_bad],

            "平均满意度最高_最优跳过人数": int(k_values[idx_avg_score]),
            "平均满意度最高_最优比例": k_values[idx_avg_score] / n,
            "平均满意度最高_最佳年龄": age_values[idx_avg_score],
            "平均满意度最高_满意度": avg_score[idx_avg_score],
            "平均满意度最高_后悔值": avg_regret[idx_avg_score]
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def plot_judgment_error(summary_df):
    plt.figure(figsize=(9, 5))
    plt.plot(
        summary_df["判断误差sigma"],
        summary_df["选中最合适_最高成功率"],
        marker="o",
        linewidth=2
    )
    plt.xlabel("判断误差 sigma")
    plt.ylabel("选中最合适对象的最高成功率")
    plt.title("判断误差越大，选中最合适对象有多难？")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_判断误差与最高成功率.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(
        summary_df["判断误差sigma"],
        summary_df["选中最合适_最佳年龄"],
        marker="o",
        linewidth=2
    )
    plt.xlabel("判断误差 sigma")
    plt.ylabel("最佳开始认真选择年龄")
    plt.title("判断误差越大，最佳开始年龄如何变化？")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_判断误差与最佳年龄.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_avoid_worst(detail_df, sigma=0, bad_m=20):
    df = detail_df[detail_df["判断误差sigma"] == sigma].copy()

    plt.figure(figsize=(9, 5))
    plt.plot(
        df["开始认真选择年龄"],
        df[f"选到后{bad_m}名对象的概率"],
        linewidth=2
    )
    plt.xlabel("开始认真选择年龄")
    plt.ylabel(f"选到后{bad_m}名对象的概率")
    plt.title("如果目标是避免最差，什么时候开始选择更安全？")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"图_避免后{bad_m}名_踩坑率.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(
        df["开始认真选择年龄"],
        df["平均真实满意度"],
        linewidth=2
    )
    plt.xlabel("开始认真选择年龄")
    plt.ylabel("平均真实满意度")
    plt.title("不同开始年龄下，最终选择对象的平均质量")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("图_平均真实满意度.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n = 100
    trials = 100000
    batch_size = 10000

    start_age = 18
    end_age = 40

    sigmas = [0, 2, 5, 10, 15, 20, 30]
    top_m = 5
    bad_m = 20

    detail_df, summary_df = simulate_noise_and_risk(
        n=n,
        trials=trials,
        batch_size=batch_size,
        sigmas=sigmas,
        top_m=top_m,
        bad_m=bad_m,
        start_age=start_age,
        end_age=end_age,
        seed=42
    )

    detail_df.to_csv(
        "实验_判断误差与避免最差_详细结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        "实验_判断误差与避免最差_汇总结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(summary_df)

    plot_judgment_error(summary_df)
    plot_avoid_worst(detail_df, sigma=0, bad_m=bad_m)
