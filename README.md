# 公众号文章实验代码：秘书问题 / 37% 法则

这份代码整理了文章中用到的主要实验。

## 文件说明

| 文件 | 用途 |
|---|---|
| `01_basic_simulation.py` | 基础实验：100 个候选人，遍历观察人数，画成功率曲线 |
| `02_stability_trials.py` | 模拟次数稳定性实验：10万、100万、1000万、1亿、10亿 |
| `03_repeat_1m_100_runs.py` | 严格重复实验：100万次模拟重复 100 组 |
| `04_top_m_1_to_100.py` | 成功标准从“前1名”放宽到“前100名” |
| `05_two_way_choice.py` | 双向选择模型：我愿意 + 对方也愿意 |
| `06_noise_and_avoid_worst.py` | 判断误差 + 避免选到后20名 |
| `07_combine_images.py` | 把两张图片上下拼接成一张图 |
| `requirements.txt` | 依赖库 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python 01_basic_simulation.py
```

为了方便先跑通，部分脚本默认 `trials=100000`。如果需要更稳定的结果，可以把脚本底部的参数改成：

```python
trials = 1_000_000
```

## 注意

- `02_stability_trials.py` 中的 10 亿次实验使用了数学等价的快速统计方法，否则运行时间会非常长。
- `03_repeat_1m_100_runs.py` 是严格版，会真的重复 100 次、每次 100 万次模拟，运行时间较长。
- 图片默认保存到当前目录。
