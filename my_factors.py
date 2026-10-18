"""
my_factors.py — 三个 Alpha 因子（单只股票）

输入
----
df : 一只股票的日线，已按日期从早到晚排序。
     列至少要有 open, high, low, close, vol, amount。
     索引保持原样（日期或行号都可以）。

输出
----
pd.Series，索引与 df 相同，值是当天的因子分数。
窗口不够的日期是 NaN。

关键假设
----
- 价格、成交量按调用方给的原样使用（本文件不做复权）。
- 三个因子都是沿时间计算，所以一只股票一张表就够。

已知局限
----
- Alpha#23 的公式来自论文，Parsnip77 仓库没有实现。
- 未做去极值、标准化；原始分数尺度各不相同。
- amount 本课三个公式都没用到，列要求只是为了和日线表对齐。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_sorted_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """缺列就停。为什么：公式绑死列名，静默用错列会算出看似正常的错数。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少列: {missing}")


def calc_alpha6(df: pd.DataFrame) -> pd.Series:
    """Alpha#6。

    公式来源：101 Formulaic Alphas，Alpha#6
        (-1) * correlation(open, volume, 10)
    经济逻辑：开盘价和成交量越是同向变动，分数越低。
    """
    _require_sorted_columns(df, ["open", "vol"])

    # 价量是否同向：公式里的 correlation(open, volume, 10)
    corr_10 = df["open"].rolling(window=10, min_periods=10).corr(df["vol"])
    # 取负：同向相关越强，因子越负
    alpha = -corr_10
    alpha.name = "alpha6"
    return alpha


def calc_alpha12(df: pd.DataFrame) -> pd.Series:
    """Alpha#12。

    公式来源：101 Formulaic Alphas，Alpha#12
        sign(delta(volume, 1)) * (-1 * delta(close, 1))
    经济逻辑：成交量变大时，对当天收盘涨跌取反。
    """
    _require_sorted_columns(df, ["close", "vol"])

    # 量能方向的原料：公式里的 delta(volume, 1)，后面只保留符号
    vol_chg_1 = df["vol"].diff(1)
    # 把量的变动收成 +1 / -1 / 0，对应公式里的 sign(...)
    vol_sign = np.sign(vol_chg_1)
    # 当天价格变动：公式里的 delta(close, 1)
    close_chg_1 = df["close"].diff(1)
    # 量增则逆着价格变动打分，量缩则顺着
    alpha = vol_sign * (-close_chg_1)
    alpha.name = "alpha12"
    return alpha


def calc_alpha23(df: pd.DataFrame) -> pd.Series:
    """Alpha#23。

    公式来源：101 Formulaic Alphas，Alpha#23
        ((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0
    经济逻辑：最高价站上近 20 日均高时，对高点的两日变化取反；否则给 0。
    仓库 Parsnip77 没有这个因子。
    """
    _require_sorted_columns(df, ["high"])

    # 近期高点基准：公式里的 sum(high, 20) / 20
    avg_high_20 = df["high"].rolling(window=20, min_periods=20).mean()
    # 高点两日变化：公式里的 delta(high, 2)，只在条件成立时使用
    high_chg_2 = df["high"].diff(2)

    # 今天的高点是否高于近 20 日平均高点
    above = avg_high_20 < df["high"]
    # 条件成立用 -delta；不成立用 0。窗口不足时 avg 为 NaN，结果保持 NaN
    alpha = pd.Series(np.where(above, -high_chg_2, 0.0), index=df.index)
    alpha = alpha.where(avg_high_20.notna())
    alpha.name = "alpha23"
    return alpha
