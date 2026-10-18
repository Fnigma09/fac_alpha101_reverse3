"""
my_combine.py — 用滚动 RankIC 加权合成一个 composite 因子，并和单因子对比。

输入
----
data/factors_clean.parquet
data/prices.parquet（用来做 1 日远期收益）

输出
----
打印一张表：alpha6、alpha12、alpha23、composite
列：RankIC 均值、多空年化、夏普、最大回撤
另存 plots/{因子}_ic.png、plots/{因子}_backtest.png（含 composite）

计算逻辑
----
1. 每个交易日截面 RankIC（Spearman）
2. 在调仓日 t，权重用 t 之前 252 个交易日的 IC 均值（不含当天）
3. w_i = IC均值_i / sum(|IC均值|)
4. composite = w6*alpha6 + w12*alpha12 + w23*alpha23
5. 对比表只使用 composite 有权重的那些交易日

关键假设
----
- 当天 IC 要用到次日收益，调仓当天还不知道，所以权重用 shift 后的历史
- |IC| 只出现在分母里；分子保留符号，负 IC 会把该因子反过来加
- 夏普 = 日收益均值 / 日收益标准差 * sqrt(252)，无风险利率按 0
- 最大回撤用多空日收益的累计净值
- 不扣手续费

已知局限
----
- 滚动 252 日会吃掉开头约一年，有效交易日变少
- 权重只看各自的 IC，不看因子之间是否重复（未做共线性 / 回归合成）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from my_evaluate import (
    FACTOR_COLS,
    PLOTS_DIR,
    TRADING_DAYS,
    calc_ic,
    ic_by_date,
    layer_backtest,
    layer_daily,
    make_daily_panel,
    plot_ic,
    plot_layer_backtest,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ROLL_DAYS = TRADING_DAYS


def rolling_ic_weights(ic_wide: pd.DataFrame, window: int = ROLL_DAYS) -> pd.DataFrame:
    """每个调仓日一行权重。

    为什么用 IC 加权而不是等权：预测力更强的因子权重大一些。
    为什么用滚动窗口而不是全样本：全样本会用到未来日子的 IC，容易把历史拟合得太好（过拟合）。
    为什么先 shift(1)：当天 RankIC 含有次日收益，调仓日当天还不能用。
    分母用 |IC| 之和：让权重大小加起来为 1；分子保留正负，负 IC 因子会被减掉而不是加反。
    """
    past_mean = ic_wide.shift(1).rolling(window, min_periods=window).mean()
    denom = past_mean.abs().sum(axis=1).replace(0, np.nan)
    return past_mean.div(denom, axis=0)


def build_composite(day_factors: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """把三个因子按当日权重加成 composite。没有权重的交易日整行丢掉。"""
    w = weights.dropna(how="any")
    fac = day_factors[day_factors["date"].isin(w.index)].copy()
    for col in FACTOR_COLS:
        fac[col + "_w"] = fac["date"].map(w[col])
    # 三个因子都在时才合成，避免缺一个因子时权重含义变了
    fac = fac.dropna(subset=FACTOR_COLS + [c + "_w" for c in FACTOR_COLS])
    fac["composite"] = 0.0
    for col in FACTOR_COLS:
        fac["composite"] = fac["composite"] + fac[col] * fac[col + "_w"]
    return fac[["date", "ts_code", "composite"]]


def _sharpe_mdd(daily_ret: pd.Series) -> tuple[float, float]:
    """多空日收益 → 夏普、最大回撤。

    夏普用日频年化（sqrt(252)），无风险利率取 0。
    最大回撤是累计净值相对历史高点的最大跌幅，负数。
    """
    r = daily_ret.dropna()
    if len(r) < 2 or r.std() == 0 or np.isnan(r.std()):
        return np.nan, np.nan
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))
    nav = (1.0 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    return sharpe, mdd


def evaluate_one(name: str, factors_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """和单因子同一套 IC、分层；额外给出夏普和最大回撤。"""
    ic = calc_ic(factors_df, returns_df)
    bt = layer_backtest(factors_df, returns_df)
    panel = layer_daily(factors_df, returns_df)
    sharpe, mdd = _sharpe_mdd(panel["ls"]) if len(panel) else (np.nan, np.nan)
    return {
        "factor": name,
        "n_days": bt["n_days"],
        "rank_ic": ic["ic_mean"],
        "ls_ann": bt["ls_ann"],
        "sharpe": sharpe,
        "max_dd": mdd,
    }


def main() -> None:
    factors = pd.read_parquet(DATA_DIR / "factors_clean.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    day_fac, returns_df = make_daily_panel(factors, prices)

    ic_cols = {}
    for col in FACTOR_COLS:
        ic_cols[col] = ic_by_date(day_fac[["date", "ts_code", col]], returns_df)
    ic_wide = pd.DataFrame(ic_cols).sort_index()

    weights = rolling_ic_weights(ic_wide, ROLL_DAYS)
    composite = build_composite(day_fac, weights)
    # 只在合成因子有定义的交易日上比较，否则单因子会多出没有权重的那段
    keep_dates = set(composite["date"].unique())
    day_fac = day_fac[day_fac["date"].isin(keep_dates)]
    returns_df = returns_df[returns_df["date"].isin(keep_dates)]

    rows = []
    for col in FACTOR_COLS:
        rows.append(evaluate_one(col, day_fac[["date", "ts_code", col]], returns_df))
    rows.append(evaluate_one("composite", composite, returns_df))
    table = pd.DataFrame(rows)

    print(f"对比交易日数: {table['n_days'].iloc[0] if len(table) else 0}")
    print(f"权重样本: 滚动 {ROLL_DAYS} 个交易日 RankIC 均值，分母为 |IC| 之和")
    show = ["factor", "n_days", "rank_ic", "ls_ann", "sharpe", "max_dd"]
    print(table[show].to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    print("\n画 IC / 回测曲线 →", PLOTS_DIR)
    for col in FACTOR_COLS:
        one = day_fac[["date", "ts_code", col]]
        plot_ic(ic_by_date(one, returns_df), factor_name=col)
        plot_layer_backtest(one, returns_df, factor_name=col)
    plot_ic(ic_by_date(composite, returns_df), factor_name="composite")
    plot_layer_backtest(composite, returns_df, factor_name="composite")


if __name__ == "__main__":
    main()
