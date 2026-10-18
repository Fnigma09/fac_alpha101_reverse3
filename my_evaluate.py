"""
my_evaluate.py — 日频因子评估（最简）

输入
----
factors_df : 列含 date, ts_code，以及一个或多个因子列（如 alpha6）
returns_df : 列含 date, ts_code, fwd_ret
             fwd_ret 必须已经对齐到「信号日」：date 当天看到因子，
             fwd_ret 是持有 1 个交易日的收益（次日收盘 / 当日收盘 − 1）

输出
----
calc_ic        : dict，IC 均值、ICIR、IC 胜率
layer_backtest : dict，各组年化、多空年化
summary        : DataFrame，三个因子各一行
plot_ic        : 日度 IC 柱状图 + 累计 IC，默认写到 plots/{名}_ic.png
plot_layer_backtest : 日收益滚成累计净值图，默认写到 plots/

关键假设
----
- 每个交易日一个截面（与仓库阶段 2 一致）
- 年化按日收益复利：(连乘(1+r)) ** (252/日数) - 1
- 不扣手续费（日频满换手若硬扣 0.3%/日会虚高）

已知局限
----
- 不做涨跌停、停牌过滤；fwd_ret 用未复权收盘价。
- 全市场日频截面比月频慢；样本短时年化仍会晃。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
FACTOR_COLS = ["alpha6", "alpha12", "alpha23"]
TRADING_DAYS = 252


def _one_factor_col(factors_df: pd.DataFrame) -> str:
    cols = [c for c in factors_df.columns if c not in ("date", "ts_code")]
    if len(cols) != 1:
        raise ValueError(f"这里需要恰好一个因子列，实际是 {cols}")
    return cols[0]


def _merge(factors_df: pd.DataFrame, returns_df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """按主键 date + ts_code 合并因子和未来收益。"""
    fac = factors_df[["date", "ts_code", factor_col]].copy()
    ret = returns_df[["date", "ts_code", "fwd_ret"]].copy()
    fac["date"] = fac["date"].astype(str)
    ret["date"] = ret["date"].astype(str)
    merged = fac.merge(ret, on=["date", "ts_code"], how="inner")
    return merged.dropna(subset=[factor_col, "fwd_ret"])


def ic_by_date(factors_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
    """每个调仓日一个 Spearman RankIC。日期索引。样本不足的日子为 NaN。"""
    factor_col = _one_factor_col(factors_df)
    merged = _merge(factors_df, returns_df, factor_col)

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 20:
            return np.nan
        if g[factor_col].nunique() < 2 or g["fwd_ret"].nunique() < 2:
            return np.nan
        return g[factor_col].corr(g["fwd_ret"], method="spearman")

    ic = merged.groupby("date", sort=True).apply(_spearman, include_groups=False)
    ic.name = factor_col
    return ic


def calc_ic(factors_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """日频截面 RankIC。

    输入：factors_df 只有一个因子列；returns_df 含 fwd_ret。
    输出：ic_mean、icir（IC 均值 / IC 标准差）、ic_win_rate（IC>0 的交易日占比）、n_days。

    计算：每个 date 上，因子与 fwd_ret 做 Spearman 相关。
    用 Spearman 而不是 Pearson：先换成名次再算相关，个别极端分数不容易把当天相关拉歪。

    假设：一行日期代表一个交易日；样本少于 20 只股票的日子丢掉。
    """
    ic = ic_by_date(factors_df, returns_df).dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_win_rate": np.nan, "n_days": 0}

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan
    return {
        "ic_mean": ic_mean,
        "icir": float(icir),
        "ic_win_rate": float((ic > 0).mean()),
        "n_days": int(len(ic)),
    }


def _annualize_daily(r: pd.Series) -> float:
    """把一串日收益复利成年化。日数太少时这个数会很晃，只作对照。"""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return np.nan
    return growth ** (TRADING_DAYS / n) - 1.0


def layer_daily(
    factors_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """每个调仓日一行：G1…G5、ls。索引是 date。

    分组规则与 layer_backtest 相同；留下日收益序列，方便画图和算夏普。
    """
    factor_col = _one_factor_col(factors_df)
    merged = _merge(factors_df, returns_df, factor_col)
    labels = [f"G{i}" for i in range(1, n_groups + 1)]
    rows = []

    for date, g in merged.groupby("date", sort=True):
        # 股票太少时五组会切得很碎，这一天不用
        if len(g) < n_groups * 20:
            continue
        g = g.copy()
        # 先排名再 qcut：避免因子大量打平时切不开
        g["group"] = pd.qcut(
            g[factor_col].rank(method="first"),
            q=n_groups,
            labels=labels,
        )
        means = g.groupby("group", observed=True)["fwd_ret"].mean()
        if len(means) < n_groups:
            continue
        row = {lab: float(means[lab]) for lab in labels}
        row["date"] = date
        row["ls"] = row[labels[-1]] - row[labels[0]]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", *labels, "ls"]).set_index("date")
    return pd.DataFrame(rows).set_index("date")


def layer_backtest(
    factors_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_groups: int = 5,
) -> dict:
    """按因子分 5 组，等权持有 1 天。

    分组：每个交易日，因子从小到大排序，切成 G1…G5（G1 最低，G5 最高）。
    持有期：使用 returns_df 里已经对齐好的 1 日 fwd_ret，本函数不再移位。
    组收益：组内股票 fwd_ret 的平均（等权）。
    多空：当天 G5 − G1；不扣手续费。
    返回：各组年化、多空年化 ls_ann、有效交易日数 n_days。
    """
    labels = [f"G{i}" for i in range(1, n_groups + 1)]
    panel = layer_daily(factors_df, returns_df, n_groups=n_groups)

    if panel.empty:
        out = {f"{lab}_ann": np.nan for lab in labels}
        out["ls_ann"] = np.nan
        out["n_days"] = 0
        return out

    out = {f"{lab}_ann": _annualize_daily(panel[lab]) for lab in labels}
    out["ls_ann"] = _annualize_daily(panel["ls"])
    out["n_days"] = int(len(panel))
    return out


def plot_ic(
    ic_series: pd.Series,
    *,
    factor_name: str = "",
    plots_dir: Path | None = PLOTS_DIR,
    show: bool = False,
) -> plt.Figure | None:
    """画日度 IC 柱状图 + 累计 IC（对照仓库双子图布局）。

    输入：ic_by_date 返回的 Series（索引是调仓日）。
    输出：Figure；若 plots_dir 非空，另存为 {因子名}_ic.png。

    上图：每日一根柱，正蓝负红；橙线是全样本均值。
    下图：IC 累加，看长期是否往上爬。
    """
    ic_clean = ic_series.dropna()
    if ic_clean.empty:
        print(f"  [{factor_name or 'ic'}] 没有可用 IC，跳过画图。")
        return None

    name = factor_name or (ic_clean.name or "ic")
    cum_ic = ic_clean.cumsum()

    try:
        dates = pd.to_datetime(ic_clean.index.astype(str), format="%Y%m%d")
    except (TypeError, ValueError):
        dates = pd.to_datetime(ic_clean.index.astype(str), errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"IC Analysis  —  {name}  (daily)", fontsize=13, fontweight="bold")

    ax_bar = axes[0]
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in ic_clean.values]
    ax_bar.bar(dates, ic_clean.values, color=colors, width=1.5, alpha=0.8)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
    mean_ic = float(ic_clean.mean())
    ax_bar.axhline(mean_ic, color="#FF9800", linewidth=1.2, label=f"Mean = {mean_ic:.4f}")
    ax_bar.set_ylabel("IC")
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bar.set_title("IC Time Series", fontsize=10)

    ax_cum = axes[1]
    ax_cum.plot(dates, cum_ic.values, color="#4CAF50", linewidth=1.5)
    ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_cum.fill_between(
        dates, 0, cum_ic.values, where=cum_ic.values >= 0, alpha=0.15, color="#4CAF50"
    )
    ax_cum.fill_between(
        dates, 0, cum_ic.values, where=cum_ic.values < 0, alpha=0.15, color="#F44336"
    )
    ax_cum.set_ylabel("Cumulative IC")
    ax_cum.set_title("Cumulative IC", fontsize=10)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m")
    ax_cum.xaxis.set_major_locator(locator)
    ax_cum.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"{name}_ic.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_layer_backtest(
    factors_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    factor_name: str | None = None,
    n_groups: int = 5,
    plots_dir: Path | None = PLOTS_DIR,
    show: bool = False,
) -> plt.Figure | None:
    """画分层 + 多空累计净值曲线（日频、不扣费）。

    输入：单因子 factors_df + 已对齐的 returns_df。
    输出：matplotlib Figure；若 plots_dir 非空，另存为 {因子名}_backtest.png。

    曲线：G1…G5 实线、L-S 黑虚线。净值从 1 起，每日乘上 (1+当日收益)。
    """
    factor_col = factor_name or _one_factor_col(factors_df)
    panel = layer_daily(factors_df, returns_df, n_groups=n_groups)
    if panel.empty:
        print(f"  [{factor_col}] 没有可用交易日，跳过画图。")
        return None

    labels = [f"G{i}" for i in range(1, n_groups + 1)]
    nav = (1.0 + panel[labels]).cumprod()
    nav_ls = (1.0 + panel["ls"]).cumprod()

    try:
        dates = pd.to_datetime(panel.index.astype(str), format="%Y%m%d")
    except (TypeError, ValueError):
        dates = pd.to_datetime(panel.index.astype(str), errors="coerce")

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        f"Layered Backtest  —  {factor_col}  "
        f"({n_groups} groups, daily 1d, L-S = G{n_groups} - G1)",
        fontsize=12,
        fontweight="bold",
    )

    # 不用 RdYlGn 中间浅黄：白底上看不清 G3
    group_colors = ["#b2182b", "#ef8a62", "#2c7bb6", "#66bd63", "#1a9850"]
    if n_groups != len(group_colors):
        cmap = plt.get_cmap("tab10", n_groups)
        group_colors = [cmap(i) for i in range(n_groups)]
    for i, col in enumerate(labels):
        ax.plot(dates, nav[col].values, color=group_colors[i], linewidth=1.6, label=col)

    ax.plot(dates, nav_ls.values, color="black", linewidth=2.0, linestyle="--", label="L-S")

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Cumulative NAV")
    ax.legend(loc="upper left", fontsize=9, ncol=n_groups + 1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"{factor_col}_backtest.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig


def summary(factors_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """对 alpha6、alpha12、alpha23 各跑一遍 IC 和分层，合成一张表。

    输入：宽表因子 + 日频 fwd_ret。
    输出：一行一个因子。含 IC 三项、G1–G5 年化、多空年化。
    """
    rows = []
    for col in FACTOR_COLS:
        one = factors_df[["date", "ts_code", col]]
        ic = calc_ic(one, returns_df)
        bt = layer_backtest(one, returns_df, n_groups=5)
        row = {"factor": col, **ic, **bt}
        rows.append(row)
    return pd.DataFrame(rows)


def make_daily_panel(factors_df: pd.DataFrame, prices_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从日频价格算 1 日远期收益，并与因子表按主键对齐。

    fwd_ret = 次日收盘 / 当日收盘 − 1，记在当日这一行上。
    为什么：分层要「今天看到因子、拿到明天收益」，收益必须对齐到信号日。
    """
    px = prices_df.copy()
    if "trade_date" in px.columns and "date" not in px.columns:
        px = px.rename(columns={"trade_date": "date"})
    px["date"] = px["date"].astype(str)
    px = px.sort_values(["ts_code", "date"])

    # 按股票把下一日收盘挪到今天：缺下一交易日（停牌末段、退市）则为空
    px["close_next"] = px.groupby("ts_code", sort=False)["close"].shift(-1)
    px["fwd_ret"] = px["close_next"] / px["close"] - 1
    returns_df = px[["date", "ts_code", "fwd_ret"]]

    fac = factors_df.copy()
    fac["date"] = fac["date"].astype(str)
    # 只留两边都有的交易日，避免因子日期漂在收益表外面
    keep_dates = set(returns_df["date"]) & set(fac["date"])
    fac = fac[fac["date"].isin(keep_dates)]
    returns_df = returns_df[returns_df["date"].isin(keep_dates)]
    return fac, returns_df


def main() -> None:
    factors = pd.read_parquet(DATA_DIR / "factors_clean.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    fac, rets = make_daily_panel(factors, prices)
    table = summary(fac, rets)

    show_cols = [
        "factor",
        "n_days",
        "ic_mean",
        "icir",
        "ic_win_rate",
        "G1_ann",
        "G2_ann",
        "G3_ann",
        "G4_ann",
        "G5_ann",
        "ls_ann",
    ]
    print(table[show_cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    print("\n画 IC / 回测曲线 →", PLOTS_DIR)
    for col in FACTOR_COLS:
        one = fac[["date", "ts_code", col]]
        plot_ic(ic_by_date(one, rets), factor_name=col)
        plot_layer_backtest(one, rets, factor_name=col)

    print("\n核对提示：")
    print("- IC 均值常见讨论区间大约 0.02~0.05；样本短时可以落在区间外。")
    print("- IC 图：橙线接近表里的 ic_mean；累计 IC 长期往上通常更好。")
    print("- 分层：若因子正向，G5 年化应高于 G1；反向则相反。")
    print("- 本版不扣费；净值图只有一条 L-S 虚线。")


if __name__ == "__main__":
    main()
