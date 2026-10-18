"""
my_data_prep.py — 精简版数据准备（第一段重写）

把本地已下载的 Tushare 表格，整理成与 Parsnip77 下游脚本同结构的两张表：
prices.parquet、meta.parquet。

输入（默认指向 tushare_data/stock_data_v1_.../）
----
- daily.parquet       : 日线，至少含 ts_code, trade_date, open, high, low, close, vol
- adj_factor.parquet  : 复权因子（与日线按 ts_code+trade_date 对齐）
- daily_basic.parquet : 每日估值/换手/市值等
- stock_basic.parquet : 股票列表，用其中的 industry 当行业名

输出（默认写到 factor_lite/data/）
----
- prices.parquet : 平铺列，无行索引；钥匙列为 trade_date, ts_code
- meta.parquet   : 同上钥匙 + industry + 15 个基本面列

关键假设
----
- trade_date 最终统一成 8 位字符串，例如 "20180102"
- prices 里的价是未复权原始价；复权留给以后算因子时再用 adj_factor
- 行业来自 stock_basic 的「当前这份名单」，同一股票所有历史日期共用同一行业

已知局限
----
- 行业是下载当日的快照，不是历史上每一天真实所属行业（会有少量错位）
- tradable 只用「有成交且收盘价有效」的简单规则，尚未做涨跌停 / ST / 新股过滤
- 未限制为沪深300成分；默认导出本地 daily 日期窗口内的全部 A 股日线
- amount 不写入 prices（与仓库导出一致）；需要时可再从 daily 读取
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径与日期窗口：改这里即可，不必改下面函数
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# 本地 Tushare 落盘目录（你已有的那份）
RAW_DIR = ROOT.parent / "tushare_data" / "stock_data_v1_20180101_20241231"
# 整理后的输出目录
OUT_DIR = ROOT / "data"

# 只导出这段日期，减轻内存；想要全量可改成 None
START_DATE = "20190101"
END_DATE = "20231231"

# 仓库 meta 里那 15 个每日资料列（名字必须一致，方便以后对接清洗模块）
META_BASIC_COLS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]


def _as_yyyymmdd(series: pd.Series) -> pd.Series:
    """把日期列统一成 8 位字符串。

    为什么：仓库全程用 '20190102' 这种格式做合并键；
    若混进 datetime 或 '2019-01-02'，后面 join 会对不上。
    """
    s = series.astype(str).str.replace(r"\D", "", regex=True).str[:8]
    return s


def load_daily(path: Path) -> pd.DataFrame:
    """读取日线，并只留下 prices 需要的价量列。"""
    df = pd.read_parquet(path)
    need = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"daily 缺少列: {missing}")

    out = df[need].copy()
    out["trade_date"] = _as_yyyymmdd(out["trade_date"])

    # 仓库 prices 列名是 volume，不是 Tushare 的 vol
    out = out.rename(columns={"vol": "volume"})
    return out


def load_adj_factor(path: Path) -> pd.DataFrame:
    """读取复权因子。对应仓库 prices 里的 adj_factor 列。"""
    df = pd.read_parquet(path)
    need = ["ts_code", "trade_date", "adj_factor"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"adj_factor 缺少列: {missing}")

    out = df[need].copy()
    out["trade_date"] = _as_yyyymmdd(out["trade_date"])
    return out


def load_daily_basic(path: Path) -> pd.DataFrame:
    """读取每日资料，作为 meta 的数值部分。"""
    df = pd.read_parquet(path)
    need = ["ts_code", "trade_date"] + META_BASIC_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"daily_basic 缺少列: {missing}")

    out = df[need].copy()
    out["trade_date"] = _as_yyyymmdd(out["trade_date"])
    return out


def load_industry_map(path: Path) -> pd.Series:
    """从 stock_basic 得到 股票代码 -> 行业名。

    对应仓库 meta 的 industry 列。
    局限：这是名单快照，不是历史时点行业。
    """
    df = pd.read_parquet(path, columns=["ts_code", "industry"])
    # 同一代码若有多行，保留最后一行即可
    s = df.drop_duplicates(subset=["ts_code"], keep="last").set_index("ts_code")["industry"]
    return s


def filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """按字符串日期过滤。为什么用字符串比较：格式已统一为 YYYYMMDD，字典序=时间序。"""
    out = df
    if start is not None:
        out = out[out["trade_date"] >= start]
    if end is not None:
        out = out[out["trade_date"] <= end]
    return out


def build_prices(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """拼出与 Parsnip77 同结构的 prices 表。"""
    # 左连接：以有行情的行为准；缺复权因子时 adj_factor 为空，后面算因子时再处理
    prices = daily.merge(adj, on=["ts_code", "trade_date"], how="left")

    # 简单可交易标记（仓库还有涨跌停/ST/新股等，这里先留接口）
    # True = 当天至少发生过成交，且收盘价有效
    prices["tradable"] = (prices["volume"] > 0) & (prices["close"].notna()) & (prices["close"] > 0)

    # 列顺序与仓库导出保持一致，方便肉眼对照
    cols = [
        "trade_date",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "volume",  # 对应仓库 prices.volume（由 Tushare vol 改名）
        "adj_factor",  # 对应仓库 prices.adj_factor
        "tradable",  # 对应仓库 prices.tradable（规则更简单）
    ]
    return prices[cols]


def build_meta(basic: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """拼出与 Parsnip77 同结构的 meta 表。"""
    meta = basic.copy()
    # 把静态行业贴到每一天每一只股票上 → 仓库 meta.industry
    meta["industry"] = meta["ts_code"].map(industry)
    # 没有行业名时标成「其他」，避免后面做行业处理时全是空
    meta["industry"] = meta["industry"].fillna("其他")

    cols = ["trade_date", "ts_code", "industry"] + META_BASIC_COLS
    return meta[cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取本地 Tushare 文件...")
    daily = load_daily(RAW_DIR / "daily.parquet")
    adj = load_adj_factor(RAW_DIR / "adj_factor.parquet")
    basic = load_daily_basic(RAW_DIR / "daily_basic.parquet")
    industry = load_industry_map(RAW_DIR / "stock_basic.parquet")

    print("按日期窗口过滤...")
    daily = filter_date_range(daily, START_DATE, END_DATE)
    adj = filter_date_range(adj, START_DATE, END_DATE)
    basic = filter_date_range(basic, START_DATE, END_DATE)

    print("组装 prices / meta...")
    prices = build_prices(daily, adj)
    meta = build_meta(basic, industry)

    prices_path = OUT_DIR / "prices.parquet"
    meta_path = OUT_DIR / "meta.parquet"
    # index=False：与仓库一致，钥匙留在列里，不写行索引
    prices.to_parquet(prices_path, index=False)
    meta.to_parquet(meta_path, index=False)

    # ----- 跑完立刻打印核对信息（你应重点看这几行）-----
    print("\n===== 验证信息 =====")
    print(f"prices -> {prices_path}")
    print(f"  shape     : {prices.shape}")
    print(f"  columns   : {prices.columns.tolist()}")
    print(f"  date range: {prices['trade_date'].min()} -> {prices['trade_date'].max()}")
    print(f"  tradable% : {prices['tradable'].mean() * 100:.2f}%")
    print(f"  adj 缺失% : {prices['adj_factor'].isna().mean() * 100:.2f}%")

    print(f"\nmeta   -> {meta_path}")
    print(f"  shape     : {meta.shape}")
    print(f"  columns   : {meta.columns.tolist()}")
    print(f"  date range: {meta['trade_date'].min()} -> {meta['trade_date'].max()}")
    print(f"  industry 样例: {meta['industry'].dropna().unique()[:8].tolist()}")
    print(f"  industry 空值%: {meta['industry'].isna().mean() * 100:.2f}%")
    print("====================\n")


if __name__ == "__main__":
    main()
