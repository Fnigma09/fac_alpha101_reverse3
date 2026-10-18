"""
my_preprocess.py — 因子预处理（去极值 → 标准化 → 行业中性化）

输入
----
- factors_raw.parquet
  列：date, ts_code, alpha6, alpha12, alpha23
  （若文件不存在，可用本脚本根据 prices.parquet 先生成）
- meta.parquet（取 industry，按 date+ts_code 或 ts_code 贴上）

输出
----
- factors_clean.parquet
  列：date, ts_code, alpha6, alpha12, alpha23（已清洗）

处理顺序（每个因子列各自做）
----
1. MAD 去极值，按 date 截面，边界 = 中位数 ± 3 × 1.4826 × MAD
2. 截面 z-score
3. 截面 OLS：因子 ~ 行业哑变量，取残差（见 neutralize）

已知局限
----
- 行业来自 Tushare stock_basic.industry，非申万分类，且是名单快照，
  不是历史上每一天的真实行业 → 存在行业分类前视偏差。
- 相对仓库 FactorCleaner：这里不做市值中性化，中性化后也不再二次 z-score，
  也不把空值填成 0。
- MAD 公式与仓库一致（含 1.4826）；若某日有效样本太少或 MAD=0，该日可能几乎不变或全空。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from my_factors import calc_alpha6, calc_alpha12, calc_alpha23

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FACTOR_COLS = ["alpha6", "alpha12", "alpha23"]

# 与 FactorCleaner 一致：3 × 1.4826 × MAD
MAD_LIMITS = 3.0
MAD_SCALE = 1.4826


def winsorize_mad_cs(df: pd.DataFrame, factor_col: str, date_col: str = "date") -> pd.Series:
    """截面 MAD 去极值。

    为什么：极端值会主导当天排序和均值，把尾巴夹住后再标准化更稳。
    """

    def _clip_one_day(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if valid.empty:
            return s
        med = valid.median()
        mad = (valid - med).abs().median()
        if mad == 0 or np.isnan(mad):
            # 几乎没有离散度时没法定边界，原样返回
            return s
        delta = MAD_LIMITS * MAD_SCALE * mad
        return s.clip(lower=med - delta, upper=med + delta)

    return df.groupby(date_col, group_keys=False)[factor_col].transform(_clip_one_day)


def standardize_zscore_cs(df: pd.DataFrame, factor_col: str, date_col: str = "date") -> pd.Series:
    """截面 z-score。

    为什么：不同股票原始分数尺度不同，拉到同一天均值约 0、波动约 1，才能公平比较。
    """

    def _z_one_day(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std()
        if sigma == 0 or np.isnan(sigma):
            return s * np.nan
        return (s - mu) / sigma

    return df.groupby(date_col, group_keys=False)[factor_col].transform(_z_one_day)


def neutralize(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "industry",
    date_col: str = "date",
) -> pd.Series:
    """截面行业中性化：对每个交易日，用 industry 哑变量做 OLS，取残差。

    行业分类来源为 Tushare stock_basic 的 industry 字段，非申万分类，
    非历史时点分类，存在行业分类前视偏差。

    规则
    ----
    - industry 为 NaN 的股票：不参与该日回归，该日该股结果为 NaN。
    - 因子本身为 NaN 的股票：不参与回归，结果保持 NaN。
    - 用 numpy 组哑变量矩阵 + lstsq，减少额外依赖。

    为什么做：去掉「只是某行业整体偏高/偏低」带来的分数，留下行业解释不了的部分。
    """
    if industry_col not in df.columns:
        raise ValueError(f"缺少行业列: {industry_col}")

    out = pd.Series(np.nan, index=df.index, dtype=float)

    for _, g in df.groupby(date_col, sort=False):
        y_all = g[factor_col]
        ind_all = g[industry_col]

        # 因子有效且行业非空才进回归
        mask = y_all.notna() & ind_all.notna()
        if mask.sum() < 3:
            continue

        y = y_all.loc[mask].to_numpy(dtype=float)
        industries = ind_all.loc[mask].astype(str)

        # 哑变量：丢掉一个行业类别，避免与截距完全共线
        dummies = pd.get_dummies(industries, drop_first=True, dtype=float)
        if dummies.shape[1] == 0:
            # 当天只剩一个行业，没法做行业中性化
            continue

        # 设计矩阵：截距 + 行业哑变量
        X = np.column_stack([np.ones(len(y)), dummies.to_numpy(dtype=float)])
        # 观测要比参数多一点，否则残差不稳定
        if len(y) < X.shape[1] + 2:
            continue

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out.loc[g.index[mask]] = resid
        # industry 为 NaN 的行保持 NaN（out 默认已是 NaN）

    out.name = factor_col
    return out


def attach_industry(factors: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """把 industry 贴到因子表。

    meta 用 trade_date；factors_raw 用 date。先对齐列名再按主键合并。
    """
    m = meta.copy()
    if "trade_date" in m.columns and "date" not in m.columns:
        m = m.rename(columns={"trade_date": "date"})
    m["date"] = m["date"].astype(str)
    m["ts_code"] = m["ts_code"].astype(str)

    keep = m[["date", "ts_code", "industry"]].drop_duplicates(subset=["date", "ts_code"])
    out = factors.merge(keep, on=["date", "ts_code"], how="left")
    return out


def process_factors(factors_raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """完整清洗流水线。"""
    need = ["date", "ts_code", *FACTOR_COLS]
    missing = [c for c in need if c not in factors_raw.columns]
    if missing:
        raise ValueError(f"factors_raw 缺少列: {missing}")

    df = factors_raw[need].copy()
    df["date"] = df["date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    df = attach_industry(df, meta)

    cleaned = df[["date", "ts_code"]].copy()
    for col in FACTOR_COLS:
        # 1) 去极值：防止极端值主导截面排序与均值
        s = winsorize_mad_cs(df, col)
        tmp = df[["date"]].copy()
        tmp[col] = s
        # 2) 标准化：同一天分数拉到可比尺度
        s = standardize_zscore_cs(tmp, col)
        tmp[col] = s
        tmp["industry"] = df["industry"].values
        # 3) 行业中性化：去掉行业整体偏置
        cleaned[col] = neutralize(tmp, col)

    return cleaned


def build_factors_raw_from_prices(
    prices: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从 prices 按股票算三个因子，写成 factors_raw 长表。

    prices 列名用 trade_date / volume；因子函数要 vol，这里做映射。
    """
    p = prices.copy()
    if "trade_date" in p.columns:
        p = p.rename(columns={"trade_date": "date"})
    if "volume" in p.columns and "vol" not in p.columns:
        p = p.rename(columns={"volume": "vol"})

    p["date"] = p["date"].astype(str)
    if start_date is not None:
        p = p[p["date"] >= start_date]
    if end_date is not None:
        p = p[p["date"] <= end_date]

    frames = []
    groups = list(p.groupby("ts_code", sort=False))
    n = len(groups)
    for i, (code, g) in enumerate(groups, start=1):
        g = g.sort_values("date")
        one = pd.DataFrame(
            {
                "date": g["date"].values,
                "ts_code": code,
                "alpha6": calc_alpha6(g).to_numpy(),
                "alpha12": calc_alpha12(g).to_numpy(),
                "alpha23": calc_alpha23(g).to_numpy(),
            }
        )
        frames.append(one)
        if i % 500 == 0 or i == n:
            print(f"  因子进度: {i}/{n}")

    return pd.concat(frames, ignore_index=True)


def _verify(raw: pd.DataFrame, clean: pd.DataFrame, meta: pd.DataFrame) -> None:
    """打印几项核对数字，方便肉眼看清洗有没有生效。"""
    col = "alpha12"  # 原始尺度变化大，适合看去极值
    day = (
        clean.dropna(subset=[col])
        .groupby("date")[col]
        .count()
        .idxmax()
    )

    day_raw = raw.loc[raw["date"] == day, ["date", "ts_code", col]].copy()
    r = day_raw[col]

    print("\n===== 验证信息 =====")
    print(f"样本日: {day}  因子: {col}")
    print(f"去极值前 max/min : {r.max():.4f} / {r.min():.4f}")

    only_w = winsorize_mad_cs(day_raw, col)
    print(f"仅去极值后 max/min: {only_w.max():.4f} / {only_w.min():.4f}")
    print("  → 极值应被夹住（max 变小或 min 变大）")

    tmp = day_raw[["date", "ts_code"]].copy()
    tmp[col] = only_w
    only_z = standardize_zscore_cs(tmp, col)
    print(f"仅标准化后 均值/标准差: {only_z.mean():.4f} / {only_z.std():.4f}")
    print("  → 均值应接近 0，标准差接近 1")

    m = meta.rename(columns={"trade_date": "date"}) if "trade_date" in meta.columns else meta
    m = m[["date", "ts_code", "industry"]].copy()
    m["date"] = m["date"].astype(str)
    m_day = m[m["date"] == day]

    before = tmp.copy()
    before[col] = only_z
    before = before.merge(m_day, on=["date", "ts_code"], how="left")

    after = clean.loc[clean["date"] == day, ["date", "ts_code", col]].merge(
        m_day, on=["date", "ts_code"], how="left"
    )

    disp_before = before.dropna(subset=[col, "industry"]).groupby("industry")[col].mean().std()
    disp_after = after.dropna(subset=[col, "industry"]).groupby("industry")[col].mean().std()
    print(f"行业均值的标准差(中性化前/后): {disp_before:.4f} / {disp_after:.4f}")
    print("  → 中性化后应更小（行业之间平均分差变小）")
    print(f"factors_clean shape: {clean.shape}  cols: {clean.columns.tolist()}")
    print("====================\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "factors_raw.parquet"
    clean_path = DATA_DIR / "factors_clean.parquet"
    prices_path = DATA_DIR / "prices.parquet"
    meta_path = DATA_DIR / "meta.parquet"

    meta = pd.read_parquet(meta_path)

    if not raw_path.exists():
        print("未找到 factors_raw.parquet，从 prices 计算三个因子（可能较慢）...")
        prices = pd.read_parquet(prices_path)
        # 与 my_data_prep 默认窗口一致；可按需改
        raw = build_factors_raw_from_prices(prices, start_date="20190101", end_date="20231231")
        raw.to_parquet(raw_path, index=False)
        print(f"已写入 {raw_path}  shape={raw.shape}")
    else:
        raw = pd.read_parquet(raw_path)
        raw["date"] = raw["date"].astype(str)
        print(f"读取 {raw_path}  shape={raw.shape}")

    print("清洗中...")
    clean = process_factors(raw, meta)
    clean.to_parquet(clean_path, index=False)
    print(f"已写入 {clean_path}")

    _verify(raw, clean, meta)


if __name__ == "__main__":
    main()
