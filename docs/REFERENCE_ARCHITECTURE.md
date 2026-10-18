# Parsnip77 参考架构（速查表）

> **学习请读**：[docs/lessons/](./lessons/README.md)（白话讲解、术语解释）  
> 本文件只做对照速查，术语较多；新手优先 lessons。  
> 来源仓库：`Parsnip77/`；不含可运行业务代码。

---

## 一、三阶段流程与 I/O

### 阶段 1：数据层 + 因子计算 + 预处理

| 步骤 | 输入 | 输出 |
|------|------|------|
| 下载 | Tushare（沪深300成分、日线、基本面、复权、ST） | `data/stock_data.db`（SQLite） |
| 加载 | SQLite | `df_price / df_basic / df_mv / df_industry / df_adj / df_st` |
| 可交易掩码 | 价格 + ST + 上市日 | 布尔 Series：`tradable` |
| 原始因子 | 前复权价格宽表 | `factors_raw`（MultiIndex date×code） |
| 清洗因子 | 原始因子 + 市值 + 行业 | `factors_clean`（不可交易格为 NaN） |
| 导出 | 上述表 | 四张 Parquet：`prices / meta / factors_raw / factors_clean` |

入口脚本：`data_preparation_main.py`

### 阶段 2：IC 评估 + 分层回测 + 线性合成

| 步骤 | 输入 | 输出 |
|------|------|------|
| 标签 | `prices.parquet` | `forward_return`（默认 1 日） |
| 单因子 IC | `factors_clean` + 标签 | 日频 Spearman IC 序列、IC Mean / ICIR |
| 筛选 | IC 阈值 | 有效因子列表（\|IC mean\|>0.02 且 \|ICIR\|>0.30） |
| 分层回测 | 单因子 + 标签 | G1–G5 与 L-S 绩效表、净值图 |
| 合成 | 有效因子滚动 OLS（正交化） | `synthetic_factor` 的 IC + 回测 |
| 报告 | 全程打印缓冲 | `result.txt` + `plots/*.png` |

入口脚本：`analyze_main.py`  
注意：README 写了“扣费净收益回测”，**当前 `analyze_main.py` 实际未扣手续费**。

### 阶段 3：LightGBM 合成（精简复写可后置）

| 步骤 | 输入 | 输出 |
|------|------|------|
| 滚动训练 | 清洗因子作特征 | ML Alpha 分数 |
| 评估 | ML Alpha + 标签 | IC、分层回测、特征重要性、SHAP |
| 报告 | — | `result_ml.txt` |

入口脚本：`ml_analyze_main.py`

---

## 二、配置项落点（注意：不都在 config）

仓库里只有 `src/config_template.py`（需复制为 `config.py`）。

| 类别 | 实际位置 | 值 |
|------|----------|-----|
| 数据路径 | `config.DB_PATH` | `../data/stock_data.db`（相对 src） |
| Parquet 目录 | `data_preparation_main` / `analyze_main` | `./data/` |
| 日期范围 | `START_DATE` / `END_DATE` | `20190101` ~ `20231231` |
| 股票池 | `UNIVERSE_INDEX` | `000300.SH` |
| 因子列表 | **不在 config**；由 `Alpha101` 中所有 `alpha*` 方法决定 | 见下节 |
| 分组数 | `LayeredBacktester(num_groups=5)` | 5 |
| 持有期 / 调仓 | `FORWARD_DAYS=1`（日频换仓） | 1 个交易日 |
| 手续费 | **未在阶段 2 实现** | — |
| IC 筛选阈值 | `analyze_main` | \|IC mean\|>0.02，\|ICIR\|>0.30 |

已实现因子名：`alpha001, alpha002, alpha003, alpha006, alpha012, alpha038, alpha040, alpha041, alpha042, alpha054, alpha072, alpha088, alpha094, alpha098, alpha101, alpha_5_day_reversal`  
（`alpha021` 源码中已注释禁用；**无 Alpha#23**。）

---

## 三、Parquet 主键约定

四张表共享逻辑主键：`(trade_date, ts_code)`，平铺列、无 MultiIndex 写入文件。

- `trade_date`：字符串 `YYYYMMDD`（与 Tushare / SQLite 一致）
- `ts_code`：如 `000001.SZ`

### prices.parquet

列：`trade_date, ts_code, open, high, low, close, volume, adj_factor, tradable`  
（`vol` 导出时改名为 `volume`；价格为库内原始未复权价。）

### meta.parquet

列：`trade_date, ts_code, industry` + 15 个基本面字段  
（turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv）

---

## 四、FactorCleaner 流水线（截面）

对每个 alpha、每个交易日截面：

1. ±inf → NaN  
2. MAD 去极值：Median ± 3 × 1.4826 × MAD  
3. 截面 Z-score  
4. 对 log(total_mv) + 行业 dummy OLS，取残差（市值+行业中性化）  
5. 残差再做截面 Z-score  
6. 剩余 NaN → 0；随后主脚本把不可交易格改回 NaN  

---

## 五、评估约定（阶段 2）

- IC：每日截面 **Spearman**（非 Pearson）；截面有效样本 ≥5  
- 分层：5 组，因子截面 qcut；组内等权；收益 = 组内 `forward_return` 均值  
- 多空：G5 − G1（高因子组减低因子组）  
- 手续费：阶段 2 代码路径中未扣除  

---

## 六、复写建议范围（精简版）

建议第一期只覆盖：

1. 数据加载（可用已有 Tushare 缓存，不必复刻 SQLite）  
2. 2–3 个简单因子（如 #6、#12）  
3. FactorCleaner 精简管线  
4. IC + 5 层回测  

LightGBM / 滚动 OLS 合成可放第二期。
