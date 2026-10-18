# 第 6 课：第一段重写——数据准备

> 对照：`Parsnip77/data_preparation_main.py` 导出部分  
> 我们写的脚本：`factor_lite/my_data_prep.py`  
> 本地数据：`tushare_data/stock_data_v1_20180101_20241231/`

---

## 第一步：仓库期望什么格式？

两张表都是**普通表格**存进文件：没有把日期设成行标签。  
对齐用的钥匙是两列：`trade_date` + `ts_code`。

日期一律用 **8 位字符串**，例如 `"20180102"`（不是 `2018-01-02`）。

### prices.parquet

| 列名 | 意思 |
|------|------|
| `trade_date` | 交易日 |
| `ts_code` | 股票代码 |
| `open, high, low, close` | 开高低收（未复权原始价） |
| `volume` | 成交量（注意：仓库不用 `vol` 这个名字） |
| `adj_factor` | 复权因子（把分红送股前后的价格接到同一尺子上用） |
| `tradable` | 这一天能不能当真去买（True/False） |

**索引**：文件里 `index=False`，没有额外行索引。

### meta.parquet

| 列名 | 意思 |
|------|------|
| `trade_date`, `ts_code` | 同一把钥匙 |
| `industry` | 行业名字 |
| 再加约 15 个每日资料列 | 换手、估值、股本、市值等（见下表） |

15 个资料列（与仓库一致）：  
`turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv`

**索引**：同样是平铺列，无行索引。

### Tushare 字段分别进哪个文件？

| 你本地的源文件 | 主要字段 | 进哪张输出 |
|----------------|----------|------------|
| `daily.parquet` | open/high/low/close/vol/amount… | → `prices`（价量；`vol` 改名 `volume`） |
| `adj_factor.parquet` | adj_factor | → `prices` |
| （本脚本简化规则） | tradable | → `prices`（先用简单规则，不是仓库全套过滤） |
| `daily_basic.parquet` | 换手、PE、市值等 | → `meta` |
| `stock_basic.parquet` | ts_code → industry | → `meta` 的 industry（贴到每一天） |

`amount` 仓库的 prices 里**不导出**；以后算均价（VWAP）时可以从 daily 再取。

---

## 第二步：我们脚本做了什么 / 没做什么

做了：读本地四张表 → 对齐列名 → 写出与仓库同结构的 `prices.parquet`、`meta.parquet`。

故意简化：

- **行业**用的是 `stock_basic` 里「现在这份名单」的行业，不是历史上每一天的行业。  
- **tradable** 只用「成交量>0 且收盘价>0」，还没做涨跌停 / ST / 新股过滤。

---

## 怎么验证输出？

在项目里跑完脚本后：

```text
1) 打印 shape：行数应接近 daily 与 daily_basic 对齐后的行数（百万级，若你截了日期则更小）
2) 列名必须完全对得上上面两张表
3) trade_date 应是字符串、长度为 8
4) prices 里应有 volume（没有 vol）、有 adj_factor、有 tradable
5) meta 里 industry 不应几乎全空；抽几只股票看行业名是否合理
6) 日期范围：min/max 应落在你设定的 START/END 内
```

合理范围举例：全市场 2018–2024 日线大约几百万行；若你只留沪深300，大约几十万行量级。
