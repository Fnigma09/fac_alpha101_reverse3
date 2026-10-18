# 代码地图（调用关系）

看代码时先看这一页，再进具体函数。  
**调用树**：从上往下表示「谁调用谁」。上面是入口，下面是被叫去干活的函数。

名词：

- **入口**：你在终端直接运行的那个文件，一般是 `main()`。
- **调用**：A 函数里写了 `B(...)`，就是 A 调用 B。

三个脚本**没有互相自动串起来**。你要按顺序自己跑：先数据，再预处理（预处理内部会去调因子函数）。

---

## 现在有哪些文件

| 文件 | 你什么时候跑 | 读入 | 写出 |
|------|----------------|------|------|
| `my_data_prep.py` | 第一步 | `tushare_data/...` 里的 daily、adj_factor、daily_basic、stock_basic | `data/prices.parquet`、`data/meta.parquet` |
| `my_factors.py` | 一般不单独跑 | 一只股票的日线（列含 `vol`） | 三个因子的 Series |
| `my_preprocess.py` | 第二步 | `prices`（没有 raw 时）、`meta` | `data/factors_raw.parquet`、`data/factors_clean.parquet` |
| `my_evaluate.py` | 第三步 | `factors_clean`、`prices` | 打印表；`plots/*_ic.png`、`*_backtest.png` |
| `my_combine.py` | 第四步 | 同上 | 对比表 + composite 图 |

`my_preprocess.py` 会 `import` `my_factors.py` 里的三个计算函数。

---

## 1. `my_data_prep.py`

```text
main()
├── load_daily()            读日线，vol 改名为 volume
│   └── _as_yyyymmdd()      日期收成 8 位字符串
├── load_adj_factor()       读复权因子
│   └── _as_yyyymmdd()
├── load_daily_basic()      读每日资料（换手、市值等）
│   └── _as_yyyymmdd()
├── load_industry_map()     股票代码 → 行业名
├── filter_date_range()     按 START_DATE / END_DATE 裁切（调用 3 次）
├── build_prices()          拼 prices（含 volume、adj_factor、tradable）
└── build_meta()            拼 meta（含 industry + 15 个资料列）
```

建议阅读顺序：`main` → `build_prices` → `build_meta`。其它函数是为主流程准备表。

---

## 2. `my_factors.py`

没有 `main`。三个函数彼此不调用，都只检查列名。

```text
calc_alpha6(df)   → _require_sorted_columns()
calc_alpha12(df)  → _require_sorted_columns()
calc_alpha23(df)  → _require_sorted_columns()
```

输入是**一只股票**、已按日期排序的表。列名用 `vol`，不是 `volume`。

---

## 3. `my_preprocess.py`

```text
main()
├── （若没有 factors_raw）build_factors_raw_from_prices()
│   └── 每只股票调用 calc_alpha6 / calc_alpha12 / calc_alpha23
├── process_factors()
│   ├── attach_industry()          从 meta 贴上 industry
│   └── 对 alpha6、alpha12、alpha23 各做一遍：
│       ├── winsorize_mad_cs()     截面去极值
│       ├── standardize_zscore_cs() 截面 z-score
│       └── neutralize()           截面行业回归，取残差
└── _verify()                      只打印核对数字，不改文件
    ├── winsorize_mad_cs()
    └── standardize_zscore_cs()
```

建议阅读顺序：`main` → `process_factors` → `neutralize`。因子公式仍回 `my_factors.py` 看。

---

## 4. `my_evaluate.py`

不改因子，只评估。**日频、持有 1 天、不扣费**（与仓库阶段 2 设定一致）。

```text
main()
├── make_daily_panel()       日频价格 → 1 日 fwd_ret，与因子按主键对齐
├── summary()                对 alpha6 / alpha12 / alpha23 各跑一遍
│   ├── calc_ic()            每日 Spearman，汇总均值、ICIR、胜率
│   └── layer_backtest()     每日分 5 组等权，年化（×252）
├── plot_ic()                日度 IC 柱 + 累计 IC → plots/{因子}_ic.png
│   └── ic_by_date()
└── plot_layer_backtest()     G1…G5 + L-S → plots/{因子}_backtest.png
    └── layer_daily()
```

读入：`factors_clean.parquet`、`prices.parquet`。  
写出：打印表；图写到 `plots/{因子}_ic.png`、`plots/{因子}_backtest.png`。

建议阅读顺序：`main` → `calc_ic` → `plot_ic` → `layer_backtest` → `plot_layer_backtest`。

---

## 5. `my_combine.py`

在日频因子上合成，再调用 `my_evaluate.py` 里的同一套 IC 和分层。

```text
main()
├── make_daily_panel()         来自 my_evaluate
├── ic_by_date()               三个因子各一条日度 RankIC
├── rolling_ic_weights()       过去 252 个交易日 IC 均值 / |IC| 之和（不含当天）
├── build_composite()          加权得到 composite
├── evaluate_one()             单因子和 composite 各一次
│   ├── calc_ic()
│   ├── layer_backtest()
│   └── layer_daily()          用来算夏普、最大回撤
├── plot_ic()                 三个单因子 + composite 各一张 IC 图
└── plot_layer_backtest()      三个单因子 + composite 各一张净值图
```

读入：`factors_clean.parquet`、`prices.parquet`。  
写出：打印对比表；图写到 `plots/`（含 `composite_ic.png`、`composite_backtest.png`）。

---

## 数据怎么流

```text
Tushare 原始 parquet
        │
        ▼
my_data_prep.py
        │
        ├── prices.parquet ──┐
        └── meta.parquet ────┤
                             ▼
                    my_preprocess.py
                     （内部用 my_factors.py）
                             │
                             ├── factors_raw.parquet
                             └── factors_clean.parquet
                                      │
                                      ▼
                             my_evaluate.py  （再读 prices，算 1 日 fwd_ret）
                             打印 IC / 分组年化
                             → plots/*_ic.png、*_backtest.png
                                      │
                                      ▼
                             my_combine.py
                             滚动 252 日 IC 加权 → composite
                             打印对比表 → plots/（含 composite_ic / backtest）
```

主键（能标出「哪一天、哪只股票」的列）：

- prices / meta：`trade_date` + `ts_code`
- factors_raw / factors_clean：`date` + `ts_code`（同一含义，列名不同）

---

## 以后加了新脚本

在本文件追加一节：入口、调用树、读入写出。不要只改代码不改这一页。
