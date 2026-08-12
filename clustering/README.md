# Credit Card Customer Segmentation

Unsupervised segmentation of 8,950 active credit-card holders using PCA and four
clustering algorithms. The dataset has no target column — the goal is to discover natural
behavioural groups that marketing and risk teams can act on.

**Headline result:** three segments that split almost perfectly by *how* the card is used.
92% of segment 0 never made a single purchase; 97% of segment 1 never took a cash advance;
segment 2 does both heavily and carries the largest balances.

**Last run:** 12 August 2026, Google Colab, `RANDOM_STATE = 42`. All 45 code cells executed
without error, 25 figures.

---

## Contents

| File | Description |
|---|---|
| `Credit_Card_Clustering.ipynb` | The full analysis — EDA, preprocessing, PCA, four clustering algorithms, profiling and export. 74 cells, runs top to bottom. |
| `CC GENERAL.csv` | Source data — 8,950 customers × 18 columns, ~12 months of behaviour. |
| `customer_segments.csv` | Every customer with its final label, the labels from all four algorithms, and its 8 principal-component scores. |
| `cluster_profile_summary.csv` | Mean of every feature per cluster, in original units. |
| `model_comparison.csv` | Validation metrics for all four algorithms. |
| `clustering_model.joblib` | Fitted scaler + PCA + K-Means + GMM, with the training caps and medians, for scoring new customers. |

## Running it

Written for Google Colab; no setup needed beyond the data file.

1. Open `Credit_Card_Clustering.ipynb` in Colab.
2. Put `CC GENERAL.csv` beside the notebook, in `MyDrive`, or let the loader cell prompt you to upload it.
3. Runtime → Run all. Roughly 5–10 minutes; the DBSCAN grid and the silhouette scans dominate.

Dependencies — `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib` — are preinstalled in Colab.

---

## Method

**1. Data quality.** 8,950 rows, no duplicate rows or customer IDs. Missing values in
`MINIMUM_PAYMENTS` (313 rows, 3.5%) and `CREDIT_LIMIT` (1 row) filled with the median —
312.34 and 3,000.00 — because both columns are far too skewed for the mean to be safe.

**2. Feature engineering.** Six ratios describing behaviour independently of how long the
customer has held the card: `MONTHLY_AVG_PURCHASE`, `MONTHLY_CASH_ADVANCE`, `LIMIT_USAGE`,
`PAYMENT_MINPAY_RATIO`, `AVG_PURCHASE_PER_TRX`, `AVG_CASH_ADV_PER_TRX`. 17 original + 6
engineered = **23 features**.

**3. Skew and outliers.** Every monetary column is severely right-skewed; without treatment
a handful of extreme spenders would dictate the centroids. Values above each feature's 99th
percentile were clipped (17 features affected, ~90 rows each), then `log1p` applied to the
18 non-negative skewed columns.

| Feature | Skew before | Skew after |
|---|---|---|
| `PAYMENT_MINPAY_RATIO` | 43.00 | 0.78 |
| `MINIMUM_PAYMENTS` | 13.85 | 0.14 |
| `AVG_PURCHASE_PER_TRX` | 11.50 | −0.72 |
| `ONEOFF_PURCHASES` | 10.05 | 0.18 |
| `PURCHASES` | 8.14 | −0.78 |

**4. Scaling.** `StandardScaler`, mandatory here: `BALANCE` reaches five figures while the
frequency columns live in [0, 1], and every algorithm used is distance-based.

**5. PCA.** The raw features are heavily redundant — `PURCHASES`↔`ONEOFF_PURCHASES` = 0.92,
`PURCHASES_FREQUENCY`↔`PURCHASES_INSTALLMENTS_FREQUENCY` = 0.86, `CASH_ADVANCE`↔`CASH_ADVANCE_TRX` = 0.80.
**23 features → 8 components retaining 90.36% of variance.**

| Component | Variance | Interpretation |
|---|---|---|
| PC1 | 37.3% | Purchasing vs cash advance — `PURCHASES` +0.29 against `CASH_ADVANCE` −0.28. The dominant axis, and the one the final segments follow. |
| PC2 | 19.8% | Overall account size — balance, minimum payments, payments |
| PC3 | 9.4% | Repayment health — `PAYMENT_MINPAY_RATIO` +0.53 against `LIMIT_USAGE` −0.35 |
| PC4 | 7.4% | Instalment buying vs one-off buying |

---

## Results

### Choosing k

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 125,777 | **0.307** | 1.353 | **4,285** |
| **3** | **102,147** | **0.291** | **1.336** | **3,673** |
| 4 | 88,331 | 0.253 | 1.509 | 3,297 |
| 5 | 79,458 | 0.257 | 1.423 | 2,999 |
| 6 | 72,708 | 0.262 | 1.346 | 2,787 |
| 7 | 68,515 | 0.242 | 1.500 | 2,556 |
| 8 | 64,839 | 0.236 | 1.539 | 2,387 |
| 9 | 61,976 | 0.204 | 1.579 | 2,237 |
| 10 | 59,224 | 0.205 | 1.510 | 2,126 |

Silhouette peaks at k=2 (0.3074), which is the usual behaviour on data like this — a 2-way
split is trivially well separated but amounts to "cash users vs card users" and is too
coarse to act on. The notebook therefore imposes `MIN_BUSINESS_K = 3` and takes the best
silhouette above that floor, giving **k=3 at 0.2912** — a cost of only 0.016.

The floor does not fight the metrics: **Davies-Bouldin's own optimum is k=3** (1.336),
reached independently of any constraint. The two criteria agree.

### Model comparison

| Model | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise |
|---|---|---|---|---|---|
| **K-Means (k=3)** | 3 | **0.2912** | **1.3355** | 3,672.6 | 0% |
| Agglomerative, Ward (k=2) | 2 | 0.2817 | 1.4871 | 3,689.4 | 0% |
| DBSCAN (eps=1.85, min_samples=16) | 2 | 0.2727 | 0.7752 | 51.4 | 6.0% |
| GMM (k=5) | 5 | 0.1601 | 2.1020 | 2,197.7 | 0% |

**K-Means was selected** — best silhouette and best Davies-Bouldin, every customer assigned,
and centroids that are straightforward to explain to a non-technical audience. Agglomerative
edges it on Calinski-Harabasz (3,689 vs 3,673, a 0.5% difference), but that metric rewards
fewer clusters, and the comparison is k=2 against k=3.

Two results in this table are traps:

* **DBSCAN's Davies-Bouldin of 0.7752 is the best number in the table and is meaningless.**
  Its partition is degenerate — 8,392 customers in one cluster, **19** in the other, 539
  marked noise. Calinski-Harabasz of 51.4 against K-Means' 3,673 exposes it, as does an ARI
  of 0.014 against K-Means. The parameter grid shows this is not a tuning failure: every eps
  tested except 1.85 collapses to a single cluster. In PCA space these customers form one
  continuous mass with no density gaps, so DBSCAN has nothing to find.
* **GMM's BIC never bottoms out** — it falls monotonically from 214,278 at k=2 to 121,669 at
  k=10 and would keep going. Taking the minimum would just return the edge of whatever range
  was scanned, so the notebook falls back to the knee of the BIC curve at k=5. That the curve
  has no minimum is itself the finding: this data is not a mixture of a few Gaussians.

### Cross-algorithm agreement

| Comparison | ARI | NMI |
|---|---|---|
| K-Means vs Agglomerative | 0.704 | 0.671 |
| K-Means vs GMM | 0.568 | 0.654 |
| K-Means vs DBSCAN | 0.014 | 0.012 |

The agreement with Ward is structural rather than coincidental. Cross-tabulating the two
shows the hierarchical k=2 solution is almost exactly K-Means' segments **0 + 2 merged**:

| | Ward cluster 0 | Ward cluster 1 |
|---|---|---|
| K-Means 0 | 2,136 | 83 |
| K-Means 1 | 110 | 4,634 |
| K-Means 2 | 1,951 | 36 |

Ward independently finds the same top-level division — anyone who touches cash advances
versus anyone who only makes purchases — and K-Means k=3 refines the cash-advance side into
two groups. This nesting is good evidence the k=3 structure is real and not an artifact of
forcing a third cluster.

### The three segments

| | **0 · Cash-Advance Only** | **1 · Transactors** | **2 · Dual-Use, High Balance** |
|---|---|---|---|
| **Customers** | 2,219 (24.8%) | 4,744 (53.0%) | 1,987 (22.2%) |
| Avg balance | 2,093 | 773 | **2,864** |
| Avg purchases | **4** | 1,343 | 1,307 |
| Avg cash advance | 1,929 | **5** | **2,244** |
| Purchase frequency | 0.01 | 0.66 | 0.61 |
| Cash-advance frequency | 0.27 | 0.00 | 0.30 |
| Purchase transactions | 0.1 | 19.4 | 19.7 |
| Credit limit | 3,949 | 4,399 | **5,330** |
| Payments | 1,613 | 1,434 | **2,581** |
| Limit usage (balance / limit) | 0.57 | **0.22** | 0.58 |
| Pays balance in full | 4.4% | **25.0%** | 4.7% |
| Payments / min payment | 9.59 | 10.91 | **4.05** |
| **% with zero purchases** | **92.1%** | 0.0% | 0.0% |
| **% with zero cash advances** | 0.2% | **97.4%** | 0.1% |
| Share of all purchase volume | 0.1% | **71.0%** | 28.9% |
| Share of all cash-advance volume | 48.9% | 0.2% | **50.9%** |

The bottom four rows are the striking part. These are not groups that merely lean one way —
they are close to mutually exclusive behaviours. Segment 0 contributes **0.1%** of the bank's
purchase volume; segment 1 contributes **0.2%** of its cash-advance volume.

**Segment 0 — Cash-Advance Only (24.8%).** The card is an ATM. 92% never made a single
purchase all year, and average total purchases are 4 currency units against 1,929 in cash
advances taken 466 at a time. They run at 57% limit usage, only 4.4% ever clear the balance,
and they hold the lowest credit limits (3,949) — the bank has already priced some of this
risk in.
*Actions:* this is the credit-risk concentration, so lead with risk monitoring and
early-warning flags. Cash advances are the most expensive way for them to borrow, so
consolidation or fixed-term instalment loans serve the customer and cut the bank's exposure.
Do not spend acquisition-style purchase incentives here; nothing suggests they will convert
to card spending.

**Segment 1 — Transactors (53.0%).** The healthy core. 97.4% never touched a cash advance,
they buy 19 times a year across 1,343 in purchases, keep balances at 22% of limit, and are
**more than five times as likely** to pay in full as either other group (25.0% against 4.4%
and 4.7%). They generate 71% of all purchase volume.
*Actions:* protect this segment. Cashback and rewards to stay top-of-wallet, instalment
offers on large purchases, and credit-limit increases — the lowest-risk group to extend.
Low interest revenue, but steady interchange income and minimal losses.

**Segment 2 — Dual-Use, High Balance (22.2%).** Heavy on both sides: purchase volumes
matching the transactors (1,307) *plus* the largest cash advances (2,244). They carry the
highest balances (2,864), hold the highest limits (5,330), and make the largest payments
(2,581) — but only 4.7% pay in full, and their payments-to-minimum ratio of 4.05 is the
lowest of the three, meaning they are servicing rather than clearing debt.
*Actions:* the highest-revenue and highest-exposure segment simultaneously. They are already
trusted with the largest limits, so the priority is watching for deterioration — rising limit
usage or a falling payment ratio — rather than growth. Balance-transfer offers can retain
them; premium/rewards tiers fit their spending, but any limit increase deserves scrutiny.

Note that segments 0 and 2 both borrow heavily via cash advances and look similar on balance
and limit usage. **What separates them is purchasing**, which is exactly PC1, the dominant
component. If a use case only cares about cash-advance exposure, segments 0 and 2 collapse
into one group — precisely the k=2 solution Ward finds.

---

## Caveats

**1. Separation is modest in absolute terms.** The best silhouette is 0.29. This is normal
for behavioural data: the segments are real, interpretable and stable across two independent
algorithms, but they are regions of a continuum, not well-separated islands. Any write-up
should say so rather than implying crisp natural clusters.

**2. k=3 was chosen with a business floor, not by the top metric.** Silhouette alone prefers
k=2. The choice is defensible — Davies-Bouldin independently prefers k=3, the silhouette cost
is 0.016, and Ward's k=2 solution is a clean merge of two of the three segments — but it is a
judgement call, and `MIN_BUSINESS_K` in the K-Means section is the one line that governs it.
Setting it to 2 reproduces the two-segment result.

**3. Two algorithms did not produce usable segmentations.** DBSCAN and GMM are reported for
completeness and as evidence about the data's shape (one dense mass, not a Gaussian mixture),
not as candidates. Do not quote DBSCAN's Davies-Bouldin without its cluster sizes.

**4. Winsorising trades information for stability.** Clipping at the 99th percentile
deliberately discards detail about roughly 90 extreme customers per feature. If those heavy
spenders are the point of the analysis, carve them out and study them separately.

**5. One snapshot only.** These are ~12 months of behaviour with `TENURE` nearly constant
(11.3–11.6 months across all three segments), so the segmentation cannot speak to how
customers move between groups. Re-fit periodically and verify membership is stable before
wiring it into campaigns.

**6. Cluster numbering is not stable across runs.** K-Means label ids depend on
initialisation. `RANDOM_STATE = 42` makes this run reproducible, but if you re-run with a
different seed, identify segments by their profile, not by their number.

---

## Scoring new customers

`clustering_model.joblib` carries the training-set caps and medians, so a single new customer
scores identically to one inside a large batch:

```python
# predict_segment() is defined in the last section of the notebook.
# To use it outside Colab, copy that cell into a .py file beside clustering_model.joblib.
import pandas as pd

new = pd.read_csv('new_customers.csv')   # same raw columns as CC GENERAL.csv
labels = predict_segment(new)            # -> array of cluster ids
```

It replays the whole chain: median imputation, the six engineered ratios, winsorising at the
stored caps, `log1p`, scaling, PCA, then `kmeans.predict()`. Verified against the training
data — the first five customers score `[1 0 1 2 1]`, matching their fitted labels exactly.
