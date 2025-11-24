import os, json, math, warnings, textwrap, itertools, gc, pickle, datetime as dt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    roc_auc_score,
)
from scipy.stats import spearmanr


warnings.filterwarnings("ignore")

PROJECT_DIR = "/Users/jsu_m3/Desktop/MCS-UIUC/ISYE7406/Project"
OUT_FIG = os.path.join(PROJECT_DIR, "outputs", "figures")
OUT_TAB = os.path.join(PROJECT_DIR, "outputs", "tables")
OUT_MOD = os.path.join(PROJECT_DIR, "outputs", "models")

os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_MOD, exist_ok=True)

np.random.seed(7406)


def _load_table(basename):
    paths = [
        os.path.join(PROJECT_DIR, "outputs", "data", f"{basename}.parquet"),
        os.path.join(PROJECT_DIR, "outputs", "data", f"{basename}.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError(f"Missing {basename}.parquet/csv in outputs/tables")


products_df = _load_table("products")
ingredients_df = _load_table("ingredients")
labels_df = _load_table("labels")
print(products_df.shape, ingredients_df.shape, labels_df.shape)


def _norm_cols(df):
    return {c: c.strip() for c in df.columns}


products_df.rename(columns=_norm_cols(products_df), inplace=True)
ingredients_df.rename(columns=_norm_cols(ingredients_df), inplace=True)
labels_df.rename(columns=_norm_cols(labels_df), inplace=True)

prod_id_col = next(
    c
    for c in products_df.columns
    if c.lower() in ["id", "product_id", "label_id", "dsld_id"]
)
brand_col = next((c for c in products_df.columns if "brand" in c.lower()), None)
date_col = next((c for c in products_df.columns if "entrydate" in c.lower()), None)
phys_col = next((c for c in products_df.columns if "physicalstate" in c.lower()), None)
ptype_col = next((c for c in products_df.columns if "producttype" in c.lower()), None)
targets_col = next(
    (c for c in products_df.columns if "targetgroups" in c.lower()), None
)
nact_col = "n_actives" if "n_actives" in products_df.columns else None

ing_pid_col = next(
    c
    for c in ingredients_df.columns
    if c.lower() in ["product_id", "id", "dsld_id", "label_id"]
)
ing_name_col = next(
    (
        c
        for c in ingredients_df.columns
        if "ingredient" in c.lower() and "name" in c.lower()
    ),
    None,
)
ing_cat_col = next((c for c in ingredients_df.columns if "category" in c.lower()), None)
dose_col = next(
    (c for c in ingredients_df.columns if c.lower() in ["dose", "quantity", "amount"]),
    None,
)
unit_col = next((c for c in ingredients_df.columns if "unit" in c.lower()), None)

lab_pid_col = next(
    c
    for c in labels_df.columns
    if c.lower() in ["product_id", "id", "dsld_id", "label_id"]
)
label_col = next(
    (c for c in labels_df.columns if c.lower() in ["label", "claim", "category"]), None
)

PRIMARY_LABELS = [
    "Immune",
    "Energy",
    "Sleep/Calm",
    "Cognitive/Focus",
    "Joint/Bone",
    "Heart/Cardio",
    "Digestive/Gut",
    "Men’s/Women’s",
    "Sports/Performance",
    "General Wellness",
]

labels_agg = (
    labels_df[[lab_pid_col, label_col]]
    .dropna()
    .assign(**{label_col: lambda d: d[label_col].astype(str)})
    .groupby(lab_pid_col)[label_col]
    .agg(lambda x: sorted(set([y for y in x if y in PRIMARY_LABELS])))
    .reset_index()
)
labels_agg.rename(
    columns={lab_pid_col: "product_id", label_col: "labels"}, inplace=True
)

prod = products_df.rename(columns={prod_id_col: "product_id"}).merge(
    labels_agg, on="product_id", how="left"
)
prod["labels"] = prod["labels"].apply(lambda v: v if isinstance(v, list) else [])

prod = prod[prod["labels"].map(len) > 0].copy()
print("Cohort with proposal-aligned labels:", prod.shape)


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

K_TOP = 500


def clean_ing(x):
    return str(x).strip().lower()


ing = ingredients_df.rename(
    columns={ing_pid_col: "product_id", ing_name_col: "ingredient_name"}
).copy()
ing["ingredient_name"] = ing["ingredient_name"].map(clean_ing)

ing_counts = (
    ing.groupby("ingredient_name")["product_id"].nunique().sort_values(ascending=False)
)
top_ings = set(ing_counts.head(K_TOP).index.tolist())

ing_top = (
    ing[ing["ingredient_name"].isin(top_ings)]
    .drop_duplicates(subset=["product_id", "ingredient_name"])
    .assign(val=1)
    .pivot(index="product_id", columns="ingredient_name", values="val")
    .fillna(0)
    .astype("uint8")
)

X_blocks = []

if nact_col and nact_col in prod.columns:
    n_active = prod[["product_id", nact_col]].set_index("product_id").astype("float32")
else:
    n_active = ing.groupby("product_id").size().to_frame("n_actives").astype("float32")
X_blocks.append(n_active)

cat_frames = []
for cname in [phys_col, ptype_col]:
    if cname and cname in prod.columns:
        s = prod[["product_id", cname]].fillna("Unknown").copy()
        s[cname] = s[cname].astype(str)
        oh = pd.get_dummies(s.set_index("product_id")[cname], prefix=cname[:8])
        cat_frames.append(oh.astype("uint8"))
if cat_frames:
    X_blocks.append(pd.concat(cat_frames, axis=1))

if targets_col and targets_col in prod.columns:
    tg = prod[["product_id", targets_col]].copy()

    def _as_list(v):
        if isinstance(v, list):
            return v
        if pd.isna(v):
            return []
        if "|" in str(v):
            return [t.strip() for t in str(v).split("|")]
        if ";" in str(v):
            return [t.strip() for t in str(v).split(";")]
        return [str(v).strip()] if str(v).strip() else []

    tg["tg_list"] = tg[targets_col].apply(_as_list)
    rows = []
    for pid, lst in tg[["product_id", "tg_list"]].itertuples(index=False):
        for t in set(lst):
            rows.append((pid, t))
    if rows:
        tg_long = pd.DataFrame(rows, columns=["product_id", "target"])
        tg_oh = (
            pd.get_dummies(tg_long.set_index("product_id")["target"], prefix="tg")
            .groupby(level=0)
            .max()
        )
        X_blocks.append(tg_oh.astype("uint8"))

X = pd.concat([ing_top] + X_blocks, axis=1).fillna(0)
X = X.loc[sorted(set(prod["product_id"]).intersection(X.index))]

Y_labels = prod[["product_id", "labels"]].set_index("product_id").loc[X.index]
mlb = MultiLabelBinarizer(classes=PRIMARY_LABELS)
Y = pd.DataFrame(
    mlb.fit_transform(Y_labels["labels"]), index=Y_labels.index, columns=mlb.classes_
).astype("uint8")


binary_cols = X.columns[(X.max() == 1) & (X.min() == 0)]
num_cols = X.columns.difference(binary_cols)
X_scaled = X.copy()
if len(num_cols) > 0:
    scaler = StandardScaler()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

print("Feature matrix:", X_scaled.shape, "| Label matrix:", Y.shape)


primary_for_strat = Y.idxmax(axis=1)

X_tr, X_te, Y_tr, Y_te = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=7406, stratify=primary_for_strat
)

logit_cv = OneVsRestClassifier(
    LogisticRegressionCV(
        Cs=[0.1, 0.5, 1.0, 2.0, 5.0],
        cv=5,
        penalty="l2",
        solver="liblinear",
        n_jobs=-1,
        scoring="f1_macro",
        max_iter=200,
    ),
    n_jobs=-1,
)
logit_cv.fit(X_tr, Y_tr)

P_te = pd.DataFrame(logit_cv.predict_proba(X_te), index=X_te.index, columns=Y.columns)
Y_hat = (P_te.values >= 0.5).astype(int)


def multilabel_metrics(Y_true, Y_prob, Y_pred):

    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    ap = {c: average_precision_score(Y_true[c], Y_prob[c]) for c in Y_true.columns}
    return micro, macro, ap


log_micro, log_macro, log_ap = multilabel_metrics(
    Y_te, P_te, pd.DataFrame(Y_hat, index=Y_te.index, columns=Y_te.columns)
)

pd.Series({"micro_f1": log_micro, "macro_f1": log_macro}).to_csv(
    os.path.join(OUT_TAB, "logit_holdout_f1.csv")
)
pd.Series(log_ap, name="AP").to_csv(
    os.path.join(OUT_TAB, "logit_holdout_ap_per_class.csv")
)
P_te.to_csv(os.path.join(OUT_TAB, "logit_holdout_pred_proba.csv"))

with open(os.path.join(OUT_MOD, "logit_ovr_cv.pkl"), "wb") as f:
    pickle.dump(logit_cv, f)

print("Logistic OVR — micro F1:", round(log_micro, 3), "macro F1:", round(log_macro, 3))

rf = OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=10,
        n_jobs=-1,
        random_state=7406,
    ),
    n_jobs=-1,
)
rf.fit(X_tr, Y_tr)
P_te_rf = pd.DataFrame(rf.predict_proba(X_te), index=X_te.index, columns=Y_te.columns)
Y_hat_rf = (P_te_rf.values >= 0.5).astype(int)

rf_micro, rf_macro, rf_ap = multilabel_metrics(
    Y_te, P_te_rf, pd.DataFrame(Y_hat_rf, index=Y_te.index, columns=Y_te.columns)
)
pd.Series({"micro_f1": rf_micro, "macro_f1": rf_macro}).to_csv(
    os.path.join(OUT_TAB, "rf_holdout_f1.csv")
)
pd.Series(rf_ap, name="AP").to_csv(os.path.join(OUT_TAB, "rf_holdout_ap_per_class.csv"))
P_te_rf.to_csv(os.path.join(OUT_TAB, "rf_holdout_pred_proba.csv"))

with open(os.path.join(OUT_MOD, "rf_ovr.pkl"), "wb") as f:
    pickle.dump(rf, f)

print(
    "RandomForest OVR — micro F1:", round(rf_micro, 3), "macro F1:", round(rf_macro, 3)
)

PXI = (ing_top.reindex(X.index).fillna(0) > 0).astype(int)
PXL = Y.copy()


N = len(PXI)
ing_tot = PXI.sum(axis=0)
lab_tot = PXL.sum(axis=0)
co_mat = PXI.T.dot(PXL)


rows = []
for ing_name in PXI.columns:
    a = co_mat.loc[ing_name]
    b = ing_tot[ing_name] - a
    c = lab_tot - a
    d = N - (a + b + c)

    OR = ((a + 1) * (d + 1) / ((b + 1) * (c + 1))).astype(float)

    p_xy = (a + 1) / (N + len(PXI.columns))
    p_x = (ing_tot[ing_name] + 1) / (N + len(PXI.columns))
    p_y = (lab_tot + 1) / (N + len(PXI.columns))
    PMI = np.log2(p_xy / (p_x * p_y))
    for lab in PXL.columns:
        rows.append((ing_name, lab, int(a[lab]), float(OR[lab]), float(PMI[lab])))

assoc = pd.DataFrame(rows, columns=["ingredient", "label", "n_co", "odds_ratio", "pmi"])
assoc.sort_values(["label", "odds_ratio"], ascending=[True, False], inplace=True)
assoc.to_csv(os.path.join(OUT_TAB, "ingredient_label_associations.csv"), index=False)


topk = []
for lab in PXL.columns:
    t = assoc[(assoc["label"] == lab) & (assoc["n_co"] >= 25)].head(20).copy()
    t["rank"] = range(1, len(t) + 1)
    topk.append(t)
topk = pd.concat(topk, axis=0)
topk.to_csv(
    os.path.join(OUT_TAB, "ingredient_top20_drivers_per_label.csv"), index=False
)
print("Saved ingredient→label association tables.")


UL_TABLE = {
    "vitamin c": (2000.0, "mg", "UL (NIH ODS, adults)"),
    "vitamin b6": (100.0, "mg", "UL (adults)"),
    "vitamin b12": (None, "mcg", "No UL; flag very high doses >2000 mcg"),
    "calcium": (2500.0, "mg", "UL (adults 19-50)"),
    "magnesium": (350.0, "mg", "UL (supplemental, not total)"),
    "zinc": (40.0, "mg", "UL (adults)"),
    "sodium": (2300.0, "mg", "CDC limit; not a UL)"),
}


if all(c is not None for c in [dose_col, unit_col]):
    dose = ingredients_df.rename(
        columns={
            ing_pid_col: "product_id",
            ing_name_col: "ingredient_name",
            dose_col: "dose",
            unit_col: "unit",
        }
    )
    dose["ingredient_name"] = dose["ingredient_name"].str.lower().str.strip()
    dose["unit"] = dose["unit"].str.lower().str.strip()

    UNIT_TO_MG = {
        "mg": 1.0,
        "mcg": 1 / 1000.0,
        "µg": 1 / 1000.0,
        "mcg/µg": 1 / 1000.0,
        "g": 1000.0,
    }

    def to_mg(row, target_unit):

        u = str(row["unit"]).lower()
        val = pd.to_numeric(row["dose"], errors="coerce")
        if pd.isna(val) or u not in UNIT_TO_MG:
            return np.nan
        mg = val * UNIT_TO_MG[u]
        if target_unit == "mg":
            return mg
        if target_unit == "mcg":
            return mg * 1000.0
        return np.nan

    flags = []
    for name, (ul, unit, note) in UL_TABLE.items():
        sub = dose[dose["ingredient_name"].str.contains(name)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["dose_norm"] = sub.apply(lambda r: to_mg(r, unit), axis=1)
        if ul is not None:
            sub["flag"] = sub["dose_norm"] > ul
        else:

            if name == "vitamin b12":
                sub["flag"] = sub["dose_norm"] > 2000.0
            else:
                sub["flag"] = False
        flags.append(
            sub.assign(ref_ul=ul, ref_unit=unit, ref_note=note)[
                [
                    "product_id",
                    "ingredient_name",
                    "dose",
                    "unit",
                    "dose_norm",
                    "ref_ul",
                    "ref_unit",
                    "ref_note",
                    "flag",
                ]
            ]
        )

    if flags:
        dose_flags = pd.concat(flags, axis=0)
        dose_flags.to_csv(os.path.join(OUT_TAB, "dose_ul_flags.csv"), index=False)
        if brand_col and brand_col in products_df.columns:
            brand_map = products_df.rename(columns={prod_id_col: "product_id"})[
                ["product_id", brand_col]
            ]
            brand_flags = dose_flags.merge(brand_map, on="product_id", how="left")
            brand_sum = (
                brand_flags.groupby(brand_col)["flag"]
                .mean()
                .sort_values(ascending=False)
                .rename("flag_rate")
                .to_frame()
            )
            brand_sum.to_csv(os.path.join(OUT_TAB, "dose_ul_flag_rate_by_brand.csv"))
        print("Saved dose UL flags.")
    else:
        print("No matched ingredients for UL table.")
else:
    print("Dose/unit columns not available; skipping UL flags for now.")


pca = PCA(n_components=10, random_state=7406)
Z = pca.fit_transform(PXI.loc[X.index])
k = 8
km = KMeans(n_clusters=k, random_state=7406, n_init=20)
cl = km.fit_predict(Z)

clusters = pd.DataFrame({"product_id": X.index, "cluster": cl})
clusters.to_csv(os.path.join(OUT_TAB, "clusters_pca10_k8.csv"), index=False)

lab_enrich = (
    clusters.merge(Y, left_on="product_id", right_index=True)
    .groupby("cluster")[Y.columns]
    .mean()
    .sort_index()
)
lab_enrich.to_csv(os.path.join(OUT_TAB, "cluster_label_prevalence.csv"))


ing_prev = clusters.join(PXI, on="product_id").groupby("cluster")[PXI.columns].mean()

rows = []
for cid, row in ing_prev.iterrows():
    top15 = row.sort_values(ascending=False).head(15)
    for name, rate in top15.items():
        rows.append((cid, name, rate))
pd.DataFrame(rows, columns=["cluster", "ingredient", "presence_rate"]).to_csv(
    os.path.join(OUT_TAB, "cluster_top15_ingredients.csv"), index=False
)
print("Saved clustering outputs.")


if date_col and date_col in products_df.columns:
    prod_dates = products_df.rename(columns={prod_id_col: "product_id"})[
        ["product_id", date_col]
    ].copy()
    prod_dates[date_col] = pd.to_datetime(prod_dates[date_col], errors="coerce")
    month_map = prod_dates.dropna().assign(
        month=lambda d: d[date_col].values.astype("datetime64[M]")
    )[["product_id", "month"]]
    Yw = Y.merge(
        month_map.set_index("product_id"), left_index=True, right_index=True, how="left"
    ).dropna(subset=["month"])

    trend_rows = []
    months_sorted = np.sort(Yw["month"].unique())
    month_rank = {m: i for i, m in enumerate(months_sorted)}
    for lab in Y.columns:
        df_lab = Yw[["month", lab]].copy()
        df_lab["t"] = df_lab["month"].map(month_rank)

        mon = df_lab.groupby("t")[lab].mean()
        if len(mon) >= 3:
            rho, p = spearmanr(mon.index, mon.values)
            trend_rows.append((lab, float(rho), float(p), mon.iloc[0], mon.iloc[-1]))
    trend = pd.DataFrame(
        trend_rows,
        columns=["label", "spearman_rho", "p_value", "start_prev", "end_prev"],
    )
    trend.to_csv(os.path.join(OUT_TAB, "monthly_label_trend_spearman.csv"), index=False)


if brand_col and brand_col in products_df.columns:
    brand_map = products_df.rename(columns={prod_id_col: "product_id"})[
        ["product_id", brand_col]
    ]
    Yb = Y.merge(
        brand_map.set_index("product_id"), left_index=True, right_index=True, how="left"
    ).dropna(subset=[brand_col])

    counts = Yb.groupby(brand_col).size()
    keep_brands = counts[counts >= 30].index
    Yb = Yb[Yb[brand_col].isin(keep_brands)]
    brand_prev = Yb.groupby(brand_col)[Y.columns].mean()
    brand_prev.to_csv(os.path.join(OUT_TAB, "brand_label_prevalence.csv"))
    print("Saved brand prevalence.")


OUT = "/Users/jsu_m3/Desktop/MCS-UIUC/ISYE7406/Project/outputs"
TAB, MOD = os.path.join(OUT, "tables"), os.path.join(OUT, "models")
os.makedirs(TAB, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
np.random.seed(7406)

pid_col = next(
    c
    for c in products_df.columns
    if c.lower() in ["id", "product_id", "label_id", "dsld_id"]
)
brand_map = (
    products_df[[pid_col, brand_col]]
    .rename(columns={pid_col: "product_id"})
    .set_index("product_id")
)
groups = brand_map.reindex(X_scaled.index)[brand_col].fillna("Unknown")

gkf = GroupKFold(n_splits=5)
oof = np.zeros((len(X_scaled), Y.shape[1]), dtype=float)
fold_rows = []
models = []

for fold, (tr, va) in enumerate(gkf.split(X_scaled, Y, groups), 1):
    X_tr, X_va = X_scaled.iloc[tr], X_scaled.iloc[va]
    Y_tr, Y_va = Y.iloc[tr], Y.iloc[va]
    base = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=10,
        random_state=7406,
        n_jobs=-1,
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_tr, Y_tr)
    P = pd.DataFrame(clf.predict_proba(X_va), index=X_va.index, columns=Y.columns)
    oof[va, :] = P.values
    Yb = (P.values >= 0.5).astype(int)
    micro = f1_score(Y_va, Yb, average="micro", zero_division=0)
    macro = f1_score(Y_va, Yb, average="macro", zero_division=0)
    ap = {c: average_precision_score(Y_va[c], P[c]) for c in Y.columns}
    fold_rows.append(
        {"fold": fold, "micro_f1": micro, "macro_f1": macro}
        | {f"AP_{k}": v for k, v in ap.items()}
    )
    models.append(clf)
    print(f"[RF] Fold {fold}: microF1={micro:.3f}, macroF1={macro:.3f}")

oof_df = pd.DataFrame(oof, index=X_scaled.index, columns=Y.columns)
oof_df.to_csv(os.path.join(TAB, "rf_oof_pred_proba.csv"), index=True)
pd.DataFrame(fold_rows).to_csv(
    os.path.join(TAB, "rf_groupcv_fold_metrics.csv"), index=False
)
with open(os.path.join(MOD, "rf_groupcv_models.pkl"), "wb") as f:
    pickle.dump(models, f)

thr = {}
for c in Y.columns:
    y = Y[c].values.astype(int)
    p = oof_df[c].values.astype(float)
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    thr[c] = float(grid[int(np.argmax(f1s))])
thr_s = pd.Series(thr, name="threshold")
thr_s.to_csv(os.path.join(TAB, "thresholds_oof_f1_opt.csv"))
print("Saved thresholds:", thr_s.to_dict())


date_map = products_df[[pid_col, date_col]].rename(columns={pid_col: "product_id"})
date_map[date_col] = pd.to_datetime(date_map[date_col], errors="coerce")
date_idx = date_map.set_index("product_id").reindex(X_scaled.index).dropna()
train_idx = date_idx[date_idx[date_col].dt.year == 2023].index
test_idx = date_idx[date_idx[date_col].dt.year.isin([2024, 2025])].index

X_tr, Y_tr = X_scaled.loc[train_idx], Y.loc[train_idx]
X_te, Y_te = X_scaled.loc[test_idx], Y.loc[test_idx]

clf_rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=500, random_state=7406, n_jobs=-1), n_jobs=-1
)
clf_rf.fit(X_tr, Y_tr)
P_te = pd.DataFrame(clf_rf.predict_proba(X_te), index=X_te.index, columns=Y.columns)

thr = pd.read_csv(os.path.join(TAB, "thresholds_oof_f1_opt.csv"), index_col=0)[
    "threshold"
].to_dict()
Yb_te = pd.DataFrame(
    {c: (P_te[c].values >= thr[c]).astype(int) for c in Y.columns}, index=X_te.index
)

micro = f1_score(Y_te, Yb_te, average="micro", zero_division=0)
macro = f1_score(Y_te, Yb_te, average="macro", zero_division=0)
ap = {c: average_precision_score(Y_te[c], P_te[c]) for c in Y.columns}
pd.Series({"micro_f1": micro, "macro_f1": macro}).to_csv(
    os.path.join(TAB, "rf_temporal_f1.csv")
)
pd.Series(ap, name="AP").to_csv(os.path.join(TAB, "rf_temporal_ap.csv"))
print(f"[Temporal RF] microF1={micro:.3f}, macroF1={macro:.3f}")

clf_l1 = OneVsRestClassifier(
    LogisticRegression(
        penalty="l1", C=0.5, solver="liblinear", max_iter=300, n_jobs=-1
    ),
    n_jobs=-1,
)
clf_l1.fit(X_scaled, Y)

rows = []
for j, c in enumerate(Y.columns):
    w = clf_l1.estimators_[j].coef_.ravel()
    top = np.argsort(-w)[:30]
    for idx in top:
        rows.append((c, X_scaled.columns[idx], float(w[idx])))
pd.DataFrame(rows, columns=["label", "feature", "coef"]).to_csv(
    os.path.join(TAB, "logit_l1_topcoef_per_label.csv"), index=False
)
print("Saved: logit_l1_topcoef_per_label.csv")
