#
#  -*- coding: utf-8 -*-

"""
Analisis Sentimen Ulasan Aplikasi Ruangguru
Metode: TF-IDF + SMOTE/RandomOverSampler + Multinomial Naive Bayes
"""

import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PALETTE = {
    "Negatif": "#EF4444",
    "Netral": "#F59E0B",
    "Positif": "#10B981",
}
SEED = 42
APP_ID = "com.ruangguru.livestudents"
N_REVIEWS = 1000
SCRAPE = False

# Semua file keluaran disimpan di folder yang sama dengan skrip ini.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "scrapped_data.csv"
STEM_CACHE_FILE = BASE_DIR / "stem_cache.json"
MODEL_PATH = BASE_DIR / "model_nb_sentimen.joblib"
CLEAN_DATA_PATH = BASE_DIR / "data_bersih_berlabel.csv"


def scrape_reviews():
    """Mengambil ulasan dari Google Play dan menyimpannya ke CSV."""
    from google_play_scraper import Sort, reviews

    result, _ = reviews(
        APP_ID,
        lang="id",
        country="id",
        sort=Sort.MOST_RELEVANT,
        count=N_REVIEWS,
        filter_score_with=None,
    )
    if not result:
        raise RuntimeError("Scraping tidak menghasilkan ulasan.")

    df_scrape = pd.DataFrame(result)
    required = ["userName", "score", "at", "content"]
    missing = [col for col in required if col not in df_scrape.columns]
    if missing:
        raise ValueError(f"Kolom hasil scraping tidak tersedia: {missing}")

    df_scrape = df_scrape[required].sort_values("at", ascending=False)
    df_scrape.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Jumlah ulasan yang diambil: {len(df_scrape)}")
    print(f"Data scraping disimpan di: {DATA_PATH}")


# 1. Memuat data
if SCRAPE or not DATA_PATH.exists():
    print("Melakukan scraping ulasan Google Play...")
    scrape_reviews()

df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
assert {"score", "content"}.issubset(df_raw.columns), (
    "Kolom 'score' dan 'content' wajib tersedia dalam dataset."
)

df = df_raw.dropna(subset=["content", "score"]).copy()
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])
df["at"] = pd.to_datetime(df["at"], errors="coerce") if "at" in df.columns else pd.NaT
df = df.reset_index(drop=True)

if df.empty:
    raise ValueError("Dataset kosong setelah menghapus data yang tidak valid.")

print(f"Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print("\nInformasi data:")
df.info()
print("\nMissing value:\n", df.isna().sum())
if df["at"].notna().any():
    print(
        f"\nRentang tanggal: {df['at'].min():%d %b %Y} - "
        f"{df['at'].max():%d %b %Y}"
    )
print(f"Ulasan duplikat: {df.duplicated(subset='content').sum()}")

# 2. Eksplorasi data (EDA)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

sns.countplot(data=df, x="score", color="#7C3AED", ax=ax[0])
ax[0].set_title("Distribusi Rating")
for patch in ax[0].patches:
    ax[0].annotate(
        int(patch.get_height()),
        (patch.get_x() + patch.get_width() / 2, patch.get_height()),
        ha="center",
        va="bottom",
    )

df["panjang_kata"] = df["content"].astype(str).str.split().str.len()
sns.histplot(df["panjang_kata"], bins=30, color="#7C3AED", ax=ax[1])
ax[1].set_title("Distribusi Panjang Ulasan (kata)")

df_tanggal = df.dropna(subset=["at"]).copy()
if not df_tanggal.empty:
    df_tanggal["tahun"] = df_tanggal["at"].dt.year
    yr = df_tanggal.groupby("tahun").agg(
        jumlah=("score", "size"),
        rating=("score", "mean"),
    )
    ax[2].bar(yr.index, yr["jumlah"], color="#A78BFA")
    ax2 = ax[2].twinx()
    ax2.plot(yr.index, yr["rating"], color="#EF4444", marker="o")
    ax2.set_ylim(1, 5)
    ax2.set_ylabel("Rata-rata rating")
else:
    ax[2].text(0.5, 0.5, "Tanggal tidak tersedia", ha="center", va="center")
ax[2].set_title("Jumlah Ulasan & Rating per Tahun")
plt.tight_layout()
plt.show()


# 3. Pelabelan sentimen berdasarkan rating
def beri_label(score):
    if score <= 2:
        return "Negatif"
    if score <= 3:
        return "Netral"
    return "Positif"


df["label"] = df["score"].apply(beri_label)
order = ["Negatif", "Netral", "Positif"]
dist = df["label"].value_counts().reindex(order, fill_value=0)
print("\nDistribusi sentimen:")
print(dist)
print((dist / dist.sum() * 100).round(1).astype(str) + " %")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x="label", order=order, palette=PALETTE, ax=ax[0])
ax[0].set_title("Jumlah per Sentimen")
ax[1].pie(
    dist.values,
    labels=order,
    autopct="%1.1f%%",
    colors=[PALETTE[label] for label in order],
    wedgeprops=dict(width=0.45),
)
ax[1].set_title("Proporsi Sentimen")
plt.tight_layout()
plt.show()


# 4. Preprocessing dan normalisasi teks
STOPWORDS = {
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh", "tentang",
    "kepada", "yang", "dan", "atau", "tetapi", "tapi", "karena",
    "meskipun", "jika", "agar", "saya", "aku", "kamu", "dia", "kita",
    "mereka", "ini", "itu", "lah", "tuh", "nya", "dong", "aja", "gitu",
    "deh", "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun",
    "juga", "sudah", "akan", "lagi", "pernah", "hanya", "adalah",
    "dalam", "ada", "para", "sebagai", "saat", "ketika", "kalau",
    "kalo", "jadi", "sama", "mau",
}
NEGATIONS = {
    "tidak", "bukan", "belum", "jangan", "kurang", "gak", "ga",
    "nggak", "ngga", "tak", "gk", "enggak",
}
DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "gk": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "bgt": "banget", "bgus": "bagus", "bagu": "bagus", "bagust": "bagus",
    "mantul": "mantap", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "yg": "yang", "dgn": "dengan", "krn": "karena",
    "karna": "karena", "utk": "untuk", "jd": "jadi", "tp": "tapi",
    "tpi": "tapi", "klo": "kalau", "kalo": "kalau", "sy": "saya",
    "sm": "sama", "sma": "sama", "bs": "bisa", "bsa": "bisa",
    "apk": "aplikasi", "app": "aplikasi", "aplikasinya": "aplikasi",
    "eror": "error", "error": "error", "erorr": "error",
    "terimakasih": "terima kasih", "trims": "terima kasih",
    "oke": "ok", "okey": "ok", "recomended": "recommended",
    "rekomen": "recommended", "kereen": "keren", "jelek": "buruk",
    "jlek": "buruk", "pake": "pakai", "pke": "pakai", "jg": "juga",
    "dr": "dari", "dl": "dulu", "bgmn": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "kyk": "seperti", "bnyk": "banyak",
    "bebrpa": "beberapa",
}

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError as exc:
    raise ImportError(
        "Paket Sastrawi belum terpasang. Jalankan: "
        "python -m pip install Sastrawi"
    ) from exc

stemmer = StemmerFactory().create_stemmer()
if STEM_CACHE_FILE.exists():
    with STEM_CACHE_FILE.open("r", encoding="utf-8") as f:
        stem_cache = json.load(f)
else:
    stem_cache = {}


def stem_kata(word):
    if word not in stem_cache:
        stem_cache[word] = stemmer.stem(word)
    return stem_cache[word]


def bersihkan_teks(teks, stemming=True, hapus_stopword=True):
    text = str(teks).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bruang\s+guru\b", "ruangguru", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = " ".join(SLANG.get(word, word) for word in text.split()).split()

    if hapus_stopword:
        tokens = [
            word for word in tokens
            if (word not in STOPWORDS or word in NEGATIONS)
            and word not in DOMAIN_STOPWORDS
        ]
        tokens = [word for word in tokens if len(word) > 1]

    if stemming:
        tokens = [stem_kata(word) for word in tokens]
    return " ".join(tokens)


contoh = "Aplikasi nya bgt membantuuu!! tpi kadang lemot & gak bisa login https://x.id"
print("\nContoh preprocessing:")
print("Asli :", contoh)
print("Bersih:", bersihkan_teks(contoh))

df["clean_content"] = df["content"].apply(bersihkan_teks)
with STEM_CACHE_FILE.open("w", encoding="utf-8") as f:
    json.dump(stem_cache, f, ensure_ascii=False)

n_awal = len(df)
df = df[df["clean_content"].str.strip().str.len() > 0].reset_index(drop=True)
df["jumlah_kata"] = df["clean_content"].str.split().str.len()
print(f"\nData setelah preprocessing: {len(df)} (dibuang {n_awal - len(df)})")
print(df[["content", "clean_content", "label"]].head(8))

if df["label"].nunique() < 2:
    raise ValueError("Data harus memiliki setidaknya dua kelas sentimen.")
if df["label"].value_counts().min() < 2:
    raise ValueError(
        "Setiap kelas perlu minimal 2 data agar pembagian data stratified dapat dilakukan."
    )


# 5. WordCloud dan kata teratas per sentimen
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, lab in zip(axes, order):
    text = " ".join(df.loc[df["label"] == lab, "clean_content"])
    if text.strip():
        wc = WordCloud(
            width=700,
            height=450,
            background_color="white",
            colormap="Reds" if lab == "Negatif" else "Oranges" if lab == "Netral" else "Greens",
            max_words=80,
            random_state=SEED,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, "Tidak ada teks", ha="center", va="center")
    ax.axis("off")
    ax.set_title(f"WordCloud - {lab}", fontsize=14)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, lab in zip(axes, order):
    text = " ".join(df.loc[df["label"] == lab, "clean_content"])
    top = pd.Series(text.split()).value_counts().head(12)[::-1]
    if not top.empty:
        ax.barh(top.index, top.values, color=PALETTE[lab])
    else:
        ax.text(0.5, 0.5, "Tidak ada kata", ha="center", va="center")
    ax.set_title(f"12 Kata Teratas - {lab}")
plt.tight_layout()
plt.show()


# 6. Pembagian data latih dan uji
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["clean_content"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=SEED,
)
print(f"\nData latih: {len(X_train_txt)} | Data uji: {len(X_test_txt)}")
print(pd.DataFrame({
    "Latih": y_train.value_counts(),
    "Uji": y_test.value_counts(),
}).fillna(0).astype(int))


def buat_sampler(y, faktor=1.0):
    counts = pd.Series(y).value_counts()
    min_n = int(counts.min() * faktor)
    k = min(5, min_n - 1)
    if k >= 1:
        return SMOTE(k_neighbors=k, random_state=SEED)
    return RandomOverSampler(random_state=SEED)


# 7. Pencarian parameter terbaik
N_FOLD = min(5, int(y_train.value_counts().min()))
if N_FOLD < 2:
    raise ValueError("Jumlah data tiap kelas pada data latih terlalu sedikit untuk cross-validation.")

pipe = ImbPipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("smote", buat_sampler(y_train, faktor=(N_FOLD - 1) / N_FOLD)),
    ("nb", MultinomialNB()),
])
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [2000, 5000],
    "tfidf__min_df": [1, 2, 3],
    "nb__alpha": [0.05, 0.1, 0.3, 0.5, 1.0],
}
gs = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1_macro",
    cv=StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED),
    n_jobs=-1,
    error_score="raise",
)
gs.fit(X_train_txt, y_train)
print("\nParameter terbaik:", gs.best_params_)
print(f"F1-macro CV terbaik: {gs.best_score_:.4f}")

hasil_grid = (
    pd.DataFrame(gs.cv_results_)[[
        "param_tfidf__ngram_range",
        "param_tfidf__max_features",
        "param_tfidf__min_df",
        "param_nb__alpha",
        "mean_test_score",
    ]]
    .rename(columns=lambda col: col.replace("param_", ""))
    .sort_values("mean_test_score", ascending=False)
)
print("\n10 kombinasi parameter terbaik:")
print(hasil_grid.head(10).to_string(index=False))

g = hasil_grid.groupby("nb__alpha")["mean_test_score"].max()
plt.figure(figsize=(6, 3.5))
plt.plot(g.index, g.values, marker="o", color="#7C3AED")
plt.xlabel("alpha")
plt.ylabel("F1-macro (CV, terbaik)")
plt.title("Pengaruh Alpha pada Kinerja")
plt.tight_layout()
plt.show()


# 8. TF-IDF dan penyeimbangan data latih
bp = gs.best_params_
vectorizer = TfidfVectorizer(
    ngram_range=bp["tfidf__ngram_range"],
    max_features=bp["tfidf__max_features"],
    min_df=bp["tfidf__min_df"],
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(X_train_txt)
X_test = vectorizer.transform(X_test_txt)
print("\nBentuk matriks TF-IDF latih:", X_train.shape)

sampler = buat_sampler(y_train)
X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
print("Sampler:", type(sampler).__name__)
print(pd.DataFrame({
    "Sebelum": y_train.value_counts(),
    "Sesudah": pd.Series(y_train_bal).value_counts(),
}).fillna(0).astype(int))


# 9. Pelatihan model
model = MultinomialNB(alpha=bp["nb__alpha"])
model.fit(X_train_bal, y_train_bal)
classes = list(model.classes_)
print(f"Model Multinomial Naive Bayes dilatih (alpha={bp['nb__alpha']})")
print("Kelas:", classes)


# 10. Evaluasi model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

metrik = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "F1-Score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "F1-Score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
}
tabel_metrik = pd.DataFrame(metrik, index=["Nilai"]).T
print("\nMetrik evaluasi:")
print(tabel_metrik.to_string(float_format=lambda value: f"{value:.4f}"))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=classes)
row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=ax[0])
ax[0].set(title="Confusion Matrix (jumlah)", xlabel="Prediksi", ylabel="Aktual")
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax[1])
ax[1].set(title="Confusion Matrix (% per kelas aktual / recall)", xlabel="Prediksi", ylabel="Aktual")
plt.tight_layout()
plt.show()

# ROC One-vs-Rest: dihitung hanya untuk kelas yang punya data positif dan negatif.
Y = label_binarize(y_test, classes=classes)
plt.figure(figsize=(6.5, 5))
aucs = {}
for i, c in enumerate(classes):
    if len(np.unique(Y[:, i])) < 2:
        print(f"ROC untuk kelas {c} dilewati karena data uji hanya memiliki satu nilai.")
        continue
    fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
    aucs[c] = roc_auc_score(Y[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, lw=2.5, color=PALETTE.get(c), label=f"{c} (AUC = {aucs[c]:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
if aucs:
    print(f"ROC-AUC macro: {np.mean(list(aucs.values())):.4f}")
else:
    print("ROC-AUC tidak dapat dihitung dari pembagian data uji ini.")


# 11. Cross-validation
best_pipe = ImbPipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=bp["tfidf__ngram_range"],
        max_features=bp["tfidf__max_features"],
        min_df=bp["tfidf__min_df"],
        sublinear_tf=True,
    )),
    ("smote", buat_sampler(y_train, faktor=(N_FOLD - 1) / N_FOLD)),
    ("nb", MultinomialNB(alpha=bp["nb__alpha"])),
])
skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED)
cv_acc = cross_val_score(best_pipe, X_train_txt, y_train, cv=skf, scoring="accuracy")
cv_f1 = cross_val_score(best_pipe, X_train_txt, y_train, cv=skf, scoring="f1_macro")
cv_df = pd.DataFrame({
    "Fold": [f"Fold {i + 1}" for i in range(N_FOLD)],
    "Accuracy": cv_acc,
    "F1-macro": cv_f1,
})
print("\nHasil cross-validation:")
print(cv_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"F1-macro : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
cv_df.set_index("Fold").plot(kind="bar", figsize=(7, 3.8), color=["#7C3AED", "#EC4899"], rot=0)
plt.ylim(0, 1)
plt.title("Hasil Cross-Validation")
plt.tight_layout()
plt.show()


# 12. Analisis kesalahan prediksi
salah = pd.DataFrame({
    "ulasan": X_test_txt.to_numpy(),
    "aktual": y_test.to_numpy(),
    "prediksi": y_pred,
    "keyakinan": y_prob.max(axis=1),
})
salah = salah[salah["aktual"] != salah["prediksi"]].sort_values("keyakinan", ascending=False)
print(f"\nJumlah salah prediksi: {len(salah)} dari {len(y_test)}")
print(salah.head(10).to_string(index=False))


# 13. Interpretasi fitur
fitur = np.array(vectorizer.get_feature_names_out())
flp = model.feature_log_prob_


def fitur_berpengaruh(top_n=15):
    rows = []
    for i, kelas in enumerate(classes):
        selisih = flp[i] - np.delete(flp, i, axis=0).mean(axis=0)
        for rank, j in enumerate(np.argsort(selisih)[::-1][:top_n], start=1):
            rows.append({
                "Kelas": kelas,
                "Rank": rank,
                "Fitur": fitur[j],
                "Skor Pengaruh": selisih[j],
                "Rasio Odds": np.exp(np.clip(selisih[j], -700, 700)),
                "log P(kata|kelas)": flp[i, j],
            })
    return pd.DataFrame(rows)


inf = fitur_berpengaruh(15)
fig, axes = plt.subplots(1, len(classes), figsize=(18, 6))
axes = np.atleast_1d(axes)
for ax, kelas in zip(axes, classes):
    d = inf[inf["Kelas"] == kelas].sort_values("Skor Pengaruh")
    ax.barh(d["Fitur"], d["Skor Pengaruh"], color=PALETTE.get(kelas, "#7C3AED"))
    ax.set_title(f"Influential Features - {kelas}", fontsize=13)
    ax.set_xlabel("Skor pengaruh")
plt.tight_layout()
plt.show()

print("\nFitur teratas per kelas:")
print(inf.pivot(index="Rank", columns="Kelas", values="Fitur").to_string())

pilih = []
for i in range(len(classes)):
    selisih = flp[i] - np.delete(flp, i, axis=0).mean(axis=0)
    pilih.extend(np.argsort(selisih)[::-1][:8].tolist())
pilih = list(dict.fromkeys(pilih))
hm = pd.DataFrame(flp[:, pilih].T, index=fitur[pilih], columns=classes)
plt.figure(figsize=(6, 9))
sns.heatmap(hm, annot=True, fmt=".1f", cmap="RdPu")
plt.title("log P(kata | kelas)")
plt.tight_layout()
plt.show()


# 14. Analisis sentimen berdasarkan aspek
ASPEK = {
    "Mentor / Guru": r"\b(guru|mentor|pengajar|tutor|kakak|penjelasan|menjelaskan)\b",
    "Materi & Soal": r"\b(materi|soal|latihan|rangkuman|pembahasan|kuis|pelajaran|kurikulum|bab)\b",
    "Video & Animasi": r"\b(video|animasi|tayang|nonton|subtitle|suara)\b",
    "Harga & Langganan": r"\b(harga|mahal|langganan|berlangganan|paket|bayar|biaya|promo|diskon|gratis|saldo|refund)\b",
    "Teknis & Performa": r"\b(error|bug|crash|lemot|lambat|loading|download|kuota|login|update|hang|ngelag|tidak\s+bisa|gak\s+bisa|susah)\b",
    "Fitur & Tampilan": r"\b(fitur|tampilan|menu|interface|desain|navigasi|opsi)\b",
    "Layanan & Iklan": r"\b(spam|iklan|telepon|sales|cs|customer|layanan|notifikasi|tawar)\b",
}
low = df["content"].fillna("").astype(str).str.lower()
rows = []
for nama, pola in ASPEK.items():
    mask = low.str.contains(pola, regex=True, na=False)
    sub = df[mask]
    rows.append({
        "Aspek": nama,
        "Jumlah": int(mask.sum()),
        "Rata-rata Rating": sub["score"].mean(),
        "% Negatif": (sub["label"] == "Negatif").mean() * 100 if len(sub) else 0.0,
        "% Netral": (sub["label"] == "Netral").mean() * 100 if len(sub) else 0.0,
        "% Positif": (sub["label"] == "Positif").mean() * 100 if len(sub) else 0.0,
    })
asp = pd.DataFrame(rows).sort_values("Jumlah", ascending=False).reset_index(drop=True)
print("\nAnalisis aspek:")
print(asp.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

asp.set_index("Aspek")[["% Negatif", "% Netral", "% Positif"]].plot(
    kind="barh",
    stacked=True,
    figsize=(9, 4.5),
    color=[PALETTE["Negatif"], PALETTE["Netral"], PALETTE["Positif"]],
)
plt.xlabel("% ulasan")
plt.title("Komposisi Sentimen per Aspek")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 15. Prediksi ulasan baru dan kata yang berkontribusi
def prediksi(teks, top_k=6):
    bersih = bersihkan_teks(teks)
    X = vectorizer.transform([bersih])
    probabilities = model.predict_proba(X)[0]
    pred = classes[int(probabilities.argmax())]
    kontrib = []

    if len(classes) > 1:
        ref = classes[int(np.argsort(probabilities)[::-1][1])]
        row = X.getrow(0)
        for j, value in zip(row.indices, row.data):
            score = float(
                (flp[classes.index(pred), j] - flp[classes.index(ref), j]) * value
            )
            kontrib.append((fitur[j], score))
    kontrib.sort(key=lambda item: item[1], reverse=True)
    return pred, dict(zip(classes, probabilities.round(3))), kontrib[:top_k]


contoh_ulasan = [
    "Aplikasinya bagus banget, materi lengkap dan gurunya jelas!",
    "Sering error, tidak bisa login, kecewa sekali",
    "Lumayan lah tapi masih banyak yang perlu diperbaiki",
]
for teks in contoh_ulasan:
    pred, prob, kontrib = prediksi(teks)
    print(f"\nUlasan: {teks}")
    print(f"Prediksi: {pred} | Probabilitas: {prob}")
    print("Kata pendorong:", [kata for kata, _ in kontrib])


# 16. Simpan model, dataset, dan cache stemming
joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model,
        "classes": classes,
        "best_params": bp,
        "metrics": metrik,
        "cm": cm,
    },
    MODEL_PATH,
)
df.to_csv(CLEAN_DATA_PATH, index=False, encoding="utf-8-sig")
print("\nFile berhasil disimpan:")
print(f"- Model: {MODEL_PATH}")
print(f"- Data bersih berlabel: {CLEAN_DATA_PATH}")
print(f"- Cache stemming: {STEM_CACHE_FILE}")