"""
垃圾短信检测 —— 增强版
功能：
  1. 数据统计与探索 (EDA)
  2. 两个基线模型 + 5 折交叉验证
  3. 测试集评估 + 自动诊断
  4. 可视化：混淆矩阵 / ROC / PR / 阈值调优 / 关键词对比
  5. 错误样本分析（误报 + 漏报）
  6. LIME 可解释性（正确样本 + 错误样本）
  7. 模型保存 + Markdown 报告导出
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ============================================================
# 工具函数：让输出更清晰
# ============================================================
LINE = "═" * 66
SUB  = "─" * 62

def header(title):
    print("\n" + LINE)
    print(f"  {title}")
    print(LINE)

def sub(title):
    print(f"\n── {title} " + "─" * max(2, 62 - len(title) * 2))

def kv(key, value, indent=2, width=22):
    print(" " * indent + f"{key:<{width}} {value}")

def status_tag(ok, warn=False):
    if ok:     return "✅"
    if warn:   return "⚠️ "
    return "❌"

# ============================================================
# 1. 数据加载与统计
# ============================================================
header("📊 第 1 部分：数据加载与统计")

DATA_FILE = "SMSSpamCollection"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"未找到数据文件 {DATA_FILE}，请先下载解压到当前目录。\n"
        "下载地址：https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    )

df = pd.read_csv(DATA_FILE, sep="\t", names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

n_total = len(df)
n_spam  = int(df["label"].sum())
n_ham   = n_total - n_spam

kv("数据集",        DATA_FILE)
kv("总样本数",      f"{n_total:,}")
kv("垃圾短信 spam", f"{n_spam:,}  ({n_spam / n_total:.1%})")
kv("正常短信 ham",  f"{n_ham:,}  ({n_ham / n_total:.1%})")

X = df["message"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
kv("训练集 / 测试集", f"{len(X_train):,} / {len(X_test):,}")
kv("训练集 spam 比例", f"{y_train.mean():.1%}")
kv("测试集 spam 比例", f"{y_test.mean():.1%}")

# ============================================================
# 2. 数据探索（EDA）
# ============================================================
header("🔎 第 2 部分：数据探索")

df["length"] = df["message"].str.len()
df["has_url"] = df["message"].str.contains(r"http|www", case=False, regex=True)
df["has_money"] = df["message"].str.contains(r"£|\$|€|cash|prize", case=False, regex=True)
df["has_bang"] = df["message"].str.contains("!")

def stat_row(name, col):
    s = df.groupby("label")[col].mean()
    return f"{name:<10}  ham={s[0]:>8.2f}  spam={s[1]:>8.2f}"

print("   指标对比 (ham vs spam)：")
print("   " + stat_row("平均长度",  "length"))
print("   " + stat_row("含URL比例", "has_url"))
print("   " + stat_row("含金额词", "has_money"))
print("   " + stat_row("含感叹号", "has_bang"))

# ============================================================
# 3. 构建模型
# ============================================================
header("🏗  第 3 部分：构建模型")

def make_tfidf():
    return TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=5000, sublinear_tf=True
    )

pipe_nb = Pipeline([
    ("tfidf", make_tfidf()),
    ("clf",   MultinomialNB())
])

pipe_svm = Pipeline([
    ("tfidf", make_tfidf()),
    ("clf",   CalibratedClassifierCV(LinearSVC(random_state=42), cv=3))
])

print("  已构建两个 Pipeline：")
print("  • MultinomialNB + TF-IDF")
print("  • LinearSVC(校准) + TF-IDF")

# ============================================================
# 4. 交叉验证
# ============================================================
header("🔁 第 4 部分：5 折交叉验证（稳定性评估）")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def cv_report(name, pipe):
    sub(name)
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    for metric in scoring:
        vals = r[f"test_{metric}"]
        kv(metric, f"{vals.mean():.4f}  ± {vals.std():.4f}")

cv_report("MultinomialNB + TF-IDF", pipe_nb)
cv_report("LinearSVC + TF-IDF",     pipe_svm)

# ============================================================
# 5. 测试集评估 + 自动诊断
# ============================================================
header("📈 第 5 部分：测试集评估")

def auto_diagnose(m):
    msgs = []
    msgs.append(("垃圾类召回率 Recall", m["recall"],
                 status_tag(m["recall"] >= 0.90, m["recall"] >= 0.80),
                 "召回率 ≥ 0.90 为理想"))
    msgs.append(("误报率 FPR", m["fpr"],
                 status_tag(m["fpr"] <= 0.02, m["fpr"] <= 0.05),
                 "FPR ≤ 2% 为理想"))
    msgs.append(("垃圾类精确率 Precision", m["precision"],
                 status_tag(m["precision"] >= 0.90, m["precision"] >= 0.80),
                 "Precision ≥ 0.90 为理想"))
    return msgs

def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "name": name, "model": pipe, "y_pred": y_pred, "y_prob": y_prob,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall":    recall_score(y_test, y_pred, pos_label=1),
        "f1":        f1_score(y_test, y_pred, pos_label=1),
        "auc":       roc_auc_score(y_test, y_prob),
        "fpr":       fp / (fp + tn) if (fp + tn) else 0,
        "cm":        np.array([[tn, fp], [fn, tp]]),
    }

    sub(name)
    kv("准确率 Accuracy",      f"{metrics['accuracy']:.4f}")
    kv("垃圾类 Precision",     f"{metrics['precision']:.4f}")
    kv("垃圾类 Recall",        f"{metrics['recall']:.4f}")
    kv("垃圾类 F1",            f"{metrics['f1']:.4f}")
    kv("ROC-AUC",              f"{metrics['auc']:.4f}")
    kv("正常短信误报率 FPR",   f"{metrics['fpr']:.4f}")

    print("\n  自动诊断：")
    for label, val, tag, hint in auto_diagnose(metrics):
        print(f"    {tag} {label}: {val:.4f}   ({hint})")

    return metrics

res_nb  = evaluate("MultinomialNB + TF-IDF", pipe_nb)
res_svm = evaluate("LinearSVC + TF-IDF",     pipe_svm)

# 汇总表
sub("模型对比汇总")
comp = pd.DataFrame([
    {"模型": r["name"],
     "Accuracy":  f"{r['accuracy']:.4f}",
     "Precision": f"{r['precision']:.4f}",
     "Recall":    f"{r['recall']:.4f}",
     "F1":        f"{r['f1']:.4f}",
     "ROC-AUC":   f"{r['auc']:.4f}",
     "FPR":       f"{r['fpr']:.4f}"}
    for r in [res_nb, res_svm]
])
print(comp.to_string(index=False))
comp.to_csv("model_comparison.csv", index=False)

# ============================================================
# 6. 可视化：混淆矩阵 / ROC / PR / 阈值调优
# ============================================================
header("🎨 第 6 部分：可视化")

# 6.1 混淆矩阵
def plot_cm(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred ham", "Pred spam"],
                yticklabels=["True ham", "True spam"])
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_cm(res_nb["cm"],  "MultinomialNB", axes[0])
plot_cm(res_svm["cm"], "LinearSVC",     axes[1])
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()
print("  ✅ confusion_matrices.png")

# 6.2 ROC + PR 曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for r, color in [(res_nb, "tab:blue"), (res_svm, "tab:orange")]:
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax1.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})", color=color)
    pr, rc, _ = precision_recall_curve(y_test, r["y_prob"])
    ax2.plot(rc, pr, label=r["name"], color=color)
ax1.plot([0, 1], [0, 1], "k--", lw=1); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve"); ax1.legend(loc="lower right"); ax1.grid(alpha=.3)
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve"); ax2.legend(); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig("roc_pr_curves.png", dpi=150); plt.close()
print("  ✅ roc_pr_curves.png")

# 6.3 阈值调优
def threshold_sweep(y_true, y_prob):
    ts = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in ts:
        yp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        rows.append({
            "thr": t,
            "precision": precision_score(y_true, yp, zero_division=0),
            "recall":    recall_score(y_true, yp, zero_division=0),
            "f1":        f1_score(y_true, yp, zero_division=0),
            "fpr":       fp / (fp + tn) if (fp + tn) else 0,
        })
    return pd.DataFrame(rows)

sweeps = {"NB":  threshold_sweep(y_test, res_nb["y_prob"]),
          "SVM": threshold_sweep(y_test, res_svm["y_prob"])}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (name, d) in zip(axes, sweeps.items()):
    ax.plot(d["thr"], d["precision"], label="Precision", color="tab:green")
    ax.plot(d["thr"], d["recall"],    label="Recall",    color="tab:red")
    ax.plot(d["thr"], d["f1"],        label="F1",        color="tab:blue", lw=2)
    ax.plot(d["thr"], d["fpr"],       label="FPR",       color="tab:gray", ls="--")
    best_t = d.loc[d["f1"].idxmax(), "thr"]
    ax.axvline(best_t, color="k", ls=":", lw=1, label=f"best F1 @ {best_t:.2f}")
    ax.set_title(f"{name}: Threshold Tuning")
    ax.set_xlabel("Threshold"); ax.grid(alpha=.3); ax.legend(fontsize=8)
axes[0].set_ylabel("Score")
plt.tight_layout(); plt.savefig("threshold_tuning.png", dpi=150); plt.close()
print("  ✅ threshold_tuning.png")

best_thr_svm = sweeps["SVM"].loc[sweeps["SVM"]["f1"].idxmax()]
print(f"\n  SVM 最佳阈值 (F1 最大): {best_thr_svm['thr']:.2f}  "
      f"→ F1={best_thr_svm['f1']:.4f}, Recall={best_thr_svm['recall']:.4f}, FPR={best_thr_svm['fpr']:.4f}")

# ============================================================
# 7. 关键词分析（哪些词最能区分两类）
# ============================================================
header("🔤 第 7 部分：关键词分析")

def top_words(pipe, X, y, top_n=15):
    vec = pipe.named_steps["tfidf"]
    Xv  = vec.transform(X)
    feats = np.array(vec.get_feature_names_out())
    spam_mean = np.asarray(Xv[y == 1].mean(axis=0)).ravel()
    ham_mean  = np.asarray(Xv[y == 0].mean(axis=0)).ravel()
    diff = spam_mean - ham_mean
    top_spam = feats[np.argsort(diff)[-top_n:][::-1]]
    top_ham  = feats[np.argsort(diff)[:top_n]]
    return top_spam, top_ham

spam_words_nb, ham_words_nb = top_words(pipe_nb, X_train, y_train)

sub("指向 spam 的 top-15 词")
print("  " + ", ".join(spam_words_nb))
sub("指向 ham 的 top-15 词")
print("  " + ", ".join(ham_words_nb))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, words, color, title in [
    (axes[0], spam_words_nb, "tab:red",  "Top spam-indicating words"),
    (axes[1], ham_words_nb,  "tab:blue", "Top ham-indicating words"),
]:
    y_pos = np.arange(len(words))[::-1]
    ax.barh(y_pos, np.ones(len(words)), color=color, alpha=.75)
    ax.set_yticks(y_pos); ax.set_yticklabels(words)
    ax.set_xticks([]); ax.set_title(title)
plt.tight_layout(); plt.savefig("discriminative_words.png", dpi=150); plt.close()
print("\n  ✅ discriminative_words.png")

# ============================================================
# 8. 错误样本分析
# ============================================================
header("🧐 第 8 部分：错误样本分析")

y_pred_svm = res_svm["y_pred"]
fp_idx = np.where((y_test == 0) & (y_pred_svm == 1))[0]  # 误报
fn_idx = np.where((y_test == 1) & (y_pred_svm == 0))[0]  # 漏报

sub(f"误报样本 (ham → spam)，共 {len(fp_idx)} 条，展示前 5 条")
for i in fp_idx[:5]:
    print(f"  · [idx {i}] {X_test[i][:100]}")

sub(f"漏报样本 (spam → ham)，共 {len(fn_idx)} 条，展示前 5 条")
for i in fn_idx[:5]:
    print(f"  · [idx {i}] {X_test[i][:100]}")

# ============================================================
# 9. LIME 解释（正确 + 误报样本）
# ============================================================
header("🔍 第 9 部分：LIME 可解释性")

try:
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["ham", "spam"])

    def lime_explain(idx, tag):
        text = X_test[idx]
        true = "spam" if y_test[idx] == 1 else "ham"
        pred = "spam" if y_pred_svm[idx] == 1 else "ham"
        exp = explainer.explain_instance(
            text, res_svm["model"].predict_proba,
            num_features=8, num_samples=500
        )
        sub(f"{tag}  [idx {idx}]  真实={true}  预测={pred}")
        print(f"  短信: {text[:120]}")
        print(f"  LIME 归因 (对 spam 的贡献)：")
        for w, v in exp.as_list(label=1):
            arrow = "→ spam" if v > 0 else "→ ham "
            print(f"    {w:<20} {v:+.4f}  {arrow}")
        exp.save_to_file(f"lime_{tag}_{idx}.html")

    # 正确识别的垃圾短信 1 条
    ok_spam = np.where((y_test == 1) & (y_pred_svm == 1))[0]
    if len(ok_spam): lime_explain(ok_spam[0], "correct_spam")

    # 误报的 ham 1 条（最有分析价值）
    if len(fp_idx):  lime_explain(fp_idx[0], "false_positive")

except ImportError:
    print("  ⚠️ 未安装 lime，跳过。安装：pip install lime")

# ============================================================
# 10. 模型保存
# ============================================================
header("💾 第 10 部分：模型保存")

joblib.dump(pipe_nb,  "spam_nb_pipeline.joblib")
joblib.dump(pipe_svm, "spam_svm_pipeline.joblib")
print("  ✅ spam_nb_pipeline.joblib")
print("  ✅ spam_svm_pipeline.joblib")

# ============================================================
# 11. 导出 Markdown 报告
# ============================================================
header("📝 第 11 部分：导出报告")

report = f"""# 垃圾短信检测实验报告

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 数据
- 总样本：{n_total:,}（spam {n_spam:,}，ham {n_ham:,}）
- 训练/测试：{len(X_train):,} / {len(X_test):,}（分层）

## 2. 模型对比
| 模型 | Accuracy | Precision | Recall | F1 | AUC | FPR |
|---|---|---|---|---|---|---|
"""
for r in [res_nb, res_svm]:
    report += (f"| {r['name']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
               f"{r['recall']:.4f} | {r['f1']:.4f} | {r['auc']:.4f} | {r['fpr']:.4f} |\n")

report += f"""
## 3. 错误分析
- 误报 (ham → spam)：{len(fp_idx)} 条
- 漏报 (spam → ham)：{len(fn_idx)} 条

## 4. 输出文件
- `confusion_matrices.png`
- `roc_pr_curves.png`
- `threshold_tuning.png`
- `discriminative_words.png`
- `lime_correct_spam_*.html`
- `lime_false_positive_*.html`
- `spam_nb_pipeline.joblib` / `spam_svm_pipeline.joblib`

## 5. 建议
"""
if res_svm["fpr"] > 0.02:
    report += "- ⚠️ 误报率偏高，建议提高判定阈值或对 ham 误报样本做进一步特征分析。\n"
if res_svm["recall"] < 0.90:
    report += "- ⚠️ 召回率不足，建议降低阈值或补充训练数据。\n"
if res_svm["f1"] >= 0.94 and res_svm["fpr"] <= 0.02:
    report += "- ✅ 整体指标达到课程设计目标水平。\n"

with open("report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("  ✅ report.md")

# ============================================================
# 总结
# ============================================================
header("🎯 总结与建议")
kv("推荐模型",    "LinearSVC (校准)" if res_svm["f1"] >= res_nb["f1"] else "MultinomialNB")
kv("最佳 F1",     f"{max(res_nb['f1'], res_svm['f1']):.4f}")
kv("建议阈值",    f"{best_thr_svm['thr']:.2f}（F1 最大点）")
print("\n  所有文件已保存到当前目录，可打开 .html 查看 LIME 可视化。")