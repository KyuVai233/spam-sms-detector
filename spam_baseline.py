"""
垃圾短信检测 —— 保底方案
TF-IDF + MultinomialNB / LinearSVC + 评估 + LIME 解释
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from lime.lime_text import LimeTextExplainer

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"总样本数: {len(df)}")
print(f"垃圾短信 (spam): {df['label'].sum()} 条")
print(f"正常短信 (ham):  {len(df) - df['label'].sum()} 条")

X = df['message'].values
y = df['label'].values

# 分层划分：保持训练集与测试集中 spam 比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 2. 构建两条 Pipeline
# ============================================================
# TF-IDF 参数说明：
#   ngram_range=(1, 2) 同时捕捉单词和二元短语（如 "call now"）
#   max_features=5000 控制特征维度，避免过拟合
#   sublinear_tf=True 对 TF 做对数平滑，缓解长文本主导
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)

pipe_nb = Pipeline([
    ('tfidf', tfidf),
    ('clf', MultinomialNB())
])

# LinearSVC 默认不输出概率，用 CalibratedClassifierCV 校准后支持 predict_proba，便于 LIME 使用
pipe_svm = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ('clf', CalibratedClassifierCV(LinearSVC(random_state=42), cv=3))
])

# ============================================================
# 3. 训练与评估
# ============================================================
def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)

    # 正常短信上的误报率 (FPR)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"模型: {name}")
    print(f"{'=' * 50}")
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"垃圾类精确率 (Precision): {prec:.4f}")
    print(f"垃圾类召回率 (Recall):    {rec:.4f}")
    print(f"垃圾类 F1:               {f1:.4f}")
    print(f"ROC-AUC:                {auc:.4f}")
    print(f"正常短信误报率 (FPR):     {fpr:.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")

    return {
        "model": pipeline,
        "name": name,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "fpr": fpr,
        "cm": confusion_matrix(y_test, y_pred)
    }

results_nb = evaluate_model("MultinomialNB + TF-IDF", pipe_nb,
                            X_train, y_train, X_test, y_test)
results_svm = evaluate_model("LinearSVC + TF-IDF", pipe_svm,
                             X_train, y_train, X_test, y_test)

# ============================================================
# 4. 混淆矩阵可视化
# ============================================================
def plot_confusion(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['预测 ham', '预测 spam'],
                yticklabels=['实际 ham', '实际 spam'])
    ax.set_title(title)
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_confusion(results_nb["cm"], "MultinomialNB", axes[0])
plot_confusion(results_svm["cm"], "LinearSVC", axes[1])
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# ============================================================
# 5. 模型对比汇总表
# ============================================================
comparison = pd.DataFrame([
    {
        "模型": r["name"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision (spam)": f"{r['precision']:.4f}",
        "Recall (spam)": f"{r['recall']:.4f}",
        "F1 (spam)": f"{r['f1']:.4f}",
        "ROC-AUC": f"{r['auc']:.4f}",
        "FPR (ham)": f"{r['fpr']:.4f}",
    }
    for r in [results_nb, results_svm]
])
print("\n模型对比汇总:")
print(comparison.to_string(index=False))
comparison.to_csv("model_comparison.csv", index=False)

# ============================================================
# 6. LIME 可解释性分析
# ============================================================
# 选取测试集中若干条样本进行解释（可手动替换 indices）
explainer = LimeTextExplainer(class_names=['ham', 'spam'])

# 优先解释被正确分类的垃圾短信，也可改成任意测试样本的索引
spam_indices = np.where((y_test == 1) & (results_svm["y_pred"] == 1))[0]
indices_to_explain = spam_indices[:3]  # 取前 3 条正确识别的垃圾短信

for idx in indices_to_explain:
    text_instance = X_test[idx]
    true_label = 'spam' if y_test[idx] == 1 else 'ham'
    pred_label = 'spam' if results_svm["y_pred"][idx] == 1 else 'ham'

    exp = explainer.explain_instance(
        text_instance,
        results_svm["model"].predict_proba,
        num_features=10,
        num_samples=500
    )

    print(f"\n{'=' * 50}")
    print(f"样本索引: {idx}")
    print(f"真实标签: {true_label}  |  预测标签: {pred_label}")
    print(f"短信内容: {text_instance[:150]}...")
    print(f"\nLIME 词语级归因 (对 spam 类的贡献):")
    for word, weight in exp.as_list(label=1):
        direction = "→ spam" if weight > 0 else "→ ham"
        print(f"  {word:20s} {weight:+.4f}  {direction}")

    # 保存 HTML 可视化
    exp.save_to_file(f"lime_explanation_{idx}.html")

print("\n完成。混淆矩阵已保存为 confusion_matrices.png，LIME 解释已保存为 lime_explanation_*.html")