# -*- coding: utf-8 -*-
"""
中文垃圾短信检测 —— 完整基线
运行：python spam_zh.py

数据格式：每行 "标签\t短信内容"，标签 0=正常 1=垃圾
若没有数据文件，会自动使用内置示例数据跑通流程（仅用于验证代码）
"""

import os
import jieba
import joblib
import numpy as np
import pandas as pd
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
    f1_score, precision_score, recall_score, roc_auc_score
)

jieba.setLogLevel(20)

# ============================================================
# 0. 中文分词函数（关键改造点）
# ============================================================
def chinese_tokenizer(text):
    # 过滤空字符串和纯空格，避免 LIME 扰动样本产生异常 token
    return [w for w in jieba.lcut(str(text)) if w.strip()]

# ============================================================
# 1. 加载数据
# ============================================================
DATA_CANDIDATES = ["带标签短信.txt", "spam_zh.txt", "chinese_spam.txt"]

def load_data():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            print(f"[数据] 读取 {path}")
            df = pd.read_csv(path, sep="\t", names=["label", "message"],
                             encoding="utf-8", on_bad_lines="skip")
            df = df.dropna()
            df["label"] = df["label"].astype(int)
            return df
    print("[数据] 未找到中文数据文件，使用内置示例数据（仅用于跑通流程）")
    samples = [
        (1, "恭喜您获得一等奖，点击链接领取奖品"),
        (1, "【XX银行】您的积分即将过期，请点击链接兑换礼品"),
        (1, "免费领取流量包，回复TD退订"),
        (1, "您有一笔退款待领取，点击链接填写银行卡信息"),
        (1, "限时优惠，全场五折，点击立即购买"),
        (1, "您的账户异常，请点击链接验证身份"),
        (1, "抽奖活动中奖通知，请回复1确认"),
        (1, "低价代购，加微信详聊"),
        (0, "晚上一起吃饭吗"),
        (0, "明天上午十点开会，别忘了"),
        (0, "快递已经放到门口了"),
        (0, "妈，我这周末回家"),
        (0, "作业提交截止时间改到周五了"),
        (0, "收到，谢谢"),
        (0, "明天天气怎么样"),
        (0, "我在路上，马上到"),
        (0, "文件已发你邮箱，请查收"),
        (0, "会议纪要整理好了，稍后发群里"),
        (1, "恭喜您被抽中为幸运用户，请拨打客服电话领取"),
        (0, "记得带伞，今天可能下雨"),
    ]
    return pd.DataFrame(samples, columns=["label", "message"])

df = load_data()
df = df[df["message"].astype(str).str.len() > 0].reset_index(drop=True)
n_total = len(df)
n_spam = int(df["label"].sum())
print(f"[数据] 总样本 {n_total} | spam {n_spam} | ham {n_total - n_spam}")

X = df["message"].astype(str).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ============================================================
# 2. 构建 Pipeline（tokenizer=chinese_tokenizer 是关键）
# ============================================================
def make_tfidf():
    return TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,      # 必须 None，否则与自定义 tokenizer 冲突
        lowercase=False,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )

pipe_nb = Pipeline([("tfidf", make_tfidf()), ("clf", MultinomialNB())])

# 小样本时 cv 折数自动降为 2，避免某折只有一类
min_class = int(min(y_train.sum(), len(y_train) - y_train.sum()))
cv_folds = 3 if min_class >= 6 else 2
pipe_svm = Pipeline([
    ("tfidf", make_tfidf()),
    ("clf", CalibratedClassifierCV(LinearSVC(random_state=42), cv=cv_folds)),
])

# ============================================================
# 3. 训练与评估
# ============================================================
def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    m = {
        "model": pipe, "name": name,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "cm": np.array([[tn, fp], [fn, tp]]),
    }
    print(f"\n===== {name} =====")
    print(f"F1(spam)   {m['f1']:.4f}")
    print(f"Precision  {m['precision']:.4f}")
    print(f"Recall     {m['recall']:.4f}")
    print(f"AUC        {m['auc']:.4f}")
    print(f"FPR        {m['fpr']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    return m

res_nb = evaluate("MultinomialNB", pipe_nb)
res_svm = evaluate("LinearSVC", pipe_svm)

# 混淆矩阵图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, r in zip(axes, [res_nb, res_svm]):
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred ham", "Pred spam"],
                yticklabels=["True ham", "True spam"])
    ax.set_title(r["name"])
plt.tight_layout()
plt.savefig("confusion_matrices_zh.png", dpi=150)
plt.close()
print("\n[图] confusion_matrices_zh.png")

# ============================================================
# 4. 保存模型
# ============================================================
joblib.dump(pipe_svm, "spam_zh_svm_pipeline.joblib")
joblib.dump(pipe_nb,  "spam_zh_nb_pipeline.joblib")
print("[保存] spam_zh_svm_pipeline.joblib / spam_zh_nb_pipeline.joblib")

# ============================================================
# 5. 预测演示
# ============================================================
demo_texts = [
    "【XX银行】您的积分即将过期，点击链接兑换礼品",
    "晚上一起吃饭吗",
    "恭喜您中奖，请回复1领取",
]
print("\n===== 预测演示 =====")
for t in demo_texts:
    p = pipe_svm.predict_proba([t])[0][1]
    print(f"spam概率={p:.2%}  {'垃圾' if p >= 0.5 else '正常'}  | {t}")

# ============================================================
# 6. LIME 中文解释
# ============================================================
try:
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(
        class_names=["ham", "spam"],
        split_expression=chinese_tokenizer,   # 中文必须传分词函数
    )
    text = "恭喜您中奖，请点击链接领取奖品"
    exp = explainer.explain_instance(
        text, pipe_svm.predict_proba, num_features=8, num_samples=300
    )
    print(f"\n[LIME] 解释：{text}")
    for w, v in exp.as_list(label=1):
        print(f"  {w:<12} {v:+.4f}  {'→ spam' if v > 0 else '→ ham'}")
    exp.save_to_file("lime_zh_example.html")
    print("[LIME] 已保存 lime_zh_example.html")
except ImportError:
    print("[LIME] 未安装 lime，跳过")
except Exception as e:
    print(f"[LIME] 解释失败：{e}")