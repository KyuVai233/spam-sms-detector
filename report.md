# 垃圾短信检测实验报告

生成时间：2026-11-12 16:19:15

## 1. 数据
- 总样本：5,572（spam 747，ham 4,825）
- 训练/测试：4,457 / 1,115（分层）

## 2. 模型对比
| 模型 | Accuracy | Precision | Recall | F1 | AUC | FPR |
|---|---|---|---|---|---|---|
| MultinomialNB + TF-IDF | 0.9749 | 1.0000 | 0.8121 | 0.8963 | 0.9880 | 0.0000 |
| LinearSVC + TF-IDF | 0.9821 | 0.9574 | 0.9060 | 0.9310 | 0.9882 | 0.0062 |

## 3. 错误分析
- 误报 (ham → spam)：6 条
- 漏报 (spam → ham)：14 条

## 4. 输出文件
- `confusion_matrices.png`
- `roc_pr_curves.png`
- `threshold_tuning.png`
- `discriminative_words.png`
- `lime_correct_spam_*.html`
- `lime_false_positive_*.html`
- `spam_nb_pipeline.joblib` / `spam_svm_pipeline.joblib`

## 5. 建议
