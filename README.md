# 垃圾短信检测

基于 TF-IDF + LinearSVC / MultinomialNB 的中英文垃圾短信分类系统，配套 Gradio 交互界面。

## 运行步骤
1. `py -3.13 -m venv venv` 并激活
2. `pip install pandas numpy scikit-learn matplotlib seaborn nltk lime jieba joblib gradio`
3. `python spam_detector.py`（英文模型）
4. `python spam_zh.py`（中文模型）
5. `python app.py`（启动界面，访问 http://127.0.0.1:7860）

## 文件说明
- `spam_detector.py` 英文训练与评估
- `spam_zh.py` 中文训练与评估
- `app.py` Gradio 界面
- `results/` 评估图表与 LIME 解释
- `*.joblib` 已训练模型