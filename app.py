"""
垃圾短信检测 —— Gradio 交互界面（中英文双语模型）
运行：python app.py
"""

import os
import joblib
import jieba
import gradio as gr
from lime.lime_text import LimeTextExplainer

jieba.setLogLevel(20)

def chinese_tokenizer(text):
    return [w for w in jieba.lcut(str(text)) if w.strip()]

# 模型名 -> (文件路径, 是否为中文模型)
MODEL_FILES = {
    "LinearSVC (English)":     ("spam_svm_pipeline.joblib",     False),
    "MultinomialNB (English)": ("spam_nb_pipeline.joblib",      False),
    "LinearSVC (中文)":         ("spam_zh_svm_pipeline.joblib",  True),
    "MultinomialNB (中文)":     ("spam_zh_nb_pipeline.joblib",   True),
}

models = {}
is_chinese = {}
for name, (path, zh) in MODEL_FILES.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        is_chinese[name] = zh

if not models:
    raise FileNotFoundError(
        "未找到任何模型文件。请先运行 python spam_detector.py 和/或 python spam_zh.py。"
    )

explainers = {}
for name in models:
    if is_chinese[name]:
        explainers[name] = LimeTextExplainer(
            class_names=["ham", "spam"], split_expression=chinese_tokenizer
        )
    else:
        explainers[name] = LimeTextExplainer(class_names=["ham", "spam"])


# ============================================================
# 单条检测
# ============================================================
def predict_single(text, model_name, threshold, do_explain):
    if not text or not text.strip():
        return "请输入短信内容", "", "", ""

    model = models[model_name]
    prob = model.predict_proba([text])[0]
    p_ham, p_spam = float(prob[0]), float(prob[1])
    is_spam = p_spam >= threshold
    label = "🚫 垃圾短信" if is_spam else "✅ 正常短信"

    bar_html = f"""
    <div style="font-family: sans-serif; padding: 6px 0;">
      <div style="margin-bottom: 10px;">
        <b>正常 ham</b>：{p_ham:.2%}
        <div style="background:#e5e7eb;border-radius:4px;height:14px;margin-top:3px;">
          <div style="width:{p_ham*100:.1f}%;background:#22c55e;height:14px;border-radius:4px;"></div>
        </div>
      </div>
      <div>
        <b>垃圾 spam</b>：{p_spam:.2%}
        <div style="background:#e5e7eb;border-radius:4px;height:14px;margin-top:3px;">
          <div style="width:{p_spam*100:.1f}%;background:#ef4444;height:14px;border-radius:4px;"></div>
        </div>
      </div>
    </div>
    """

    detail = (
        f"模型：{model_name}\n"
        f"判定阈值：{threshold:.2f}\n"
        f"正常概率：{p_ham:.2%}\n"
        f"垃圾概率：{p_spam:.2%}\n"
        f"结论：{'判为垃圾（spam 概率 ≥ 阈值）' if is_spam else '判为正常（spam 概率 < 阈值）'}"
    )

    explain_html = ""
    if do_explain:
        try:
            exp = explainers[model_name].explain_instance(
                text, model.predict_proba, num_features=10, num_samples=300
            )
            rows = exp.as_list(label=1)
            max_abs = max((abs(v) for _, v in rows), default=1) or 1
            bars = []
            for word, w in rows:
                width = abs(w) / max_abs * 100
                color = "#ef4444" if w > 0 else "#3b82f6"
                sign = "+" if w > 0 else ""
                bars.append(
                    f"<div style='display:flex;align-items:center;margin:3px 0;"
                    f"font-family:sans-serif;font-size:13px;'>"
                    f"<span style='width:130px;text-align:right;padding-right:8px;'>{word}</span>"
                    f"<div style='flex:1;background:#f3f4f6;height:16px;border-radius:3px;'>"
                    f"<div style='width:{width:.1f}%;background:{color};height:16px;border-radius:3px;'></div>"
                    f"</div>"
                    f"<span style='width:75px;padding-left:8px;color:{color};'>{sign}{w:.4f}</span>"
                    f"</div>"
                )
            explain_html = (
                "<div style='max-width:720px;padding:6px 0;'>" + "".join(bars) + "</div>"
                + "<p style='font-size:12px;color:#666;font-family:sans-serif;'>"
                + "红色条 = 把判断推向 spam，蓝色条 = 推向 ham。仅显示影响最大的 10 个词。</p>"
            )
        except Exception as e:
            explain_html = f"<p style='color:#b91c1c;'>LIME 解释失败：{e}</p>"

    return label, bar_html, detail, explain_html


# ============================================================
# 批量检测
# ============================================================
def predict_batch(texts, model_name, threshold):
    if not texts or not texts.strip():
        return []
    model = models[model_name]
    lines = [t.strip() for t in texts.split("\n") if t.strip()]
    probs = model.predict_proba(lines)[:, 1]
    return [
        ["垃圾" if p >= threshold else "正常", f"{p:.2%}", t[:90]]
        for t, p in zip(lines, probs)
    ]


# ============================================================
# 界面
# ============================================================
with gr.Blocks(title="垃圾短信检测器") as demo:
    gr.Markdown(
        "# 📱 垃圾短信检测器\n"
        "支持英文（SMS Spam Collection）与中文（jieba 分词）两套模型，"
        "可调阈值并生成 LIME 词级解释。"
    )

    with gr.Tab("单条检测"):
        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(label="短信内容", lines=4,
                                     placeholder="粘贴或输入一条短信...")
                model_sel = gr.Radio(
                    choices=list(models.keys()),
                    value=list(models.keys())[0],
                    label="选择模型"
                )
                with gr.Row():
                    thr = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.01,
                                    label="判定阈值（spam 概率 ≥ 阈值 判为垃圾）")
                    do_explain = gr.Checkbox(value=True, label="生成 LIME 解释")
                btn = gr.Button("检测", variant="primary")

            with gr.Column(scale=2):
                label_out = gr.Label(label="判定结果")
                bar_out = gr.HTML(label="概率分布")
                detail_out = gr.Textbox(label="详情", lines=6)

        gr.Markdown("### 词级解释")
        explain_out = gr.HTML()

        btn.click(predict_single,
                  inputs=[text_in, model_sel, thr, do_explain],
                  outputs=[label_out, bar_out, detail_out, explain_out])
        text_in.submit(predict_single,
                       inputs=[text_in, model_sel, thr, do_explain],
                       outputs=[label_out, bar_out, detail_out, explain_out])

        gr.Examples(
            examples=[
                ["【XX银行】您的积分即将过期，点击链接兑换礼品"],
                ["晚上一起吃饭吗"],
                ["恭喜您中奖，请回复1领取"],
                ["WINNER!! You have been selected to receive a £900 prize. Call now!"],
                ["Hey, are we still meeting for lunch tomorrow?"],
            ],
            inputs=text_in,
            label="示例短信（点击填入，中英文都有）",
        )

    with gr.Tab("批量检测"):
        gr.Markdown("每行输入一条短信。中文模型请每行一条中文短信，英文模型请每行一条英文短信。")
        batch_in = gr.Textbox(label="短信列表（每行一条）", lines=12,
                              placeholder="晚上一起吃饭吗\n恭喜您中奖，请点击链接领取")
        with gr.Row():
            batch_model = gr.Radio(choices=list(models.keys()),
                                   value=list(models.keys())[0], label="模型")
            batch_thr = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.01, label="阈值")
        batch_btn = gr.Button("批量检测", variant="primary")
        batch_out = gr.Dataframe(headers=["判定", "垃圾概率", "短信"], label="结果", wrap=True)
        batch_btn.click(predict_batch,
                        inputs=[batch_in, batch_model, batch_thr],
                        outputs=batch_out)

    with gr.Tab("模型信息"):
        gr.Markdown(
            "## 已加载模型\n"
            + "\n".join(f"- **{n}** → `{MODEL_FILES[n][0]}`" for n in models)
            + "\n\n## 使用提示\n"
            "- **中文模型必须配中文短信，英文模型必须配英文短信。** 混用会得到无意义的结果。\n"
            "- **阈值**：调高（0.6–0.7）减少误报；调低（0.3–0.4）减少漏报。\n"
            "- **LIME 解释**：中文模型自动使用 jieba 分词，英文模型使用空格切分。\n"
            "- 指标详情见 `report.md` 和 `model_comparison.csv`。"
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)