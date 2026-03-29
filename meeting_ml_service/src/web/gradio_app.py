"""
Gradio web interface for Meeting ML Service.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

from src.config.settings import settings
from src.inference.predictor import Predictor


# Global predictor
predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Get or initialize predictor."""
    global predictor
    if predictor is None:
        predictor = Predictor()
        predictor.load_all_models()
    return predictor


def load_metrics_data(model_type: str, task: str) -> Optional[Dict[str, Any]]:
    """Load metrics from file."""
    metrics_path = settings.metrics_dir / model_type / task / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_history_data(model_type: str, task: str) -> Optional[Dict[str, Any]]:
    """Load training history."""
    history_path = (
        settings.metrics_dir / model_type / task / "training_history.json"
    )
    if not history_path.exists():
        return None
    with open(history_path, "r") as f:
        return json.load(f)


# =============================================================================
# INFERENCE TAB
# =============================================================================


def predict_all(
    transcript: str,
    model_type: str,
) -> Tuple[str, str, str, Dict, Dict, Dict]:
    """
    Make predictions for all tasks.

    Args:
        transcript: Input transcript
        model_type: Model type selection

    Returns:
        Results for all three tasks
    """
    if not transcript.strip():
        return (
            "Введіть транскрипцію",
            "Введіть транскрипцію",
            "Введіть транскрипцію",
            {},
            {},
            {},
        )

    pred = get_predictor()

    # Determine which model to use
    if model_type == "Both (Compare)":
        results_tfidf = pred.predict_all_tasks(transcript, "tfidf")
        results_bert = pred.predict_all_tasks(transcript, "bert")

        # Format comparison results
        def format_comparison(task: str) -> Tuple[str, Dict]:
            tfidf = results_tfidf.get(task, {})
            bert = results_bert.get(task, {})

            text = (
                f"**TF-IDF:** {tfidf.get('prediction', 'N/A')} "
                f"({tfidf.get('confidence', 0):.2%})\n\n"
                f"**BERT:** {bert.get('prediction', 'N/A')} "
                f"({bert.get('confidence', 0):.2%})"
            )

            # Combine probabilities
            probs = {}
            tfidf_probs = tfidf.get("probabilities", {})
            bert_probs = bert.get("probabilities", {})

            for key in set(tfidf_probs.keys()) | set(bert_probs.keys()):
                probs[f"TF-IDF: {key}"] = tfidf_probs.get(key, 0)
                probs[f"BERT: {key}"] = bert_probs.get(key, 0)

            return text, probs

        decision_text, decision_probs = format_comparison("decision")
        topic_text, topic_probs = format_comparison("topic_type")
        da_text, da_probs = format_comparison("da")

        return (
            decision_text,
            topic_text,
            da_text,
            decision_probs,
            topic_probs,
            da_probs,
        )

    else:
        # Single model type
        mt = "tfidf" if model_type == "TF-IDF" else "bert"
        results = pred.predict_all_tasks(transcript, mt)

        def format_single(task: str) -> Tuple[str, Dict]:
            result = results.get(task, {})

            if result.get("error"):
                return f"Error: {result['error']}", {}

            pred_val = result.get("prediction", "N/A")
            conf = result.get("confidence", 0)

            text = f"**{pred_val}** (confidence: {conf:.2%})"
            probs = result.get("probabilities", {})

            return text, probs

        decision_text, decision_probs = format_single("decision")
        topic_text, topic_probs = format_single("topic_type")
        da_text, da_probs = format_single("da")

        return (
            decision_text,
            topic_text,
            da_text,
            decision_probs,
            topic_probs,
            da_probs,
        )


def create_prob_chart(probs: Dict[str, float], title: str) -> go.Figure:
    """Create probability bar chart."""
    if not probs:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    # Sort by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_probs[:10]]  # Top 10
    values = [p[1] for p in sorted_probs[:10]]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="steelblue",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


# =============================================================================
# METRICS TAB
# =============================================================================


def get_metrics_table() -> pd.DataFrame:
    """Get metrics summary table."""
    rows = []

    for model_type in ["tfidf", "bert"]:
        for task in ["decision", "topic_type", "da"]:
            data = load_metrics_data(model_type, task)

            if data is not None:
                test = data.get("metrics", {}).get("test", {})
                rows.append({
                    "Model": model_type.upper(),
                    "Task": task,
                    "Accuracy": f"{test.get('accuracy', 0):.4f}",
                    "Precision": f"{test.get('precision', 0):.4f}",
                    "Recall": f"{test.get('recall', 0):.4f}",
                    "F1": f"{test.get('f1', 0):.4f}",
                })
            else:
                rows.append({
                    "Model": model_type.upper(),
                    "Task": task,
                    "Accuracy": "N/A",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "F1": "N/A",
                })

    return pd.DataFrame(rows)


def get_model_details(model_type: str, task: str) -> Tuple[str, Any, Any, Any]:
    """Get detailed metrics for a specific model."""
    data = load_metrics_data(model_type.lower(), task)
    history = load_history_data(model_type.lower(), task)

    if data is None:
        return (
            "Модель не натренована або метрики не знайдено",
            None,
            None,
            None,
        )

    # Summary text
    metrics = data.get("metrics", {})
    test = metrics.get("test", {})

    summary = f"""
### Метрики тестової вибірки

| Метрика | Значення |
|---------|----------|
| Accuracy | {test.get('accuracy', 0):.4f} |
| Precision | {test.get('precision', 0):.4f} |
| Recall | {test.get('recall', 0):.4f} |
| F1 Score | {test.get('f1', 0):.4f} |
"""

    if "roc_auc" in test:
        summary += f"| ROC-AUC | {test.get('roc_auc', 0):.4f} |\n"
        summary += f"| PR-AUC | {test.get('pr_auc', 0):.4f} |\n"
    elif "roc_auc_ovr" in test:
        summary += f"| ROC-AUC (OVR) | {test.get('roc_auc_ovr', 0):.4f} |\n"

    # Per-class metrics
    per_class = test.get("per_class", {})
    if per_class:
        summary += "\n### Метрики по класам\n\n"
        summary += "| Клас | Precision | Recall | F1 | Support |\n"
        summary += "|------|-----------|--------|----|---------|\n"
        for cls, cls_metrics in per_class.items():
            summary += (
                f"| {cls} | "
                f"{cls_metrics.get('precision', 0):.4f} | "
                f"{cls_metrics.get('recall', 0):.4f} | "
                f"{cls_metrics.get('f1', 0):.4f} | "
                f"{cls_metrics.get('support', 0)} |\n"
            )

    # Confusion matrix plot
    cm = test.get("confusion_matrix")
    cm_fig = None
    if cm is not None:
        cm_array = np.array(cm)
        task_config = settings.tasks.get(task, {})
        
        # Get class names from per_class metrics (these match the confusion matrix order)
        per_class = test.get("per_class", {})
        if per_class:
            # Use class names from per_class (they match the confusion matrix order)
            class_names_for_cm = list(per_class.keys())
        else:
            # Fallback to config class names (may not match if some classes are missing)
            class_names_for_cm = task_config.get("class_names", [])
            # Truncate to match matrix size
            class_names_for_cm = class_names_for_cm[:cm_array.shape[0]]

        cm_fig = px.imshow(
            cm_array,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=class_names_for_cm,
            y=class_names_for_cm,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix - {model_type.upper()} {task}",
            text_auto=True,
        )
        cm_fig.update_layout(height=400)

    # Learning curves plot
    lc_fig = None
    if history and "train_loss" in history:
        epochs = list(range(1, len(history["train_loss"]) + 1))

        lc_fig = go.Figure()

        if "train_loss" in history:
            lc_fig.add_trace(go.Scatter(
                x=epochs, y=history["train_loss"],
                mode="lines+markers", name="Train Loss"
            ))
        if "val_loss" in history:
            lc_fig.add_trace(go.Scatter(
                x=epochs, y=history["val_loss"],
                mode="lines+markers", name="Val Loss"
            ))
        if "train_accuracy" in history:
            lc_fig.add_trace(go.Scatter(
                x=epochs, y=history["train_accuracy"],
                mode="lines+markers", name="Train Acc",
                yaxis="y2"
            ))
        if "val_accuracy" in history:
            lc_fig.add_trace(go.Scatter(
                x=epochs, y=history["val_accuracy"],
                mode="lines+markers", name="Val Acc",
                yaxis="y2"
            ))

        lc_fig.update_layout(
            title=f"Learning Curves - {model_type.upper()} {task}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Accuracy",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
            height=400,
        )

    # ROC/PR curves
    roc_fig = None
    if "roc_curve" in test:
        # Binary classification ROC curve
        roc = test["roc_curve"]
        pr = test.get("pr_curve", {})

        roc_fig = go.Figure()

        # ROC curve
        roc_fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines", name=f"ROC (AUC={test.get('roc_auc', 0):.3f})"
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Random",
            line=dict(dash="dash")
        ))

        roc_fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
        )
    elif "roc_auc_ovr" in test:
        # Multiclass ROC-AUC (show value but no curve available)
        roc_fig = go.Figure()
        roc_fig.add_annotation(
            text=f"ROC-AUC (One-vs-Rest): {test.get('roc_auc_ovr', 0):.4f}<br>"
                 "Multiclass ROC curves are not available.<br>"
                 "Use per-class metrics for detailed analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14),
            align="center"
        )
        roc_fig.update_layout(
            title="ROC-AUC (Multiclass)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=350,
        )

    return summary, cm_fig, lc_fig, roc_fig


def get_comparison_chart() -> go.Figure:
    """Create comparison chart for all models."""
    data = []

    for model_type in ["tfidf", "bert"]:
        for task in ["decision", "topic_type", "da"]:
            metrics_data = load_metrics_data(model_type, task)

            if metrics_data is not None:
                test = metrics_data.get("metrics", {}).get("test", {})
                data.append({
                    "Model": f"{model_type.upper()}",
                    "Task": task,
                    "Accuracy": test.get("accuracy", 0),
                    "F1": test.get("f1", 0),
                })

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Task",
        y="F1",
        color="Model",
        barmode="group",
        title="F1 Score Comparison: TF-IDF vs BERT",
    )

    fig.update_layout(height=400)

    return fig


# =============================================================================
# GRADIO APP
# =============================================================================


def create_gradio_app() -> gr.Blocks:
    """Create Gradio application."""

    with gr.Blocks(
        title="Meeting ML Service",
    ) as app:

        gr.Markdown("""
        # 🎯 Meeting ML Service

        ML сервіс для аналізу транскрипцій зустрічей з AMI Corpus.

        **Задачі класифікації:**
        - **Decision Detection** — виявлення рішень
        - **Topic Type** — класифікація типу теми
        - **Dialogue Acts** — класифікація діалогових актів
        """)

        with gr.Tabs():
            # =========================================================
            # TAB 1: INFERENCE
            # =========================================================
            with gr.TabItem("🔮 Inference"):
                gr.Markdown("### Введіть транскрипцію для аналізу")

                with gr.Row():
                    transcript_input = gr.Textbox(
                        label="Транскрипція",
                        placeholder=(
                            "[A]: So I think we should go with the "
                            "rubber buttons. [B]: Yeah, that makes sense. "
                            "[C]: Agreed, let's do it."
                        ),
                        lines=5,
                    )

                with gr.Row():
                    model_type_input = gr.Radio(
                        choices=["TF-IDF", "BERT", "Both (Compare)"],
                        value="TF-IDF",
                        label="Тип моделі",
                    )
                    predict_btn = gr.Button(
                        "🚀 Аналізувати",
                        variant="primary",
                    )

                gr.Markdown("### Результати")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Decision Detection")
                        decision_output = gr.Markdown()
                        decision_chart = gr.Plot(label="Probabilities")

                    with gr.Column():
                        gr.Markdown("#### Topic Type")
                        topic_output = gr.Markdown()
                        topic_chart = gr.Plot(label="Probabilities")

                    with gr.Column():
                        gr.Markdown("#### Dialogue Acts")
                        da_output = gr.Markdown()
                        da_chart = gr.Plot(label="Probabilities")

                # Connect prediction function
                def predict_and_chart(transcript, model_type):
                    (
                        d_text, t_text, da_text,
                        d_probs, t_probs, da_probs
                    ) = predict_all(transcript, model_type)

                    d_chart = create_prob_chart(d_probs, "Decision Probabilities")
                    t_chart = create_prob_chart(t_probs, "Topic Type Probabilities")
                    da_chart = create_prob_chart(da_probs, "DA Probabilities")

                    return d_text, t_text, da_text, d_chart, t_chart, da_chart

                predict_btn.click(
                    fn=predict_and_chart,
                    inputs=[transcript_input, model_type_input],
                    outputs=[
                        decision_output, topic_output, da_output,
                        decision_chart, topic_chart, da_chart,
                    ],
                )

            # =========================================================
            # TAB 2: METRICS
            # =========================================================
            with gr.TabItem("📊 Metrics"):
                gr.Markdown("### Метрики натренованих моделей")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Оновити метрики")

                # Summary table
                metrics_table = gr.Dataframe(
                    value=get_metrics_table,
                    label="Зведена таблиця метрик",
                    interactive=False,
                )

                # Comparison chart
                comparison_chart = gr.Plot(
                    value=get_comparison_chart,
                    label="Порівняння моделей",
                )

                gr.Markdown("### Детальні метрики моделі")

                with gr.Row():
                    detail_model = gr.Dropdown(
                        choices=["TFIDF", "BERT"],
                        value="TFIDF",
                        label="Тип моделі",
                    )
                    detail_task = gr.Dropdown(
                        choices=["decision", "topic_type", "da"],
                        value="decision",
                        label="Задача",
                    )
                    detail_btn = gr.Button("Показати деталі")

                detail_summary = gr.Markdown()

                with gr.Row():
                    cm_plot = gr.Plot(label="Confusion Matrix")
                    lc_plot = gr.Plot(label="Learning Curves")

                roc_plot = gr.Plot(label="ROC / PR Curves")

                # Connect functions
                def refresh_metrics():
                    return get_metrics_table(), get_comparison_chart()

                refresh_btn.click(
                    fn=refresh_metrics,
                    outputs=[metrics_table, comparison_chart],
                )

                detail_btn.click(
                    fn=get_model_details,
                    inputs=[detail_model, detail_task],
                    outputs=[detail_summary, cm_plot, lc_plot, roc_plot],
                )

        gr.Markdown("""
        ---
        *Meeting ML Service - Дипломна робота*
        """)

    return app


def main():
    """Run Gradio application."""
    logger.info("Starting Gradio application...")

    app = create_gradio_app()

    app.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()

