"""
app.py — Gradio web dashboard for Medical Insurance Cost Prediction.

Run standalone: python app.py
Deploys to: Hugging Face Spaces (sdk: gradio)
"""

import gradio as gr

from predict import get_cost_category, predict_charges


# ─── Prediction Callback ──────────────────────────────────────────────────────

def run_prediction(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> tuple[str, str, str]:
    """
    Gradio event handler that calls predict_charges and formats the outputs.

    Parameters
    ----------
    age, sex, bmi, children, smoker, region — values from Gradio input widgets.

    Returns
    -------
    Tuple of three Markdown strings:
      (predicted_cost_md, category_md, advice_str)
    """
    try:
        charges  = predict_charges(age, sex, bmi, children, smoker, region)
        info     = get_cost_category(charges)

        cost_md  = f"### 💰 Predicted Cost: ${charges:,.2f} / year"
        cat_md   = f"### Cost Level: {info['category']}"
        advice   = info["advice"]

    except FileNotFoundError as e:
        # Artifacts not yet generated — guide user to run train.py first
        cost_md = "### ⚠️ Model Not Found"
        cat_md  = "### Please run `python train.py` first"
        advice  = str(e)

    except Exception as e:
        cost_md = "### ❌ Prediction Error"
        cat_md  = "### Something went wrong"
        advice  = f"Error details: {e}"

    return cost_md, cat_md, advice


# ─── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("# 🏥 Medical Insurance Cost Predictor")
    gr.Markdown(
        "Enter your personal and health details to estimate "
        "your annual medical insurance charges."
    )

    # ── Input Section ─────────────────────────────────────────────────────────
    with gr.Row():

        # Left Column — demographic & physical inputs
        with gr.Column():
            age = gr.Slider(
                minimum=18,
                maximum=64,
                value=30,
                step=1,
                label="Age",
                info="Your current age",
            )
            sex = gr.Radio(
                choices=["male", "female"],
                value="male",
                label="Sex",
            )
            bmi = gr.Slider(
                minimum=10.0,
                maximum=55.0,
                value=26.0,
                step=0.1,
                label="BMI (Body Mass Index)",
                info="Normal range: 18.5–24.9",
            )
            children = gr.Slider(
                minimum=0,
                maximum=5,
                value=0,
                step=1,
                label="Number of Children/Dependents",
            )

        # Right Column — lifestyle, region, BMI guide
        with gr.Column():
            smoker = gr.Radio(
                choices=["yes", "no"],
                value="no",
                label="Smoker?",
                info="Smoking significantly increases insurance costs",
            )
            region = gr.Dropdown(
                choices=["northeast", "northwest", "southeast", "southwest"],
                value="northeast",
                label="US Region",
            )
            bmi_info = gr.Markdown(
                "**BMI Reference:**  \n"
                "Underweight: < 18.5 &nbsp;|&nbsp; Normal: 18.5–24.9  \n"
                "Overweight: 25–29.9 &nbsp;|&nbsp; Obese: ≥ 30"
            )

    # ── Predict Button ────────────────────────────────────────────────────────
    predict_btn = gr.Button("💰 Predict Insurance Cost", variant="primary")

    # ── Output Section ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            out_cost     = gr.Markdown(label="Predicted Cost")
        with gr.Column():
            out_category = gr.Markdown(label="Cost Category")
        with gr.Column():
            out_advice   = gr.Markdown(label="Health Advice")

    # Wire button click to prediction function
    predict_btn.click(
        fn=run_prediction,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=[out_cost, out_category, out_advice],
    )

    # ── Examples Section ──────────────────────────────────────────────────────
    gr.Examples(
        label="Sample Profiles",
        examples=[
            [25, "male",   22.0, 0, "no",  "northwest"],   # Young healthy male
            [45, "female", 28.5, 2, "no",  "southeast"],   # Middle-aged female
            [35, "male",   33.0, 1, "yes", "northeast"],   # Smoker with high BMI
            [55, "female", 27.0, 3, "yes", "southwest"],   # Older smoker
        ],
        inputs=[age, sex, bmi, children, smoker, region],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        "⚠️ **Disclaimer:** This prediction is for educational purposes only "
        "and does not constitute financial or medical advice."
    )


# ─── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
