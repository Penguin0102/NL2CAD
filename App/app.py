import os, sys

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from App)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Cad_VLM.models.nl2cad import nl2cad
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch


def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    nl2cad = nl2cad(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value

        nl2cad.load_state_dict(pretrained_dict, strict=False)
    nl2cad.eval()
    return nl2cad

def test_model(model, text, config, device):
    
    if not isinstance(text, list):
        text = [text]
    pred_cad_seq_dict = model.test_decode(
        texts=text,
        maxlen=MAX_CAD_SEQUENCE_LENGTH,
        nucleus_prob=0,
        topk_index=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        pred_cad = CADSequence.from_vec(
            pred_cad_seq_dict["cad_vec"][0].cpu().numpy(),
            bit=N_BIT,
            post_processing=True,
        ).create_mesh()

        return pred_cad.mesh, pred_cad
    except Exception as e:
        return None

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data



config_path = os.path.join(project_root, "Cad_VLM", "config", "inference_user_input.yaml")
config = parse_config_file(config_path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_model(config, device)
OUTPUT_DIR="output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def genrate_cad_model_from_text(text):
    global model, config
    mesh,*extra = test_model(model=model, text=text, config=config, device=device)
    if mesh is not None:
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path)
        return output_path
    else:
        raise Exception("Error generating CAD model from text")


# examples = [
#     "A ring.",
#     "A rectangular prism.",
#     "A 3D star shape with 5 points.",
#     "The CAD model features a cylindrical object with a cylindrical hole in the center.",
#     "The CAD model features a rectangular metal plate with four holes along its length."
# ]

# title = "nl2cad: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts"
# description = """
# Generate 3D CAD models from text prompts of varying complexity, from beginner-level descriptions to expert-level specifications.

# <div style="display: flex; justify-content: center; gap: 10px; align-items: center;">

# <a href="https://arxiv.org/abs/2409.17106">
#   <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv" />
# </a>
# <a href="https://sadilkhan.github.io/nl2cad-project/">
#   <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
# </a>
# <a href="https://huggingface.co/datasets/SadilKhan/nl2cad">
#   <img src="https://img.shields.io/badge/Dataset-7D5BA6?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Dataset" />
# </a>

# </div>
# """

# # Create the Gradio interface
# demo = gr.Interface(
#     fn=genrate_cad_model_from_text,
#     inputs=gr.Textbox(label="Text", placeholder="Enter a text prompt here"),
#     outputs=gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"),
#     examples=examples,
#     title=title,
#     description=description,
#     theme=gr.themes.Soft(), 
# )

# if __name__ == "__main__":
#     demo.launch(share=True)

examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "The CAD model features a cylindrical object with a cylindrical hole in the center.",
    "The CAD model features a rectangular metal plate with four holes along its length.",
]

title = "nl2cad: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts"

description = """
Generate 3D CAD models from text prompts of varying complexity, from beginner-level descriptions to expert-level specifications.
"""

# 自定义页�?CSS：暗色背�?+ 卡片风格 + 高亮按钮
custom_css = """
body {
    background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000 100%);
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding-top: 1.5rem !important;
}

#nl2cad-header {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}

#nl2cad-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #a855f7);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 0.3rem;
}

#nl2cad-header p {
    font-size: 0.95rem;
    color: #9ca3af;
    max-width: 720px;
    margin: 0.2rem auto 0;
}

.nl2cad-card {
    background: rgba(15, 23, 42, 0.9);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 18px 50px rgba(0,0,0,0.55);
    padding: 1.2rem;
}

.nl2cad-label {
    font-weight: 600 !important;
    color: #e5e7eb !important;
}

.nl2cad-footer {
    text-align: center;
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.75rem;
}

/* 按钮高亮 */
button.primary, button.svelte-1ipelgc, .nl2cad-generate-btn button {
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    color: #0f172a !important;
    border-radius: 9999px !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 25px rgba(34,197,94,0.35);
}

button.primary:hover,
.nl2cad-generate-btn button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

/* 3D 模型区域边框 */
.nl2cad-model3d .wrap {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    overflow: hidden;
}
"""


# 包一层，返回路径 + 状态文�?
def _wrapped_generate(text):
    path = genrate_cad_model_from_text(text)
    return path, f"�?Successfully generated CAD model for: **{text}**"



with gr.Blocks(css=custom_css) as demo:
    gr.HTML(
        """
        <div id="nl2cad-header">
            <h1>基于文本�?3D CAD 生成 Demo</h1>
            <p>
                这是我们自主实验的演示页面：输入一段自然语言描述，系统会自动生成对应�?3D CAD 模型�?
                模型在现有开源基线的基础上进行了调整和改进，用于课程实验与展示�?
            </p>
        </div>
        """
    )

    with gr.Row():
        # 左侧：输�?+ 示例
        with gr.Column(scale=5):
            with gr.Group(elem_classes="nl2cad-card"):
                input_text = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe the CAD shape you want to generate...",
                    lines=4,
                    elem_classes="nl2cad-label",
                )
                with gr.Row(elem_classes="nl2cad-generate-btn"):
                    generate_btn = gr.Button("Generate CAD Model", scale=3)
                status_md = gr.Markdown("👈 Enter a prompt on the left and click **Generate**.")

            gr.Markdown("**Examples**", elem_classes="nl2cad-label")

            # 这里删除 elem_classes 参数，保持老版�?gradio 兼容
            gr.Examples(
                examples=examples,
                inputs=input_text,
            )

        # 右侧�?D 查看�?
        with gr.Column(scale=7):
            with gr.Group(elem_classes="nl2cad-card nl2cad-model3d"):
                output_model = gr.Model3D(
                    label="Generated 3D CAD Model",
                    # R, G, B, A  0~1 之间，下面是很浅的蓝白背�?
                    clear_color=[0.96, 0.97, 0.99, 1.0],
                )

    gr.Markdown(

    )

    # 绑定交互：点击按�?�?回车触发
    generate_btn.click(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, status_md],
    )
    input_text.submit(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, status_md],
    )


if __name__ == "__main__":
    demo.launch(share=True)
