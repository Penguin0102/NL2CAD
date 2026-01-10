import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch


def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
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

        text2cad.load_state_dict(pretrained_dict, strict=False)
    text2cad.eval()
    return text2cad

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



config_path = "../Cad_VLM/config/inference_user_input.yaml"
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


examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "The CAD model features a cylindrical object with a cylindrical hole in the center.",
    "The CAD model features a rectangular metal plate with four holes along its length.",
]

# Custom CSS for Industrial/B2B SaaS look
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

:root {
    --bg-color: #0a0a0c;
    --sidebar-bg: #111114;
    --accent-color: #00d2ff;
    --accent-glow: rgba(0, 210, 255, 0.4);
    --text-primary: #e0e0e0;
    --text-secondary: #888888;
    --border-color: #222226;
}

body, .gradio-container {
    background-color: var(--bg-color) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

#sidebar {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
    padding: 2.5rem;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

#viewport-container {
    position: relative;
    height: 100vh;
    background: radial-gradient(circle at center, #1a1a1e 0%, #0a0a0c 100%);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

#viewport-container::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(var(--border-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--border-color) 1px, transparent 1px);
    background-size: 60px 60px;
    background-position: center;
    opacity: 0.15;
    pointer-events: none;
}

.nl2cad-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    margin-bottom: 3rem;
    color: #fff;
}

.nl2cad-title span {
    color: var(--accent-color);
}

.design-prompt-label {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    letter-spacing: 0.1em;
}

.synthesize-btn {
    background: var(--accent-color) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 0 25px var(--accent-glow) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 1.5rem !important;
    border-radius: 8px !important;
    height: 50px !important;
}

.synthesize-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px var(--accent-glow) !important;
    filter: brightness(1.1);
}

.recent-templates-section {
    margin-top: auto;
    padding-top: 2rem;
}

.template-tag {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background: #1a1a1e;
    border: 1px solid var(--border-color);
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-right: 0.6rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.template-tag:hover {
    border-color: var(--accent-color);
    color: var(--accent-color);
    background: rgba(0, 210, 255, 0.05);
}

.export-fab {
    position: absolute !important;
    bottom: 2.5rem !important;
    right: 2.5rem !important;
    z-index: 100 !important;
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #fff !important;
    padding: 0.8rem 1.8rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    width: auto !important;
}

.export-fab:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Viewport styling */
.model-viewer {
    background: transparent !important;
    border: none !important;
    height: 80vh !important;
}

/* Hide Gradio defaults */
footer { display: none !important; }
.gr-prose { display: none !important; }
.gr-button-secondary { background: #1a1a1e !important; border-color: var(--border-color) !important; }
"""


def _wrapped_generate(text):
    try:
        path = genrate_cad_model_from_text(text)
        return path, f"✅ Generated: {text}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def export_model(model_path):
    if model_path and os.path.exists(model_path):
        return model_path
    return None

with gr.Blocks(css=custom_css, title="NL2CAD") as demo:
    with gr.Row(equal_height=True):
        # Left Panel: Sidebar
        with gr.Column(scale=4, elem_id="sidebar"):
            gr.HTML(
                """
                <div class="nl2cad-title">NL2<span>CAD</span></div>
                <div class="design-prompt-label">Design Prompt</div>
                """
            )
            
            input_text = gr.Textbox(
                show_label=False,
                placeholder="Enter a description (e.g., 'A mechanical gear with 12 teeth')...",
                lines=6,
                elem_id="prompt-input"
            )
            
            generate_btn = gr.Button("Synthesize", elem_classes="synthesize-btn")
            
            status_md = gr.Markdown("Ready to synthesize.", elem_id="status-text")
            
            with gr.Column(elem_classes="recent-templates-section"):
                gr.HTML("<div class=\"design-prompt-label\">Recent Templates</div>")
                with gr.Row():
                    template_1 = gr.Button("Mechanical Gear", elem_classes="template-tag")
                    template_2 = gr.Button("Cylindrical Bolt", elem_classes="template-tag")
                    template_3 = gr.Button("Star Prism", elem_classes="template-tag")
                    template_4 = gr.Button("Hollow Ring", elem_classes="template-tag")

        # Right Panel: Viewport
        with gr.Column(scale=9, elem_id="viewport-container"):
            output_model = gr.Model3D(
                show_label=False,
                elem_classes="model-viewer",
                clear_color=[0, 0, 0, 0] # Transparent to show the gradient background
            )
            
            export_btn = gr.Button("↓ Export STEP/STL", elem_classes="export-fab")
            
            # Hidden component to handle file download
            export_file = gr.File(label="Download Model", visible=False)

    # Interactions
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
    
    # Template clicks
    template_1.click(fn=lambda: "A mechanical gear with 12 teeth", outputs=input_text)
    template_2.click(fn=lambda: "A cylindrical bolt with a hexagonal head", outputs=input_text)
    template_3.click(fn=lambda: "A 3D star shape with 5 points", outputs=input_text)
    template_4.click(fn=lambda: "A hollow ring with a smooth finish", outputs=input_text)

    # Export interaction
    export_btn.click(
        fn=export_model,
        inputs=output_model,
        outputs=export_file
    )
    
    # Trigger download when file is ready
    export_file.change(fn=None, js="() => { document.querySelector('#viewport-container a[download]').click(); }")



if __name__ == "__main__":
    demo.launch(share=True)