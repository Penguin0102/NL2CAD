import os

with open('App/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add GRADIO_OFFLINE at the top
if 'import os' in content and 'os.environ["GRADIO_OFFLINE"]' not in content:
    content = content.replace('import os, sys', 'import os, sys\nos.environ["GRADIO_OFFLINE"] = "1"')

# 2. Update _wrapped_generate to return both path and status
old_wrapped = '''def _wrapped_generate(text):
    path = genrate_cad_model_from_text(text)
    return path, f"✅ Successfully generated CAD model for: **{text}**"'''

new_wrapped = '''def _wrapped_generate(text):
    try:
        path = genrate_cad_model_from_text(text)
        return path, path, f"✅ Successfully generated CAD model for: **{text}**"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"'''

content = content.replace(old_wrapped, new_wrapped)

# 3. Update Blocks to add a File component for download and fix outputs
old_blocks = '''        # 右侧：3D 查看器
        with gr.Column(scale=7):
            with gr.Group(elem_classes="nl2cad-card nl2cad-model3d"):
                output_model = gr.Model3D(
                    label="Generated 3D CAD Model",
                    # R, G, B, A  0~1 之间，下面是很浅的蓝白背景
                    clear_color=[0.96, 0.97, 0.99, 1.0],
                )

    gr.Markdown(

    )

    # 绑定交互：点击按钮 或 回车触发
    generate_btn.click(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, status_md],
    )
    input_text.submit(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, status_md],
    )'''

new_blocks = '''        # 右侧：3D 查看器
        with gr.Column(scale=7):
            with gr.Group(elem_classes="nl2cad-card nl2cad-model3d"):
                output_model = gr.Model3D(
                    label="Generated 3D CAD Model",
                    clear_color=[0.1, 0.1, 0.1, 1.0],
                )
                download_file = gr.File(label="Download STL Model")

    gr.Markdown(
        """
        <p style="text-align: center; color: #6b7280; font-size: 0.8rem;">
            Note: If the 3D preview doesn't load, please download the STL file to view it locally.
        </p>
        """
    )

    # 绑定交互：点击按钮 或 回车触发
    generate_btn.click(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, download_file, status_md],
    )
    input_text.submit(
        _wrapped_generate,
        inputs=input_text,
        outputs=[output_model, download_file, status_md],
    )'''

content = content.replace(old_blocks, new_blocks)

with open('App/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done! Added offline mode and download fallback.')
