# Filename: wan21_gui.py
# Place this inside the cloned Wan2.1 directory
# Use 4 spaces for indentation consistently.

import gradio as gr
import subprocess
import os
import sys
import torch
import psutil
import math
import shlex
from huggingface_hub import snapshot_download, HfFolder
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

# --- Configuration ---
DEFAULT_MODEL_DIRS = { "T2V-14B": "./Wan2.1-T2V-14B", "T2V-1.3B": "./Wan2.1-T2V-1.3B", "I2V-14B-720P": "./Wan2.1-I2V-14B-720P", "I2V-14B-480P": "./Wan2.1-I2V-14B-480P", "T2I-14B": "./Wan2.1-T2V-14B"}
MODEL_REPO_IDS = { "T2V-14B": "Wan-AI/Wan2.1-T2V-14B", "T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B", "I2V-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P", "I2V-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",}
T2V_1_3B_RESOLUTIONS = ["832*480", "480*832"]; T2V_14B_RESOLUTIONS = ["1280*720", "720*1280", "832*480", "480*832"]; I2V_AREA_RESOLUTIONS = ["1280*720", "832*480"]; T2I_RESOLUTIONS = ["1024*1024", "1280*720", "720*1280"]

# --- Constants ---
MIN_VRAM_14B = 22.0; MIN_RAM_14B = 32.0

# --- Hardware Profiles ---
PROFILES = [
    { "name": "High-End (>=40GB VRAM, >=60GB RAM)", "min_vram": 40, "min_ram": 60, "settings_1_3B": {"resolution": "832*480", "num_frames": 81, "guidance_scale": 6.0, "sample_shift": 8, "offload": False, "t5_cpu": False, "allow_run": True}, "settings_14B": {"resolution": "1280*720", "num_frames": 81, "guidance_scale": 5.0, "offload": False, "t5_cpu": False, "allow_run": True}},
    { "name": "Consumer (>=24GB VRAM, >=32GB RAM)", "min_vram": 24, "min_ram": 32, "settings_1_3B": {"resolution": "832*480", "num_frames": 81, "guidance_scale": 6.0, "sample_shift": 8, "offload": False, "t5_cpu": False, "allow_run": True}, "settings_14B": {"resolution": "1280*720", "num_frames": 60, "guidance_scale": 5.0, "offload": True, "t5_cpu": False, "allow_run": True}},
    { "name": "Colab (~15GB VRAM, ~12.7GB RAM)", "min_vram": 14.5, "min_ram": 12, "settings_1_3B": {"resolution": "832*480", "num_frames": 20, "guidance_scale": 6.0, "sample_shift": 10, "offload": True, "t5_cpu": True, "allow_run": True}, "settings_14B": {"resolution": "832*480", "num_frames": 16, "guidance_scale": 5.0, "offload": True, "t5_cpu": True, "allow_run": False}},
    { "name": "Minimum (>=10GB VRAM, >=16GB RAM)", "min_vram": 10, "min_ram": 16, "settings_1_3B": {"resolution": "832*480", "num_frames": 40, "guidance_scale": 6.0, "sample_shift": 12, "offload": True, "t5_cpu": True, "allow_run": True}, "settings_14B": {"resolution": "832*480", "num_frames": 16, "guidance_scale": 5.0, "offload": True, "t5_cpu": True, "allow_run": False}},
    { "name": "Unsupported (<10GB VRAM or <16GB RAM)", "min_vram": 0, "min_ram": 0, "settings_1_3B": {"resolution": "832*480", "num_frames": 16, "guidance_scale": 6.0, "sample_shift": 12, "offload": True, "t5_cpu": True, "allow_run": False}, "settings_14B": {"resolution": "832*480", "num_frames": 16, "guidance_scale": 5.0, "offload": True, "t5_cpu": True, "allow_run": False}},]

# --- Prerequisite Checks ---
if not os.path.exists("generate.py"): print("ERROR: Script must run in Wan2.1 directory."); sys.exit(1)
if not HfFolder.get_token(): print("INFO: Hugging Face token not found. Downloads anonymous.")

# --- Hardware Detection ---
def get_hardware_info():
    info = {"gpu_name": "N/A", "vram_gb": 0, "ram_gb": 0}
    try:
        if torch.cuda.is_available(): props = torch.cuda.get_device_properties(0); info['gpu_name'] = props.name; info['vram_gb'] = round(props.total_memory / (1024**3), 2)
        else: info['gpu_name'] = "No NVIDIA GPU / CUDA not available"
    except Exception as e: print(f"GPU Error: {e}"); info['gpu_name'] = "GPU Error"
    try: info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception as e: print(f"RAM Error: {e}")
    return info

# --- Hardware Info & Guidance ---
detected_hw = get_hardware_info()
can_run_14b = (detected_hw["vram_gb"] >= MIN_VRAM_14B and detected_hw["ram_gb"] >= MIN_RAM_14B)
def create_initial_info_markdown(hw_info, can_run_14b_flag):
    vram = hw_info['vram_gb']; ram = hw_info['ram_gb']; gpu_name = hw_info['gpu_name']
    guidance = f"**Detected Hardware:** {gpu_name} ({vram} GB VRAM), {ram} GB System RAM.\n\n"
    guidance += f"**Note:** 14B model requires approx. >= {MIN_VRAM_14B} GB VRAM AND >= {MIN_RAM_14B} GB System RAM.\n"
    if not can_run_14b_flag: guidance += "**-> 14B model options DISABLED.**\n"
    guidance += "\n**Optimization Modes Explained:**\n"
    guidance += "*   **Performance:** Fastest, uses most VRAM & RAM.\n"
    guidance += "*   **Balanced (Save VRAM):** Offloads main model. Good if VRAM limited but RAM okay (>=32GB).\n"
    guidance += "*   **Max VRAM Saving:** Offloads main model AND text encoder (T5). Lowest VRAM, slowest, uses more RAM than 'Balanced'.\n"
    guidance += "*   **Max RAM Saving (Colab Focus):** Offloads main model, keeps T5 on GPU. **Use if getting System RAM crashes (Code -9).** Also reduce 'Frames'.\n"
    return guidance
initial_info_markdown = create_initial_info_markdown(detected_hw, can_run_14b)

# --- Model Directory Check ---
def check_model_dirs_startup(model_dirs):
    not_downloaded = []; unique_dirs = set(model_dirs.values())
    for path in unique_dirs:
        if not os.path.isdir(path): models_using_path = [name for name, dir_path in model_dirs.items() if dir_path == path]; not_downloaded.append(f"- {' / '.join(models_using_path)}: Not found (DL on use)")
    if not_downloaded: return "### ⏳ Model Status\n" + "\n".join(not_downloaded)
    return "✅ All default model directories seem present."
model_status_markdown = check_model_dirs_startup(DEFAULT_MODEL_DIRS)

# --- Generation Logic ---
def run_generation(
    task_type, model_name, optimization_mode,
    prompt, negative_prompt, input_image, resolution,
    num_frames, guidance_scale, sample_shift,
    use_prompt_extend, prompt_extend_method, prompt_extend_model, dash_api_key,
    progress=gr.Progress(track_tqdm=True)
):
    # Input validation
    if not prompt and task_type not in ['i2v']: raise gr.Error("Prompt is required.")
    if task_type == 'i2v' and input_image is None: raise gr.Error("Input image required.")

    # Determine Optimization Flags
    offload_model = False; t5_cpu = False
    if optimization_mode == "Balanced (Save VRAM)": offload_model = True
    elif optimization_mode == "Max VRAM Saving": offload_model = True; t5_cpu = True
    elif optimization_mode == "Max RAM Saving (Colab Focus)": offload_model = True

    # Auto-Download Logic
    target_model_key = "T2V-14B" if task_type == 't2i' else model_name
    ckpt_dir = DEFAULT_MODEL_DIRS.get(target_model_key); repo_id = MODEL_REPO_IDS.get(target_model_key)
    if not ckpt_dir or not repo_id: raise gr.Error(f"Config error: Model '{model_name}' undefined.")
    if not os.path.isdir(ckpt_dir):
        gr.Info(f"Model '{target_model_key}' downloading..."); print(f"DL '{repo_id}'->'{ckpt_dir}'...")
        enable_progress_bars();
        try:
            snapshot_download(repo_id=repo_id, local_dir=ckpt_dir, local_dir_use_symlinks=False, token=HfFolder.get_token())
            print(f"DL complete: '{repo_id}' -> '{ckpt_dir}'"); gr.Info(f"Model '{target_model_key}' downloaded.")
        except Exception as e: print(f"DL Error: {e}"); raise gr.Error(f"DL fail. Error: {e}")
        finally: disable_progress_bars()

    # Determine Task Argument
    if task_type == 't2v': task_arg = f"t2v-{model_name.split('-')[1]}"
    elif task_type == 'i2v': task_arg = f"i2v-{model_name.split('-')[1]}"
    elif task_type == 't2i': task_arg = f"t2i-{model_name.split('-')[1]}"
    else: raise gr.Error(f"Invalid task type: {task_type}")

    # Build command
    cmd = [sys.executable, "generate.py", "--task", task_arg, "--ckpt_dir", ckpt_dir]
    if prompt: cmd.extend(["--prompt", prompt])
    if negative_prompt: cmd.extend(["--neg_prompt", negative_prompt])
    if not resolution or '*' not in resolution: raise gr.Error("Invalid resolution W*H.")
    cmd.extend(["--size", resolution])
    if task_type == 'i2v':
        if not input_image or not hasattr(input_image, 'name') or not os.path.exists(input_image.name): raise gr.Error(f"Invalid input image.")
        cmd.extend(["--image", input_image.name])
    elif task_type == 't2i': cmd.extend(["--frame_num", "1"])
    else: cmd.extend(["--frame_num", str(int(num_frames))])
    cmd.extend(["--sample_guide_scale", str(guidance_scale)])
    if task_type == 't2v' and "1.3B" in model_name and sample_shift is not None: cmd.extend(["--sample_shift", str(int(sample_shift))])
    if offload_model: cmd.extend(["--offload_model", "True"])
    if t5_cpu: cmd.append("--t5_cpu")
    if use_prompt_extend:
        cmd.append("--use_prompt_extend")
        if prompt_extend_method: cmd.extend(["--prompt_extend_method", prompt_extend_method])
        if prompt_extend_model: cmd.extend(["--prompt_extend_model", prompt_extend_model])
    env = os.environ.copy();
    if use_prompt_extend and prompt_extend_method == 'dashscope':
        if not dash_api_key: raise gr.Error("Dashscope API Key required.")
        env['DASH_API_KEY'] = dash_api_key

    # Subprocess Execution
    output_dir = os.path.join(ckpt_dir, "samples"); os.makedirs(output_dir, exist_ok=True)
    print("--- Running Command ---\n", " ".join([shlex.quote(str(c)) for c in cmd]), "\n-----------------------")
    process = None; stdout = ""; stderr = ""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=env)
        stdout, stderr = process.communicate()
        print("--- Subprocess Raw Output ---"); print("Return Code:", process.returncode); print("[STDOUT]:\n", stdout); print("[STDERR]:\n", stderr); print("----------------------------")
        if process.returncode != 0:
            error_prefix = f"Generation failed (Code {process.returncode})."; error_suffix = "\n\n**Check terminal output for full logs.**"; specific_error = ""
            if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr: specific_error = "CUDA OOM! Reduce res/frames or use different Opt Mode."
            elif "AssertionError: Unsupport size" in stderr: specific_error = f"Resolution Error: {stderr.split('AssertionError:')[-1].strip()}."
            elif "ImportError" in stderr or "ModuleNotFoundError" in stderr: specific_error = f"Import Error ({stderr.splitlines()[-1]}). Check deps."
            elif "RuntimeError: Detected that PyTorch and torchvision" in stderr: specific_error = "PyTorch/Torchvision CUDA mismatch. Reinstall."
            elif "expected one argument" in stderr: specific_error = f"Arg Error: {stderr.split('error:')[-1].strip()}. Check flags."
            elif "unrecognized arguments" in stderr: specific_error = f"Arg Error: {stderr.split('error:')[-1].strip()}. Check flags."
            elif process.returncode == -9: specific_error = "Process killed (Signal 9 - Likely System RAM OOM). Try 'Max RAM Saving', reduce Frames, close apps. Check dmesg."
            else: last_stderr_line = stderr.strip().split('\n')[-1] if stderr.strip() else "No stderr."; specific_error = f"See terminal logs. Last stderr: '{last_stderr_line}'"
            raise gr.Error(f"{error_prefix} {specific_error}{error_suffix}")
        list_of_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        if not list_of_files: raise gr.Error(f"No output file in {output_dir}.")
        latest_file = max(list_of_files, key=os.path.getctime); print(f"Output file: {latest_file}")
        return latest_file
    except FileNotFoundError: raise gr.Error(f"Execution failed: '{sys.executable} generate.py' not found.")
    except Exception as e:
        print(f"GUI script Error: {e}")
        if process and (stdout or stderr): print("--- Partial Output ---"); print("RC:", process.returncode); print("STDOUT:", stdout); print("STDERR:", stderr); print("--------------------")
        raise e if isinstance(e, gr.Error) else gr.Error(f"Unexpected GUI error: {e}")

# --- Gradio UI Update Function ---
def update_t2v_model_options(model_choice, hw_info):
    allow_run = True; warning_msg = ""
    can_run_selected_14b = (hw_info["vram_gb"] >= MIN_VRAM_14B and hw_info["ram_gb"] >= MIN_RAM_14B)
    if model_choice == "T2V-14B" and not can_run_selected_14b: allow_run = False; warning_msg = f"⚠️ 14B disabled (Req >= {MIN_VRAM_14B}GB VRAM & >= {MIN_RAM_14B}GB RAM)."
    if model_choice == "T2V-1.3B": allowed_resolutions = T2V_1_3B_RESOLUTIONS; default_res = allowed_resolutions[0]
    elif model_choice == "T2V-14B": allowed_resolutions = T2V_14B_RESOLUTIONS; default_res = T2V_14B_RESOLUTIONS[0]
    else: allowed_resolutions = []; default_res = None; allow_run = False; warning_msg = "⚠️ Invalid model."
    if default_res not in allowed_resolutions: default_res = allowed_resolutions[0] if allowed_resolutions else None
    return [gr.update(choices=allowed_resolutions, value=default_res), gr.update(interactive=allow_run), gr.update(value=warning_msg, visible=not allow_run)]

# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Wan2.1 Video Generation GUI"); gr.Markdown(initial_info_markdown); gr.Markdown(model_status_markdown)
    with gr.Tabs():
        # == Text-to-Video Tab ==
        with gr.TabItem("Text-to-Video"):
            t2v_model_choices = ["T2V-1.3B"] + (["T2V-14B"] if can_run_14b else [])
            initial_t2v_model = "T2V-14B" if can_run_14b else "T2V-1.3B"
            initial_t2v_res = T2V_14B_RESOLUTIONS if can_run_14b else T2V_1_3B_RESOLUTIONS
            initial_t2v_res_val = initial_t2v_res[0] if initial_t2v_res else None
            default_opt_mode = "Max RAM Saving (Colab Focus)" if detected_hw["ram_gb"] < 16 else "Balanced (Save VRAM)"
            with gr.Row():
                with gr.Column(scale=2):
                    t2v_model=gr.Dropdown(t2v_model_choices, label="1. Model", value=initial_t2v_model)
                    t2v_prompt=gr.Textbox(lines=3, label="2. Prompt"); t2v_neg_prompt=gr.Textbox(lines=2, label="3. Negative Prompt (Opt.)")
                    t2v_optim_mode = gr.Radio(["Performance", "Balanced (Save VRAM)", "Max VRAM Saving", "Max RAM Saving (Colab Focus)"], label="4. Optimization Mode", value=default_opt_mode)
                    with gr.Group():
                        gr.Markdown("**5. Generation Settings**")
                        t2v_resolution=gr.Dropdown(choices=initial_t2v_res, value=initial_t2v_res_val, label="Resolution (W*H)")
                        t2v_num_frames=gr.Slider(label="Frames", minimum=16, maximum=128, value=20 if detected_hw["ram_gb"] < 16 else 40, step=1, info="Reduce if RAM OOM")
                        t2v_guidance_scale=gr.Slider(label="Guidance (CFG)", minimum=1.0, maximum=15.0, value=6.0, step=0.5)
                        t2v_sample_shift=gr.Slider(label="Sample Shift (1.3B only)", minimum=8, maximum=12, value=10, step=1)
                    # Corrected Accordion Indentation
                    with gr.Accordion("6. Prompt Extension (Optional)", open=False):
                        t2v_use_pe=gr.Checkbox(label="Enable PE", value=False)
                        with gr.Row(visible=False) as t2v_pe_options:
                            t2v_pe_method=gr.Radio(["local_qwen", "dashscope"], label="Method", value="local_qwen")
                            t2v_pe_model=gr.Textbox(label="Model")
                            t2v_dash_key=gr.Textbox(label="API Key", type="password")
                        t2v_use_pe.change(lambda x: gr.update(visible=x), inputs=t2v_use_pe, outputs=t2v_pe_options)
                with gr.Column(scale=1):
                    t2v_output=gr.Video(label="Output"); t2v_warning_msg=gr.Markdown("", visible=(initial_t2v_model == "T2V-14B" and not can_run_14b)); t2v_button=gr.Button("Generate Video", variant="primary")
            t2v_model.change(fn=update_t2v_model_options, inputs=[t2v_model, gr.State(detected_hw)], outputs=[t2v_resolution, t2v_button, t2v_warning_msg])

        # == Image-to-Video Tab ==
        with gr.TabItem("Image-to-Video"):
             with gr.Row():
                with gr.Column(scale=2):
                    i2v_model=gr.Dropdown(["I2V-14B-720P", "I2V-14B-480P"], label="1. Model", value="I2V-14B-720P"); i2v_input_image=gr.Image(type="filepath", label="2. Input Image"); i2v_prompt=gr.Textbox(lines=3, label="3. Prompt (Opt.)"); i2v_neg_prompt=gr.Textbox(lines=2, label="4. Negative Prompt (Opt.)")
                    i2v_optim_mode = gr.Radio(["Performance", "Balanced (Save VRAM)", "Max VRAM Saving", "Max RAM Saving (Colab Focus)"], label="5. Optimization Mode", value="Balanced (Save VRAM)")
                    with gr.Group(): gr.Markdown("**6. Settings**"); i2v_resolution_area=gr.Dropdown(I2V_AREA_RESOLUTIONS, label="Target Area (W*H)", value=I2V_AREA_RESOLUTIONS[0], info="Aspect ratio from input"); i2v_num_frames=gr.Slider(label="Frames", minimum=16, maximum=128, value=81, step=1); i2v_guidance_scale=gr.Slider(label="Guidance (CFG)", minimum=1.0, maximum=15.0, value=5.0, step=0.5)
                    # Corrected Accordion Indentation
                    with gr.Accordion("7. Prompt Extension (Optional)", open=False):
                        i2v_use_pe=gr.Checkbox(label="Enable PE", value=False)
                        with gr.Row(visible=False) as i2v_pe_options:
                            i2v_pe_method=gr.Radio(["local_qwen", "dashscope"], label="Method", value="local_qwen")
                            i2v_pe_model=gr.Textbox(label="Model")
                            i2v_dash_key=gr.Textbox(label="API Key", type="password")
                        i2v_use_pe.change(lambda x: gr.update(visible=x), inputs=i2v_use_pe, outputs=i2v_pe_options)
                with gr.Column(scale=1): i2v_output=gr.Video(label="Output"); i2v_button=gr.Button("Generate Video from Image", variant="primary")

        # == Text-to-Image Tab ==
        with gr.TabItem("Text-to-Image"):
             with gr.Row():
                with gr.Column(scale=2):
                    t2i_model_info=gr.Markdown("Uses **T2V-14B** model."); t2i_prompt=gr.Textbox(lines=3, label="1. Prompt"); t2i_neg_prompt=gr.Textbox(lines=2, label="2. Negative Prompt (Opt.)")
                    t2i_optim_mode = gr.Radio(["Performance", "Balanced (Save VRAM)", "Max VRAM Saving", "Max RAM Saving (Colab Focus)"], label="3. Optimization Mode", value="Balanced (Save VRAM)")
                    with gr.Group(): gr.Markdown("**4. Settings**"); t2i_resolution=gr.Dropdown(T2I_RESOLUTIONS, label="Resolution (W*H)", value=T2I_RESOLUTIONS[0]); t2i_guidance_scale=gr.Slider(label="Guidance (CFG)", minimum=1.0, maximum=15.0, value=5.0, step=0.5)
                    # Corrected Accordion Indentation
                    with gr.Accordion("5. Prompt Extension (Optional)", open=False):
                        t2i_use_pe=gr.Checkbox(label="Enable PE", value=False)
                        with gr.Row(visible=False) as t2i_pe_options:
                            t2i_pe_method=gr.Radio(["local_qwen", "dashscope"], label="Method", value="local_qwen")
                            t2i_pe_model=gr.Textbox(label="Model")
                            t2i_dash_key=gr.Textbox(label="API Key", type="password")
                        t2i_use_pe.change(lambda x: gr.update(visible=x), inputs=t2i_use_pe, outputs=t2i_pe_options)
                with gr.Column(scale=1): t2i_output=gr.Image(label="Output", type="filepath"); t2i_button=gr.Button("Generate Image", variant="primary")

    # --- Button Click Handlers --- (Updated Inputs)
    t2v_button.click(fn=run_generation, inputs=[gr.State('t2v'), t2v_model, t2v_optim_mode, t2v_prompt, t2v_neg_prompt, gr.State(None), t2v_resolution, t2v_num_frames, t2v_guidance_scale, t2v_sample_shift, t2v_use_pe, t2v_pe_method, t2v_pe_model, t2v_dash_key], outputs=t2v_output)
    i2v_button.click(fn=run_generation, inputs=[gr.State('i2v'), i2v_model, i2v_optim_mode, i2v_prompt, i2v_neg_prompt, i2v_input_image, i2v_resolution_area, i2v_num_frames, i2v_guidance_scale, gr.State(None), i2v_use_pe, i2v_pe_method, i2v_pe_model, i2v_dash_key], outputs=i2v_output)
    t2i_button.click(fn=run_generation, inputs=[gr.State('t2i'), gr.State("T2V-14B"), t2i_optim_mode, t2i_prompt, t2i_neg_prompt, gr.State(None), t2i_resolution, gr.State(1), t2i_guidance_scale, gr.State(None), t2i_use_pe, t2i_pe_method, t2i_pe_model, t2i_dash_key], outputs=t2i_output)

# --- Launch ---
if __name__ == "__main__":
    demo.launch(share=True, debug=True)