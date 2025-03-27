# Gradio Web UI for Wan2.1

This script provides a user-friendly web interface using Gradio to run the Wan2.1 text-to-video, image-to-video, and text-to-image generation models locally on your machine. It includes features like automatic model downloading, hardware detection, and optimization modes to help run the models on different systems.

**Disclaimer:** This UI script is a wrapper around the official Wan2.1 inference code (`generate.py`). Please refer to the [official Wan2.1 repository](https://github.com/Wan-Video/Wan2.1) for the model details, license, and original code.

## Features

*   **Web Interface:** Easy-to-use Gradio UI accessible via a web browser.
*   **Supported Tasks:**
    *   Text-to-Video (1.3B and 14B models)
    *   Image-to-Video (14B model - **Requires sufficient hardware**)
    *   Text-to-Image (14B model - **Requires sufficient hardware**)
*   **Hardware Detection:** Automatically detects GPU VRAM, System RAM, and available Disk Space.
*   **Minimum Requirement Checks:**
    *   Displays estimated minimum VRAM, RAM, and Disk space for both 1.3B and 14B models.
    *   Disables the 14B model option (T2V) and hides the Image-to-Video / Text-to-Image tabs if detected hardware doesn't meet the 14B requirements.
*   **Optimization Modes:** Choose between different modes to balance performance vs. resource usage:
    *   `Performance`: Fastest, highest resource usage.
    *   `Balanced (Save VRAM)`: Offloads main model (saves VRAM, uses moderate RAM).
    *   `Max VRAM Saving`: Offloads main model and T5 encoder (lowest VRAM, potentially high RAM, slowest).
    *   `Max RAM Saving`: Offloads main model, keeps T5 on GPU (attempts to save System RAM, **reduce frame count significantly**).
*   **Automatic Model Downloading:** Downloads the required Wan2.1 models from Hugging Face automatically on first use if not found locally.
*   **Parameter Control:** Adjust common generation settings like prompt, negative prompt, resolution, number of frames, guidance scale, etc.
*   **Prompt Extension (Optional):** Supports using local Qwen models or Dashscope API for prompt enhancement.
*   **Gradio Sharing:** Option to create a temporary public link (`share=True`).

## Hardware Requirements (Approximate Minimums)

These are estimates. Actual usage can vary. Running close to the minimums may lead to Out-of-Memory (OOM) errors or slow performance.

| Model                    | VRAM   | System RAM | Disk Space* | Notes                                           |
| :----------------------- | :----- | :--------- | :---------- | :---------------------------------------------- |
| **Wan2.1-T2V-1.3B**      | ~9 GB  | ~16 GB     | ~25 GB      | Performance depends heavily on actual hardware. |
| **Wan2.1-T2V/I2V/T2I-14B** | ~22 GB | ~32 GB     | ~70 GB      | **Options/Tabs disabled if requirements not met.** |

\* *Disk Space is for models, environment, and potential output buffer.*

**Important:**
*   An **NVIDIA GPU** with a compatible **CUDA Toolkit** installed on your system is required.
*   **System RAM is crucial**, especially for larger models or longer videos. Running out of System RAM (often indicated by `Code -9` errors) is common on low-RAM systems like the free tier of Google Colab.

## Installation

These instructions assume you have `git` and a Python environment manager like `conda` or `venv` installed.

1.  **Clone Official Wan2.1 Repository:**
    ```bash
    git clone https://github.com/Wan-Video/Wan2.1.git
    cd Wan2.1
    ```

2.  **Create & Activate Environment (Example using Conda):**
    ```bash
    conda create -n wan21 python=3.10 -y # Or python 3.11
    conda activate wan21
    ```

3.  **Install PyTorch:** Install PyTorch, Torchvision, and Torchaudio. Ensure the installed versions are compatible with your system's CUDA toolkit. You can typically use:
    ```bash
    pip install torch torchvision torchaudio
    ```
    *(If you encounter CUDA-related errors later, you may need to consult the [official PyTorch website](https://pytorch.org/get-started/locally/) to install versions specifically built for your system's CUDA version.)*

4.  **Install Wan2.1 Requirements:** Ensure `torch >= 2.4.0` is met after the previous step.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Additional GUI Dependencies:**
    ```bash
    pip install gradio psutil huggingface_hub "xfuser>=0.4.1" ftfy
    ```

6.  **(Google Colab ONLY) Force Reinstall Gradio/NumPy:** This step is often necessary in Colab to resolve potential NumPy version conflicts after other packages are installed. **Run this LAST.**
    ```bash
    pip install --force-reinstall --no-cache-dir gradio numpy
    ```

## How to Run

1.  **Save the Script:** Download or copy the Python script provided (let's call it `wan21_gui.py`) and place it **inside** the `Wan2.1` directory you cloned earlier (the same directory containing `generate.py`).
2.  **Activate Environment:**
    ```bash
    conda activate wan21 # Or your chosen environment name
    ```
3.  **Run the Script:**
    ```bash
    python wan21_gui.py
    ```
4.  **Access the UI:** The script will print URLs to the console. Open the local URL (usually `http://127.0.0.1:7860`) or the public Gradio URL (if sharing) in your web browser.

## Troubleshooting / Notes

*   **Model Downloads:** The first time you select a specific model (e.g., 1.3B or 14B), it will be downloaded automatically. This can take significant time and disk space (~20-60GB per model size).
*   **Out-of-Memory (OOM):**
    *   **CUDA OOM:** Error message mentions "CUDA out of memory". Try selecting a less demanding Optimization Mode (e.g., "Balanced" or "Max VRAM Saving"), reduce Resolution, or reduce Number of Frames.
    *   **System RAM OOM (Code -9):** Error message mentions "Code -9" or "Process killed". This is common on low-RAM systems (like free Colab). Select the "Max RAM Saving (Colab Focus)" Optimization Mode AND significantly reduce the **Number of Frames** (try 16-20). Close other applications consuming RAM. If it persists, the hardware may be insufficient.
*   **Check Terminal:** The terminal where you launched `python wan21_gui.py` contains detailed logs (`stdout` and `stderr`) from the underlying `generate.py` script. Always check these logs for specific error details.
*   **Prompt Extension:** Requires either setting up a local Qwen model (ensure sufficient VRAM/RAM) or obtaining and entering a Dashscope API key.

## License

The underlying Wan2.1 models and code are licensed under the Apache 2.0 License. Please refer to the [official Wan2.1 repository](https://github.com/Wan-Video/Wan2.1) for full details. This GUI script itself is provided as a helpful utility.

## Acknowledgements

*   The Wan2.1 Team for creating and open-sourcing the models and inference code.
*   The Gradio team for the easy-to-use UI library.
![image](https://github.com/user-attachments/assets/67363f41-10a2-44c5-a9cf-7266bc0aa954)
