# VCAR Installation and Usage

VCAR is an interactive 3D Gaussian Splatting segmentation tool. It renders 3DGS training views, sends the rendered frames to a local SAM3 video segmentation service, projects the 2D masks back to 3D Gaussian points, and saves segmented foreground/background PLY files.

The main workflow is:

1. Start the SAM3 API server.
2. Start the Gradio interface.
3. Load a trained 3DGS model path.
4. Select a rendered view and annotate with text, points, or box.
5. Run two-round 3DGS segmentation and optional post-processing.

## Environment

The tested environment is:

- Python 3.12
- CUDA 11.8
- PyTorch 2.7.0 with CUDA 11.8

```shell
conda create -n VCAR python=3.12 -y
conda activate VCAR
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Install SAM3

Clone SAM3 into this project root and install it in editable mode.

```shell
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout a51b9f498c84824a94702cc289ed75d9cc544c64
pip install -e .
pip install modelscope
modelscope download --model facebook/sam3 --local_dir ./weights
cp ../sam3_segment_api.py ./sam3_segment_api.py
cd ..
```

Run `sam3_segment_api.py` from inside `sam3`, so `./weights/sam3.pt` resolves correctly.

## Install Gaussian Splatting Dependencies

Clone the official Gaussian Splatting repository and rename it to `gaussiansplatting`.

```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
mv gaussian-splatting gaussiansplatting
git -C gaussiansplatting checkout 54c035f7834b564019656c3e3fcc3646292f727d
git -C gaussiansplatting submodule update --init --recursive
cd gaussiansplatting/submodules
pip install --no-build-isolation ./diff-gaussian-rasterization
pip install --no-build-isolation ./simple-knn
cd ../..
```

## Install Python Packages

```shell
pip install -r requirements.txt
```


## Usage

### 1. Start the SAM3 API server

Open a terminal and run:

```shell
conda activate VCAR
cd /path/to/VCAR/sam3
python sam3_segment_api.py
```

The server runs at:

```text
http://localhost:8000
```

You can check the API page at:

```text
http://localhost:8000/docs
```

### 2. Start the Gradio interface

Open another terminal and run:

```shell
conda activate VCAR
cd /path/to/VCAR
export PYTHONPATH="$PWD:$PWD/gaussiansplatting:$PYTHONPATH"
# Optional:
# export VCAR_DEFAULT_MODEL_PATH="/path/to/trained-3dgs-output"
# export VCAR_DEFAULT_SOURCE_PATH="/path/to/source-scene"
# export VCAR_ALLOWED_PATHS="/path/to/datasets:/path/to/outputs"
python seg_iter_gradio.py
```

The Gradio interface runs at:

```text
http://localhost:7860
```

## Input Model Path

In the Gradio interface, set both the trained 3DGS model path and its original
source scene path. The explicit source path overrides the machine-specific
absolute `source_path` often stored in `cfg_args`.

The model directory should contain:

```text
cfg_args
point_cloud/iteration_30000/point_cloud.ply
```

For example:

```text
/path/to/trained-3dgs-output
```

## Outputs

Each segmentation run creates an output folder under the model path:

```text
<model_path>/<object_name>-YYYY-MM-DD-HH-MM/
```

Important output files include:

```text
coarse.ply
fine.ply
segment.ply
background.ply
compression.ply
outlier.ply
segmentation_params.json
multiview_masks.pkl
render_segment/
render_mask_segment/
```

`segment.ply` is always the latest foreground segmentation result.

## Notes

- Text prompts and point/box prompts are not mixed in one SAM3 API request.
- The SAM3 API must keep running while Gradio performs segmentation.
- If the SAM3 server cannot find weights, confirm that `sam3/weights/sam3.pt` exists and that the server was started from the `sam3` directory.
- If Python cannot import Gaussian Splatting modules, confirm that `PYTHONPATH` includes both the project root and `gaussiansplatting`.
- The project assumes CUDA is available for 3DGS rendering and mask projection.
