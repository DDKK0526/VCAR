import os
import gc
import uvicorn

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# SAM3 builders
from sam3.model_builder import (
    build_sam3_video_predictor,
    build_sam3_video_model,
)

# ====================Device Settings ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ==================== Lazy Loading ====================
VIDEO_CKPT_PATH = "./weights/sam3.pt"  # checkpoint path

_video_predictor = None  # Global cache: text prompt predictor
_tracker_predictor = None  # Global cache: point/box predictor

def get_video_predictor():
    """Return (or create) a video_predictor with text prompt support"""
    global _video_predictor
    if _video_predictor is None:
        _video_predictor = build_sam3_video_predictor(
            checkpoint_path=VIDEO_CKPT_PATH,
        )
    return _video_predictor

def get_tracker_predictor():
    """Return (or create) a tracker_predictor with point/box prompt support"""
    global _tracker_predictor
    if _tracker_predictor is None:
        sam3_model = build_sam3_video_model(
            checkpoint_path=VIDEO_CKPT_PATH,
            device=device,
        )
        _tracker_predictor = sam3_model.tracker
        _tracker_predictor.backbone = sam3_model.detector.backbone
    return _tracker_predictor


# ==================== Data Models ====================
class Point(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    label: int = Field(..., description="Label: 1=foreground, 0=background")

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

class ObjectAnnotation(BaseModel):
    object_name: str = Field(..., description="Object name")
    frame_idx: int = Field(..., description="Frame index of the annotation")
    points: Optional[List[Point]] = Field(None, description="Point annotation list")
    box: Optional[BoundingBox] = Field(None, description="Bounding box")
    text: Optional[str] = Field(None, description="Text prompt")

class SegmentRequest(BaseModel):
    frame_folder: str
    annotations: List[ObjectAnnotation]
    output_dir: Optional[str] = None
    background_color: Optional[List[int]] = Field([0, 0, 0])

class SegmentResponse(BaseModel):
    success: bool
    message: str
    frame_count: int
    objects: List[str]
    result_paths: Dict[str, Dict[str, str]]


# ==================== FastAPI Application ====================
app = FastAPI(
    title="SAM 3 Video Segmentation API",
    description="SAM 3 Video Segmentation API - Supports text, point, and box annotations",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "SAM 3 Video Segmentation API",
        "status": "running",
        "device": str(device),
        "docs": "/docs",
    }


# ==================== Process Functions ====================

def process_text_prompts(request: SegmentRequest, frame_names: List[str], annotation_to_obj_ids: Dict):
    """Process text annotations"""
    vp = get_video_predictor()

    # 1) Start session
    session_id = vp.handle_request(
        request={
            "type": "start_session",
            "resource_path": request.frame_folder,
        }
    )["session_id"]

    # 2) Add text prompt
    for idx, ann in enumerate(request.annotations):
        if ann.text:
            resp = vp.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": ann.frame_idx,
                    "text": ann.text,
                }
            )
            annotation_to_obj_ids[idx] = resp["outputs"]["out_obj_ids"].tolist()

    # 3) Propagate
    video_segments: Dict[int, Dict] = {}
    for resp in vp.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        fidx = resp["frame_index"]
        out = resp["outputs"]

        masks = out["out_binary_masks"]
        if masks.ndim == 3:
            masks = np.array([m.squeeze() for m in masks])

        video_segments[fidx] = {
            "obj_ids": out["out_obj_ids"],
            "masks": masks,
            "boxes": out["out_boxes_xywh"],
            "scores": out["out_probs"],
        }

    # 4) Close session and clean up GPU memory
    vp.handle_request(
        request={"type": "close_session", "session_id": session_id}
    )
    torch.cuda.empty_cache()
    gc.collect()

    return video_segments


def process_visual_prompts(
    request: SegmentRequest,
    frame_names: List[str],
    img_w: int,
    img_h: int,
    annotation_to_obj_ids: Dict,
):
    """Process point/box annotations"""
    tp = get_tracker_predictor()
    infer_state = tp.init_state(video_path=request.frame_folder)

    next_obj_id = 1
    for idx, ann in enumerate(request.annotations):
        obj_id = next_obj_id
        next_obj_id += 1

        # Points
        points_rel = None
        labels_tensor = None
        if ann.points:
            pts = np.array([[p.x / img_w, p.y / img_h] for p in ann.points], dtype=np.float32)
            points_rel = torch.tensor(pts)
            labels_tensor = torch.tensor([p.label for p in ann.points], dtype=torch.int32)

        # Box
        box_rel = None
        if ann.box:
            box_rel = np.array(
                [[
                    ann.box.x1 / img_w,
                    ann.box.y1 / img_h,
                    ann.box.x2 / img_w,
                    ann.box.y2 / img_h,
                ]],
                dtype=np.float32,
            )

        # Call the appropriate function based on the provided prompt type
        tp.add_new_points_or_box(
            inference_state=infer_state,
            frame_idx=ann.frame_idx,
            obj_id=obj_id,
            points=points_rel,
            labels=labels_tensor,
            box=box_rel,
        )

        annotation_to_obj_ids[idx] = [obj_id]

    # Propagate
    video_segments: Dict[int, Dict] = {}
    for fidx, obj_ids, _, res_masks, _ in tp.propagate_in_video(
        infer_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frame_names),
        reverse=False,
        propagate_preflight=True,
    ):
        masks = [(m > 0).cpu().numpy().squeeze() for m in res_masks]
        video_segments[fidx] = {
            "obj_ids": np.array(obj_ids),
            "masks": np.array(masks),
            "boxes": None,
            "scores": None,
        }

    # Clear state and GPU memory
    tp.clear_all_points_in_video(infer_state)
    del infer_state
    torch.cuda.empty_cache()
    gc.collect()
    return video_segments


# ==================== Main Interface ====================
@app.post("/segment", response_model=SegmentResponse)
async def segment_video(request: SegmentRequest):
    # ---------- Basic Validation ----------
    assert os.path.exists(request.frame_folder), f"Video frame folder does not exist: {request.frame_folder}"
    assert request.annotations, "At least one object needs to be annotated"

    frame_names = sorted(
        [f for f in os.listdir(request.frame_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
        key=lambda p: int(os.path.splitext(p)[0]) if os.path.splitext(p)[0].isdigit() else p,
    )   
    assert frame_names, "No frame images found"

    img_w, img_h = Image.open(os.path.join(request.frame_folder, frame_names[0])).size

    has_text = any(a.text for a in request.annotations)
    has_visual = any(a.points or a.box for a in request.annotations)

    ann2obj: Dict[int, List[int]] = {}

    if has_text and not has_visual:
        video_segments = process_text_prompts(request, frame_names, ann2obj)
    elif has_visual and not has_text:
        video_segments = process_visual_prompts(request, frame_names, img_w, img_h, ann2obj)
    else:
        raise NotImplementedError("Currently not supported to mix text and visual prompts in the same request, please call separately")

    # ---------- Save results ----------
    out_dir = request.output_dir or os.path.dirname(os.path.abspath(request.frame_folder))
    bg_color = request.background_color or [0, 0, 0]
    assert len(bg_color) == 3 and all(0 <= c <= 255 for c in bg_color)

    result_paths = save_segmentation_results(
        frame_folder=request.frame_folder,
        frame_names=frame_names,
        video_segments=video_segments,
        output_dir=out_dir,
        annotations=request.annotations,
        annotation_to_obj_ids=ann2obj,
        background_color=bg_color,
    )

    del video_segments
    torch.cuda.empty_cache()
    gc.collect()

    return SegmentResponse(
        success=True,
        message="Segmentation completed",
        frame_count=len(frame_names),
        objects=list(result_paths.keys()),
        result_paths=result_paths,
    )


# ==================== Helper Functions ====================

def save_segmentation_results(
    frame_folder: str,
    frame_names: List[str],
    video_segments: Dict[int, Dict],
    output_dir: str,
    annotations: List[ObjectAnnotation],
    annotation_to_obj_ids: Dict[int, List[int]],
    background_color: List[int],
) -> Dict[str, Dict[str, str]]:
    """Save segmentation results to disk, return path indices"""
    res: Dict[str, Dict[str, str]] = {}

    for ann_idx, ann in enumerate(annotations):
        obj_ids = annotation_to_obj_ids.get(ann_idx, [])
        if not obj_ids:
            continue

        # Text prompt may produce multiple instances, so we limit it to one for segmentation
        assert len(obj_ids) == 1, "Text prompt should produce only one instance for segmentation"
        name = ann.object_name
        save_single_object(name, obj_ids[0], frame_folder, frame_names, video_segments, output_dir, background_color, res)
    return res


def save_single_object(
    obj_name: str,
    obj_id: int,
    frame_folder: str,
    frame_names: List[str],
    video_segments: Dict[int, Dict],
    output_dir: str,
    background_color: List[int],
    res_dict: Dict,
):
    obj_dir = os.path.join(output_dir, obj_name)
    mask_dir = os.path.join(obj_dir, f"{obj_name}_mask")
    seg_dir = os.path.join(obj_dir, f"{obj_name}_segframe")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    for idx, fname in tqdm(list(enumerate(frame_names)), desc=f"Saving {obj_name}"):
        if idx not in video_segments:
            continue
        data = video_segments[idx]
        if obj_id not in data["obj_ids"]:
            continue
        loc = list(data["obj_ids"]).index(obj_id)
        mask = data["masks"][loc]

        # Save mask
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(mask_dir, f"{os.path.splitext(fname)[0]}.jpg")
        )

        # Save colored segmentation image (RGB)
        frame = np.array(Image.open(os.path.join(frame_folder, fname)))
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        seg = frame.copy()
        seg[~mask] = background_color
        Image.fromarray(seg.astype(np.uint8)).save(
            os.path.join(seg_dir, f"{os.path.splitext(fname)[0]}.jpg")
        )

    res_dict[obj_name] = {
        "object_folder": obj_dir,
        "mask_folder": mask_dir,
        "segframe_folder": seg_dir,
    }


# ==================== Start Script ====================
if __name__ == "__main__":

    print("=" * 60)
    print("SAM 3 Video Segmentation API")
    print("=" * 60)
    print(f"Device: {device}")
    print("=" * 60)
    print("Start service: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "sam3_segment_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )
