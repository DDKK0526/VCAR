# NVOS reproduction parameters

`nvos.csv` contains the seven NVOS targets used by the paper's default
evaluation. Prompt points, boxes, and annotation frames were recovered from
the historical `segmentation_params.json` files referenced by
`ours_nvos_eval.csv`. Algorithm parameters follow the paper settings rather
than the ambiguous single `threshold` field in the old JSON files.

## Paper parameters

| Parameter | Value |
|---|---:|
| Round 1 voting threshold `threshold_r1` | 0.5 |
| Round 2 voting threshold `threshold_r2` | 0.8 |
| Angular-gap threshold | 90° |
| Spherical spiral sampling | 4 layers × 8 points |
| Minimum ABR compression | 0.1 |
| ABR overflow tolerance | 0.6 |

Every NVOS row must set `force_round2=true`. Round 2 runs even when the
training-view coverage analysis finds no need for additional spherical views.
When coverage is insufficient, Round 2 uses the training views plus 32
spherical views.

## Fields

| Field | Description |
|---|---|
| `scene_name` | Data directory and 3DGS scene name |
| `object_name` | Segmentation target and output subdirectory name |
| `gt_mask_file` | GT filename under `<scene>/masks` |
| `camera_name` | COLMAP image name for the GT view, without its extension |
| `prompt_text` | Optional text prompt; current rows use points and boxes |
| `first_frame_idx` | Annotated training-view index in COLMAP render order |
| `fg_points` / `bg_points` | Foreground and background points as JSON |
| `box` | Optional JSON box `[x1, y1, x2, y2]` |
| `force_round2` | Fixed to `true` by the NVOS paper protocol |
| `boundary_refine` | Whether to run ABR |
| `api_url` | Local SAM3 service URL; may be overridden on the command line |
| `random_seed` | Random seed reset before each scene |

The official data also includes `horns_left`, while the historical VCAR,
LangSplat, and LUDVIG evaluation tables use only `horns_center`. The
`horns_left` annotation is preserved in the data package but is not included
in the default seven-target evaluation.
