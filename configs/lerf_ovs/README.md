# LERF-OVS reproduction configurations

Each CSV row defines one segmentation and evaluation target. `object_name` is
the filesystem-safe output name, while `gt_object_name` must exactly match the
prefix used by the curated files in `data/lerf_ovs/<scene>/masks`.

Prompt fields:

- `prompt_text`: SAM text prompt;
- `first_frame_idx`: zero-based prompt frame index;
- `fg_points` and `bg_points`: JSON arrays of `[x, y]` image coordinates;
- `box`: optional JSON `[x_min, y_min, x_max, y_max]` box.

Text and spatial prompts must not be mixed in one row. At least one text,
foreground-point, background-point, or box prompt is required.

Blank algorithm fields use these defaults from `run_lerf_ovs.py`:

| Field | Default |
|---|---:|
| `threshold_r1` | 0.7 |
| `threshold_r2` | 0.5 |
| `min_visible_ratio` | 0.01 |
| `angular_gap_threshold` | 90.0 |
| `n_layers` | 4 |
| `n_points_per_layer` | 8 |
| `boundary_refine` | true |
| `min_compression` | 0.2 |
| `tolerance_ratio` | 0.2 |
| `api_url` | `http://localhost:8000` |
| optional `random_seed` | 42 |

Existing rows may override these values. Before every target, the runner resets
Python, NumPy, and PyTorch random states using `random_seed` so that execution
order does not change the configured random state.

For the current parameter sweep, every CSV row explicitly sets
`min_visible_ratio=0.01`, `angular_gap_threshold=90`,
`min_compression=0.1`, and
`tolerance_ratio=0.6`. The existing `threshold_r1` and `threshold_r2` values
remain unchanged.

The following provisional targets currently use one deterministic foreground
point and a bounding box derived from their reviewed mask. Their algorithm
fields now use the same uniform sweep values described above:

- `teatime/hooves2`
- `teatime/hooves3`
- `teatime/three_cookies1`
- `teatime/three_cookies2`

These four rows are temporary end-to-end reproduction placeholders. Recheck
their prompts and tune their parameters before reporting final experiment
numbers.
