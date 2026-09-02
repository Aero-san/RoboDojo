# HDF5 episode video previews

`scripts/RoboDojo/visualize_hdf5.py` samples a small number of RoboDojo HDF5
episodes and creates one labelled, multi-camera MP4 per selected episode. It
streams frames directly from HDF5, so a complete episode is never loaded into
memory.

Run the default deterministic preview (three episodes sampled evenly across
`data/RoboDojo`):

```bash
conda run -n RoboDojo python scripts/RoboDojo/visualize_hdf5.py
```

Randomly sample three episodes from one task:

```bash
conda run -n RoboDojo python scripts/RoboDojo/visualize_hdf5.py \
  --input data/RoboDojo \
  --output outputs/hdf5_visualizations \
  --task pour_by_language \
  --num-episodes 3 \
  --sampling random \
  --seed 42
```

Create faster, shorter previews by keeping every second source frame and at
most 150 output frames. The default output FPS is divided by `--stride`, so
temporal playback speed is preserved:

```bash
conda run -n RoboDojo python scripts/RoboDojo/visualize_hdf5.py \
  --stride 2 \
  --max-frames 150 \
  --panel-width 400
```

Use `--camera cam_head` to export only the main camera. Repeat `--camera` to
select more than one view. Existing videos are skipped unless `--overwrite`
is supplied.

The output mirrors the source tree to avoid filename collisions:

```text
outputs/hdf5_visualizations/
  align_blocks/arx_x5/data/episode_0000000.mp4
  pour_by_language/arx_x5/data/episode_0000058.mp4
  manifest.json
```

`manifest.json` records the sampled source files, task instructions, camera
names, source/output FPS, frame counts, output paths, and any conversion
errors. Run `python scripts/RoboDojo/visualize_hdf5.py --help` for every option.
