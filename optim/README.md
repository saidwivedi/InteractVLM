# Optimization module: data preparation and usage

Follow the [root setup](../README.md), then prepare a per-sample folder containing all inputs the optimizer expects. The code reads files directly from a single sample folder whose name must match the image filename.

You can download and unzip an example folder by running `fetch_data.sh optim-demo-data`

## 📁 Sample folder structure (example)

```
optim_data/
├── 📄 tennis_racket__000000041045.jpg   # RGB input image
├── 📄 osx_human2.npz                    # OS-X output: smpl_vertices, smpl_faces, bbox_2
├── 📄 hcontact_vertices.npz             # Human contact probs (pred_contact_3d_smplx)
├── 📄 human_mask.png                    # Person mask (binary HxW) — or use JSON below
├── 📄 human_detection.json              # Alternative to PNG; contains "mask": HxW
├── 📄 object_mesh.obj                   # Retrieved object mesh (triangles, meters)
├── 📄 ocontact_vertices.npz             # Object contact probs (pred_contact_3d)
├── 📄 object_mask.png                   # Object mask (binary HxW) — or use JSON below
└── 📄 object_detection.json             # { bbox: [x,y,w,h], mask: HxW }
```

Sources for these files:
- [OS-X](https://github.com/IDEA-Research/OSX) → produces `osx_human2.npz`
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) → produces `human_mask.png` / `human_detection.json` and `object_mask.png` / `object_detection.json`
- [Object_Retrieval](https://github.com/saidwivedi/Object_Retrieval) → produces `object_mesh.obj`


# Run

Run the optimization demo:

```bash
bash scripts/run_optim.sh
```
or

```bash
python -m optim.fit \
	--input_path optim_data/tennis_racket__000000041045.jpg \
	--cfg optim/cfg/fit.yaml \
	[--out_root /path/to/output_root]
```