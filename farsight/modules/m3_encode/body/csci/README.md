# CSCI body encoder (plan_csci, M3c)

CSCI (ICCV'25, "Colors See Colors Ignore"), EVA02-L/14 at 224×224. Each track gives a 1024-d ReID token (the `fc_norm` output of the cls token). It replaces AIM (4096-d). AIM stays in place as the fallback: set `body: aim` in `configs/pipeline/*.yaml`.

| registry name | checkpoint (config.yaml) | input |
|---|---|---|
| `csci_video` | `checkpoint`: `csci_v_mevid.pth` (default) or `csci_v_ccvid.pth` | EZ-CLIP video model, 4-frame clips |
| `csci_image` | `image_checkpoint`: `csci_img_mevid.pth` / `csci_img_ccvid.pth` | single frames |

Checkpoints come from the Hugging Face repo `ppriyank/CSCI-CC-ReID`. They are listed in `manifest.yaml` with their sha256, and `fetch()` downloads them on first use. Model code comes from `third_party/CSCI`, which is a git submodule pinned to a fixed commit.

## Protocol (same as the author's `Script/test.sh` and `train_two_step.py --eval`)

1. **Split the track into clips.** `clips.recombine` reproduces `MEVID._recombination_for_testset`. Clips are 8 frames long, taken with stride 4 over 32-frame spans. The remainder uses a reduced stride, and the last clip is padded by cycling its frames. This gives `ceil(n/8)` clips that together cover every frame.
2. **Feed 4 frames per clip.** `clip[::2]` gives the T=4 frames the model takes.
3. **Preprocess.** Each frame is stretched to 224×224 with PIL bicubic and normalised with ImageNet statistics.
4. **Run the model.** Inference uses fp16 autocast.
5. **Pool.** Track feature = L2(mean of clip features).

**Clip budget (plan B3).** Each track uses at most `max_clips: 8` clips, spread evenly over the track. Set `0` to keep all clips, which matches the author exactly.

**Short tracks.** A track with fewer than 4 frames goes to the image model. That model is loaded lazily.

**Other outputs.**
- `quality`: crop height / 224 × mean detector confidence, the same formula as AIM.
- `inter_feat`: mean patch token of `blocks[16]` (plan B5, used for an optional body QE).

## Import notes

CSCI's packages (`model`, `loss`, `tools`, `config`) are imported as bare namespaces and removed from `sys.modules` afterwards. Their `__init__` files pull in training-only dependencies (mmengine and others), and the package name `tools` would clash with FarSight's own `tools/`. `torchinfo` is stubbed out. The only new pip dependency is `yacs`. timm 1.0.30 works.

## Tests

`pytest farsight/modules/m3_encode/body/csci` covers:
- clip splitting compared against the CAL rule;
- shapes with random initialisation;
- with the checkpoint: strict load, determinism, discrimination, speed and VRAM.

On an RTX 5050 Laptop GPU: 0.55 s per 8-clip track, about 58 frames/s, 2.1 GB peak VRAM.

`eval/csci_a2.py` runs plan steps A2, A3 and B6. It compares the author's own pipeline with this wrapper on a MEVID test subset (`splits/mevid_csci_subset.json`) and writes `eval/results/csci_a2.json`.
