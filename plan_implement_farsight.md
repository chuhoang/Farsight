# Kế hoạch triển khai FarSight 2.0 End-to-End (ưu tiên tốc độ)

> Tài liệu tham chiếu: *Person Recognition at Altitude and Range: Fusion of Face, Body Shape and Gait* (FarSight 2.0, arXiv 2505.04616)
> Phạm vi: dựng lại pipeline nhận dạng người toàn thân (mặt + dáng đi + hình thể) từ video tầm xa / góc cao, dùng mã nguồn mở và dữ liệu công khai.
> Ngoài phạm vi: pháp lý, license, thương mại hoá.

---

## 0. Nguyên tắc triển khai

1. **Pretrained trước, train sau.** Module nào đã có checkpoint đủ tốt thì dùng nguyên trạng, không train lại. Chỉ train khi không có checkpoint hoặc khi đo thấy kém rõ rệt trên dữ liệu đánh giá.
2. **Chạy được end-to-end sớm nhất có thể.** Phiên bản đầu dùng giải pháp đơn giản nhất cho từng khâu, sau đó mới nâng cấp từng module theo mức độ cải thiện đo được.
3. **Hoãn những thứ tốn công mà lợi ít.** Theo ablation của bài báo, phục hồi nhiễu loạn (GRTM) chỉ cải thiện vài điểm, nên chỉ làm bản rút gọn ở giai đoạn đầu.
4. **Module hoá theo interface cố định.** Mỗi module nhận và trả về đúng schema định nghĩa ở mục 3, nhờ vậy có thể thay model bên trong mà không ảnh hưởng phần còn lại.

---

## 1. Kiến trúc tổng thể

```
Video (probe hoặc gallery)
   │
   ▼
[M1] Phát hiện + theo vết
   BPJDet (thân+mặt) → YOLOv8 xác minh → PSR-ByteTrack
   │  output: tracklets (track_id, frame, body_box, face_box)
   ▼
[M2] Phục hồi nhiễu loạn (chỉ crop mặt, có điều kiện)
   Quality gate → DATUM pretrained
   │
   ▼
[M3] Mã hoá sinh trắc
   ├─ Mặt:     DFA aligner → KP-RPE ViT        → 512-d (+ quality)
   ├─ Dáng đi: BigGait / BiggerGait (DINOv2)   → feature gait
   └─ Thân:    AIM ResNet-50                   → 4096-d
   │  gộp theo track → 1 template / modality / người
   ▼
[M4] So khớp + hợp nhất
   cosine per-modality → ma trận điểm N_G × 3 → z-score fusion → (sau) QME
   │
   ▼
Danh sách ứng viên (open-set: có thể trả "không có trong gallery")
```

---

## 2. Tổng hợp model theo module

| Module | Model | Repo | Checkpoint | Train lại? |
|---|---|---|---|---|
| M1 detector | BPJDet (`ch_face_l_1536`) | github.com/hnuzhy/BPJDet · HF: HoyerChou/BPJDet | Có (CrowdHuman body-face) | **Không** |
| M1 verifier | YOLOv8x | github.com/ultralytics/ultralytics | Có (COCO) | **Không** |
| M1 tracker | ByteTrack + PSR (tự viết) | github.com/ifzhang/ByteTrack | Thuật toán, không cần train | **Không** (chỉ tune ngưỡng) |
| M1 appearance | ResNet-18 | torchvision | ImageNet | **Không** (tuỳ chọn thay bằng model ReID) |
| M2 restore | DATUM (dynamic scene) | github.com/xg416/DATUM | Có | **Không** ở giai đoạn đầu |
| M3 face align | DFA mobilenet | HF: minchul/cvlface_DFA_mobilenet | Có | **Không** |
| M3 face | KP-RPE ViT-Base (WebFace12M) | github.com/mk-minchul/CVLface · HF: minchul/cvlface_adaface_vit_base_kprpe_webface12m | Có | **Không** |
| M3 gait | BigGait / BiggerGait | github.com/ShiqiYu/OpenGait · HF: opengait/OpenGait | Có (CCPG; BiggerGait có cả CCGR) | **Không** (tuỳ chọn fine-tune) |
| M3 body | AIM (ResNet-50) | github.com/BoomShakaY/AIM-CCReID (không có file LICENSE) | **Có** (LTCC, PRCC); bản trong repo QME `checkpoints/AIM/` | **Không** |
| M4 fusion v1 | z-score + trọng số chất lượng | tự viết | — | **Không** |
| M4 QE (Quality Estimator) | QE của QME, lấy feature trung gian của encoder mặt | github.com/jiezhu23/QME_ICCV25 (MIT) | Có, nhưng train trên **AdaFace** | **Có** nếu dùng KP-RPE (QE gắn với backbone); nhẹ, ~6.000 bước |
| M4 fusion v2 | QME (MoE score-fusion) | github.com/jiezhu23/QME_ICCV25 | Có (kèm ma trận điểm tính sẵn cho CCVID, LTCC…) | Train nhẹ trên ma trận điểm, sau khi đóng băng QE |

---

## 3. Interface dữ liệu giữa các module

### 3.1 Output M1 – Tracklet
```python
Tracklet = {
  "video_id": str,
  "track_id": int,
  "frames": [
    {
      "frame_idx": int,
      "body_box": [x1, y1, x2, y2],
      "body_score": float,
      "face_box": [x1, y1, x2, y2] | None,
      "face_score": float | None,
    }, ...
  ]
}
```

### 3.2 Output M3 – Template
```python
ModalityOut = {
  "feat": float32[D],            # feature cuối, đã L2-norm, dùng để so khớp
  "quality": float,              # chất lượng heuristic (norm / độ dài track...), dùng cho fusion v1
  "inter_feat": float32[D_mid] | None,  # feature TRUNG GIAN của backbone, đầu vào cho QE (fusion v2)
  "n_frames": int,               # số frame thực sự đóng góp vào template
}

Template = {
  "subject_or_track_id": str,
  "face": ModalityOut | None,    # None = thiếu modality (không thấy mặt / chất lượng quá thấp)
  "gait": ModalityOut | None,    # None = track quá ngắn
  "body": ModalityOut | None,
  "qe_weight": {"face": float, "gait": float, "body": float} | None,  # do QE sinh ra (v2)
}
```
- Lưu template ra **HDF5** (một file gallery, nhóm theo modality), giống cách FarSight làm. Repo QME cũng dùng HDF5 cho dữ liệu và ma trận điểm, nên giữ cùng định dạng để dùng lại script của họ.
- `inter_feat` chỉ cần lưu cho **probe** và cho dữ liệu dùng train QE. Với gallery lớn có thể bỏ để tiết kiệm dung lượng.
- `inter_feat` được gộp theo track giống `feat` (trung bình theo frame), trừ khi QE yêu cầu đầu vào theo từng frame. Kiểm tra lại trong `model.py` của repo QME.

### 3.3 Output M4 – Kết quả tìm kiếm
```python
SearchResult = {
  "probe_id": str,
  "ranked": [{"gallery_id": str, "score": float, "per_modality": [s_face, s_gait, s_body]}],
  "is_known": bool,   # so với ngưỡng open-set
}
```

---

## 4. Chi tiết từng module

### M1 — Phát hiện và theo vết

**Việc cần làm**
1. Chạy BPJDet `ch_face_l` ở độ phân giải 1536 (đúng với checkpoint). Decode: NMS riêng cho thân và mặt, sau đó ghép thân–mặt bằng offset + inner IoU, dùng hàm decode có sẵn trong repo.
2. Chạy YOLOv8x lớp *person*. Chỉ giữ hộp thân của BPJDet nếu có hộp YOLOv8 tương ứng (IoU ≥ 0.5) với confidence ≥ 0.7.
3. **Tối ưu tốc độ:** tiền xử lý ảnh (resize, letterbox, normalize) một lần trên GPU rồi dùng chung cho cả hai detector; chạy theo batch frame (bs = 8).
4. **PSR-ByteTrack (tự viết):**
   - ByteTrack nhận hộp thân đã xác minh.
   - Mỗi track ID có một bộ nhớ gồm tối đa K feature ResNet-18 (gợi ý K = 10). Cứ N frame (gợi ý N = 10) thì thêm một patch mới, bỏ patch cũ nhất (FIFO).
   - Khi ByteTrack tạo ID mới hoặc khi hai track giao nhau: so feature của patch hiện tại với bộ nhớ của mọi track còn sống hoặc vừa mất (trong cửa sổ T giây), gán ID có khoảng cách nhỏ nhất nếu dưới ngưỡng, ngược lại giữ ID mới.
   - Bài báo dùng MSE trên feature ResNet-18. Có thể đổi sang cosine distance trên feature đã chuẩn hoá L2 nếu kết quả ổn định hơn.
5. **Lọc tracklet:** bỏ các track ngắn hơn 15 frame, cắt các đoạn có hộp thân quá nhỏ (chiều cao < 32 px).

**Dữ liệu**
| Mục đích | Bộ |
|---|---|
| Tune ngưỡng tracking | MOT17, MEVID, DanceTrack |
| Kiểm tra detector tầm xa / góc cao | VisDrone, UAV-Human (đánh giá định tính, không cần nhãn mặt) |

**Tiêu chí hoàn thành**
- ≥ 20 FPS cho M1 ở 1080p trên 1 GPU.
- Số lần đổi ID (ID switch) của PSR-ByteTrack thấp hơn ByteTrack thuần trên MOT17/MEVID.

**Công sức:** khoảng 2 tuần, 1 kỹ sư.

---

### M2 — Phục hồi nhiễu loạn (bản rút gọn)

**Quyết định:** không xây GRTM và không đồng tối ưu ở giai đoạn đầu. Dùng DATUM pretrained cộng một cổng chất lượng đơn giản.

**Việc cần làm**
1. Với mỗi track, lấy chuỗi crop mặt đã nới biên (padding 20%).
2. **Quality gate:** chạy KP-RPE một lần trên crop gốc và lấy norm feature làm điểm chất lượng. Chỉ đưa qua DATUM những track có quality trung vị dưới ngưỡng q₀.
3. Chạy DATUM (dynamic scene model) trên cửa sổ 10–20 frame liên tiếp.
4. **Kiểm tra an toàn:** tính feature mặt trước và sau khi phục hồi. Nếu độ tương đồng giữa hai phiên bản với template trung bình của track giảm, bỏ kết quả phục hồi và dùng crop gốc.

**Dữ liệu**
| Mục đích | Bộ |
|---|---|
| Chọn ngưỡng q₀ và kiểm tra lợi ích | Crop mặt từ CCVID/MEVID được làm suy giảm bằng ATSyn hoặc simulator (downsample + blur + nhiễu loạn mô phỏng) |
| Nếu cần fine-tune sau | ATSyn-dynamic, ATSyn-static |

**Tiêu chí hoàn thành:** trên tập mặt suy giảm, TAR@0.1%FAR của mặt không giảm và tăng ≥ 1 điểm ở nhóm chất lượng thấp. Nếu không đạt, **tắt M2**.

**Công sức:** khoảng 1–1.5 tuần, 1 kỹ sư.

**Nâng cấp sau (tuỳ chọn):** GRTM (fork DATUM, bỏ optical flow), classifier "có cần phục hồi không", đồng tối ưu với AdaFace loss. Khoảng 4–6 tuần.

---

### M3a — Khuôn mặt

**Model:** KP-RPE ViT-Base WebFace12M cộng aligner DFA, **không train**.

**Việc cần làm**
1. Crop mặt (đã phục hồi nếu có) → DFA → 5 landmark + ảnh căn chỉnh 112×112 → KP-RPE → feature 512-d.
2. **Điểm chất lượng:** norm feature trước khi chuẩn hoá L2.
3. **Gộp template theo track:** trung bình có trọng số theo norm (softmax trên norm). Có thể thay bằng CAFace (pretrained) nếu cần tốt hơn.
4. Bỏ các frame có quality dưới ngưỡng thấp. Nếu track không có frame nào đủ chất lượng thì face = None.
5. **Hook feature trung gian cho QE (bắt buộc cho fusion v2):**
   - Đăng ký `register_forward_hook` trên một block transformer ở giữa/sau của ViT (ứng viên: output block thứ ~8/12, lấy CLS token hoặc mean-pool các patch token).
   - Trả về `inter_feat` cùng với `feat` trong một lần forward, không chạy lại model.
   - Lớp cụ thể cần đối chiếu với cách repo QME lấy feature trung gian của AdaFace (`build_face_backbone()` trong `model.py`), rồi chọn lớp tương ứng trên KP-RPE. Nếu chưa chắc, xuất 2–3 lớp ứng viên và chọn bằng thực nghiệm khi train QE.

```python
class FaceEncoder:
    def __init__(self, model, hook_layer):
        self._inter = None
        hook_layer.register_forward_hook(lambda m, i, o: setattr(self, "_inter", o))
        self.model = model

    @torch.no_grad()
    def __call__(self, img, kps):
        raw = self.model(img, kps)              # trước L2-norm
        inter = pool_tokens(self._inter)        # (B, D_mid)
        return {
            "feat": F.normalize(raw, dim=-1),
            "quality": raw.norm(dim=-1),         # heuristic cho v1
            "inter_feat": inter,                 # đầu vào QE cho v2
        }
```

**Dữ liệu đánh giá:** TinyFace, IJB-S (nếu có), DroneSURF, crop mặt từ CCVID/MEVID.

**Tiêu chí hoàn thành:** kết quả trên TinyFace khớp với số liệu được công bố của checkpoint (sai lệch ≤ 1%). Đạt ≥ 30 FPS theo crop.

**Công sức:** khoảng 1 tuần.

**Fine-tune (chỉ khi cần):** WebFace4M kết hợp tăng cường suy giảm mạnh (downsample, blur, nhiễu loạn mô phỏng) theo tinh thần DaliID.

---

### M3b — Dáng đi

**Model:** BigGait (DINOv2 ViT-S) checkpoint CCPG để chạy nhanh. Thử thêm BiggerGait checkpoint CCPG/CCGR, giữ bản nào tốt hơn trên tập đánh giá. **Không train** ở giai đoạn đầu.

**Việc cần làm**
1. Từ tracklet lấy chuỗi crop thân RGB, resize đúng kích thước đầu vào của config checkpoint.
2. Lấy mẫu 30 frame mỗi lần suy luận (đúng setting của checkpoint "Frame30"). Track dài thì chia thành nhiều đoạn và lấy trung bình feature.
3. Track quá ngắn (dưới khoảng 15 frame) thì gait = None.
4. **Quality:** tỉ lệ frame có hộp thân đủ lớn × độ dài track (chuẩn hoá về 0–1).
5. **Hook feature trung gian (tuỳ chọn, cho QE nhánh thân/dáng đi):**
   - Repo QME mặc định chỉ train QE cho **mặt**; QE cho thân nằm trong phần code đang comment của `model.py`. Giai đoạn đầu **chưa bắt buộc**, nhưng wrapper vẫn nên có sẵn hook để không phải sửa lại sau.
   - Vị trí gợi ý: feature DINOv2 sau backbone (trước ba nhánh Mask/Appearance/Denoising), hoặc output giữa GaitBase trước lớp pooling cuối. Repo QME có sẵn wrapper BigGait trong `WBModules/BigGait/`, nên **ưu tiên dùng lại đúng vị trí hook của họ**.
   - `inter_feat` lấy trung bình theo các đoạn 30 frame, giống `feat`.

**Dữ liệu**
| Mục đích | Bộ |
|---|---|
| Kiểm tra checkpoint | CCPG test |
| Đánh giá chéo miền | CCGR-MINI, SUSTech1K, DroneGait |
| Fine-tune (tuỳ chọn) | CCPG + CCGR-MINI, dùng open-set loss (github.com/prevso1088/open-set-biometrics) |

**Tiêu chí hoàn thành:** tái hiện được số liệu trên CCPG; tốc độ ≥ 20 track-clip/giây.

**Công sức:** khoảng 1 tuần (thêm khoảng 2 tuần nếu fine-tune).

---

### M3c — Thân người (body)

**Model:** **AIM** (CVPR 2023, *Good is Bad: Causality Inspired Cloth-Debiasing for Cloth-Changing Person Re-Identification*), backbone ResNet-50, **dùng checkpoint có sẵn, không train**.

> Thay đổi so với FarSight 2.0: bài gốc dùng CLIP3DReID cho module này; plan thay bằng AIM vì AIM có checkpoint và code chạy được ngay.

**Nguồn checkpoint (ưu tiên theo thứ tự)**
1. **Repo QME** (`checkpoints/AIM/`): khớp sẵn với QE và ma trận điểm của QME (lộ trình 2A). Cần mở Google Drive để xác nhận checkpoint train trên bộ nào.
2. **Repo AIM gốc** `github.com/BoomShakaY/AIM-CCReID`: checkpoint LTCC và PRCC (Google Drive / Baidu). README công bố kết quả tái hiện bằng repo gần như trùng với bài báo (LTCC CC: mAP 19.2 / R1 40.8; PRCC CC: mAP 58.0 / R1 58.2).
3. Ưu tiên checkpoint **LTCC** (12 camera, nhiều trang phục) hơn PRCC; xác nhận bằng đánh giá trên CCVID/MEVID.

**Thông số suy luận (theo config của repo)**
| Mục | Giá trị |
|---|---|
| Đầu vào | Crop thân RGB 384×192 |
| Backbone dùng khi suy luận | Chỉ `model` (nhánh danh tính, ResNet-50). Nhánh quần áo `model2` chỉ dùng khi train, **không nạp** khi chạy |
| Pooling | `maxavg` (ghép max + avg) |
| Feature | **4096-d** |

**Việc cần làm**
1. Viết wrapper `farsight/modules/m3_encode/body/aim/model.py`: nạp `model_state_dict` từ checkpoint AIM (bỏ `model2_state_dict`), chuẩn hoá ảnh giống `data/img_transforms` của repo, xuất feature 4096-d đã L2-norm.
2. Gộp theo track bằng trung bình feature các frame (lấy mẫu tối đa ~32 frame/track để giữ tốc độ).
3. **Giảm kích thước template (tuỳ chọn):** 4096-d nặng gấp đôi feature body của FarSight. Thử chỉ giữ nửa avg-pool (2048-d) và đo lại mAP trên CCVID; chỉ giữ cách cắt giảm nếu mất ≤ 1 điểm.
4. **Quality (cho fusion v1):** độ phân giải trung bình của crop thân × tỉ lệ frame không bị che (theo confidence detector).
5. **Hook feature trung gian (cho QE nhánh thân, tuỳ chọn):** output `layer3` của ResNet-50 sau global-average-pool. Repo QME có sẵn wrapper AIM, nên ưu tiên dùng đúng vị trí hook của họ.
6. **Kiểm tra mức phụ thuộc vào khuôn mặt:** AIM chỉ được dạy "bỏ quần áo", nên feature có thể chứa cả mặt và tóc. Đánh giá trên CCVID/MEVID với vùng mặt bị làm mờ để biết AIM còn mạnh đến đâu khi không thấy mặt, vì đây là điều kiện module body quan trọng nhất.

**Dữ liệu**
| Mục đích | Bộ |
|---|---|
| Đánh giá checkpoint | CCVID, MEVID (protocol cloth-changing), thêm bản làm mờ mặt |
| Đánh giá góc cao | AG-ReID.v2 |
| Fine-tune (chỉ khi cần) | CCVID + LTCC + PRCC, dùng code train của repo AIM (2 GPU, chạy phân tán); cần tự thêm loader CCVID vì repo AIM chỉ có loader LTCC/PRCC |

**Tiêu chí hoàn thành:** tái hiện số liệu LTCC/PRCC của repo (sai lệch ≤ 1 điểm); chọn được checkpoint (LTCC hay PRCC, bản repo AIM hay bản repo QME) tốt nhất trên CCVID/MEVID.

**Công sức:** khoảng 3–5 ngày (wrapper, kiểm tra và chọn checkpoint). Thêm khoảng 1 tuần nếu phải fine-tune.

---

### M4 — So khớp và hợp nhất

**v1 (không train):**
1. Với mỗi modality: tính cosine giữa probe và từng template gallery → vector điểm dài N_G. Modality bị thiếu thì gán NaN.
2. **Chuẩn hoá z-score** theo thống kê điểm impostor của từng modality, ước lượng trên tập validation.
3. **Hợp nhất:** `s = Σ_m w_m · q_m · z_m / Σ_m w_m · q_m`, bỏ qua các modality bị thiếu. Trọng số w_m chọn bằng grid search trên validation; q_m là quality của modality.
4. **Open-set:** đặt ngưỡng τ trên điểm top-1 để đạt FPIR = 1% trên tập có người không nằm trong gallery (distractor).

**v2 (QME): chia làm hai lộ trình**

| Lộ trình | Encoder | QE | Mục đích |
|---|---|---|---|
| **2A – theo repo QME** | AdaFace + BigGait + AIM (đúng như repo) | Dùng **checkpoint QE có sẵn** | Kiểm chứng nhanh toàn bộ pipeline fusion, có số liệu đối chiếu với bài QME |
| **2B – encoder bản cuối** | KP-RPE + BigGait + AIM | **Train lại QE** trên `inter_feat` của KP-RPE | Bản cuối của hệ thống |

Làm 2A trước (khoảng 2–3 ngày, chủ yếu là tải dữ liệu .h5 và chạy `demo.ipynb`), sau đó mới chuyển sang 2B.

#### 2B.1 Chuẩn bị dữ liệu train QME

**Yêu cầu:** cùng một người phải có **đủ ba modality** trong cùng video/track, kèm nhãn danh tính. Trong các bộ public chỉ **CCVID** và **MEVID** đáp ứng. (Repo QME còn có protocol cho LTCC, nhưng LTCC là ảnh tĩnh, không có dáng đi.)

**Chia dữ liệu theo danh tính, không theo video** để tránh rò dữ liệu:

| Bộ | Phần dùng | Vai trò |
|---|---|---|
| CCVID | Tập **train** chính thức (75 ID) | Train QE + QME |
| CCVID | Tập **test** chính thức (151 ID), chia đôi theo ID | Một nửa làm **validation** (chọn checkpoint, ngưỡng τ, trọng số v1); một nửa làm **test** |
| MEVID | Tập train chính thức | Train QE + QME (gộp với CCVID) |
| MEVID | Tập test chính thức | **Chỉ dùng để đánh giá**, không bao giờ dùng để chọn tham số |
| AG-ReID.v2, MSU-BRC | Toàn bộ | **Chỉ đánh giá**, để đo khả năng tổng quát hoá sang miền khác |

Quy tắc bắt buộc:
- Danh tính trong tập train QME **không được xuất hiện** trong bất kỳ tập validation/test nào.
- Các encoder (KP-RPE, BigGait, AIM) cũng **không được train trên phần test** của CCVID/MEVID. Checkpoint AIM gốc train trên LTCC/PRCC nên không chồng ID với CCVID/MEVID; nếu fine-tune AIM trên CCVID (M3c) thì chỉ dùng split train.
- Ghi lại danh sách ID của từng split vào file (`splits/ccvid_qme.json`, `splits/mevid_qme.json`) và commit vào repo.

#### 2B.2 Tạo thêm trường hợp thiếu modality khi train

Nếu tập train luôn đủ cả ba modality, QME sẽ không học được cách xử lý khi một modality vắng mặt. Cần chủ động tạo các trường hợp này:

| Kiểu thiếu | Cách tạo | Tỉ lệ gợi ý |
|---|---|---|
| **Không thấy mặt** | Bỏ toàn bộ frame có mặt của probe → `face = None` | ~20% probe |
| **Mặt chất lượng thấp** | Downsample crop mặt ×4–×8, thêm blur/nhiễu loạn mô phỏng (dùng lại tool M5.2) | ~20% probe |
| **Track ngắn** | Cắt track còn 5–15 frame → `gait = None` hoặc gait từ rất ít frame | ~15% probe |
| **Thân bị che một phần** | Che ngẫu nhiên 30–50% crop thân | ~10% probe |
| **Bình thường** | Giữ nguyên | phần còn lại |

Cách biểu diễn modality thiếu trong ma trận điểm:
- Điểm của modality thiếu: gán giá trị trung tính (ví dụ 0 sau chuẩn hoá) **và** thêm một cờ mask nhị phân `missing[m]`.
- Trọng số QE của modality thiếu: đặt bằng 0.
- Kiểm tra trong `model.py` của repo QME xem họ đã có cơ chế mask chưa; nếu chưa thì thêm `missing` làm đầu vào phụ cho lớp MoE.

Áp dụng **cùng tỉ lệ** cho tập validation để chỉ số đánh giá phản ánh đúng điều kiện có thiếu modality. Tập test thì báo cáo **tách riêng** từng điều kiện (đủ / thiếu mặt / track ngắn).

#### 2B.3 Train hai giai đoạn

**Giai đoạn 1 – Train QE**
- Đầu vào: `inter_feat` của encoder mặt (KP-RPE), theo từng frame hoặc theo track tuỳ cách repo QME cài đặt.
- Đầu ra: trọng số chất lượng `W ∈ (0, 1)`.
- Loss: **pseudo-quality loss** của QME. Nhãn chất lượng không do người gán mà suy ra từ kết quả nhận dạng thực tế: mẫu nào được xếp hạng đúng (người đúng nằm trong top-k, tham số `--rank_threshold` trong repo) thì được coi là chất lượng cao.
- Encoder **đóng băng hoàn toàn**; chỉ train QE.
- Cấu hình tham khảo từ repo: khoảng 5 epoch, tối đa ~6.000 bước.
- Kiểm tra: vẽ phân phối `W` trên ảnh sạch và ảnh đã làm suy giảm. Hai phân phối phải tách nhau rõ; nếu không, thử lớp hook khác.

**Giai đoạn 2 – Train fusion (QME)**
- **Đóng băng QE.**
- Đầu vào: ma trận điểm N_G × 3 (đã qua lớp Norm), trọng số QE, cờ `missing`.
- Mô hình: lớp MoE gồm Z chuyên gia; trọng số QE điều khiển cách cộng đầu ra các chuyên gia thành ma trận điểm cuối S'.
- Loss: **score triplet loss**, đẩy điểm của cặp khác người xuống dưới 0 và điểm của cặp cùng người lên trên biên m.
- Số chuyên gia Z: bắt đầu bằng giá trị mặc định trong config của repo; ablation của bài QME cho thấy tăng Z giúp ích dần.

#### 2B.4 So sánh và chọn bản cuối

So v1 (z-score), 2A và 2B trên **cùng tập test**, báo cáo theo bảng mục 5.4, tách riêng theo điều kiện thiếu modality. Chỉ chọn QME nếu nó tốt hơn v1 ở cả điều kiện đủ modality lẫn thiếu mặt.

**Công sức:** v1 khoảng 3–4 ngày; 2A khoảng 2–3 ngày; 2B khoảng 1.5–2 tuần (bao gồm hook, chia dữ liệu, tạo dữ liệu thiếu modality, hai giai đoạn train).

---

## 5. Protocol đánh giá end-to-end

### 5.1 Bộ dữ liệu
| Bộ | Vai trò | Điểm mạnh |
|---|---|---|
| **MEVID** | Đánh giá chính | Video, nhiều camera, thay quần áo, có cả mặt và thân |
| **CCVID** | Đánh giá chính + train fusion | Video thay quần áo, mặt thường nhìn rõ |
| **AG-ReID.v2** | Góc nhìn cao | Drone + CCTV + kính đeo |
| **CCPG / CCGR-MINI** | Kiểm tra riêng dáng đi | Dữ liệu gait có thay quần áo |
| **MSU-BRC** (nếu xin được) | So sánh trực tiếp với FarSight Public | Gần điều kiện BRIAR nhất |

### 5.2 Biến thể "tầm xa" tự tạo
Để mô phỏng điều kiện BRIAR trên dữ liệu public, tạo thêm các phiên bản probe:
- **Downsample** ×2, ×4, ×8 rồi upsample lại kích thước ban đầu.
- **Nhiễu loạn mô phỏng** bằng simulator (D/r₀ trong khoảng 1–10) hoặc ghép theo phong cách ATSyn.
- **Motion blur** và nén mạnh (JPEG/H.264 bitrate thấp).

Gallery giữ nguyên ảnh sạch, tương tự cách BRIAR dùng ảnh chụp trong nhà ở cự ly gần làm gallery.

### 5.3 Metric
- **Verification:** TAR@0.1%FAR, TAR@1%FAR.
- **Closed-set identification:** Rank-1, Rank-5, Rank-20.
- **Open-set identification:** FNIR@1%FPIR (có thêm người không nằm trong gallery).
- Báo cáo **riêng từng modality** và **sau fusion**, chia theo điều kiện: có mặt / không thấy mặt, sạch / suy giảm.

### 5.4 Bảng kết quả mẫu cần điền
| Cấu hình | TAR@0.1% | Rank-20 | FNIR@1% |
|---|---|---|---|
| Face only | | | |
| Gait only | | | |
| Body only | | | |
| Fusion v1 (z-score) | | | |
| Fusion 2A (QME, encoder theo repo) | | | |
| Fusion 2B (QME, encoder FarSight 2.0) | | | |
| + M2 restore | | | |

Mỗi dòng fusion báo cáo thêm ba cột con theo điều kiện: **đủ modality / thiếu mặt / track ngắn**.

---

## 6. Tích hợp hệ thống

### 6.1 Cấu trúc mã nguồn đề xuất

**Nguyên tắc tổ chức**
1. **Mỗi module (M1–M4) là một folder riêng** trong `farsight/modules/`.
2. **Mỗi model trong module có một folder riêng**, chứa đủ mọi thứ của model đó: wrapper, config, manifest checkpoint, README, test. Thay hoặc nâng cấp một model chỉ đụng vào đúng folder của nó.
3. **Mã gốc của bên thứ ba** nằm trong `third_party/<model>/` (git submodule, khoá theo commit), tách khỏi wrapper của mình.
4. **Trọng số** nằm trong `weights/<module>/<model>/`, không commit vào git; mỗi model khai báo checkpoint trong `manifest.yaml`.
5. Mọi model cùng loại tuân theo **một interface chung** (`farsight/core/interfaces.py`), nên pipeline chỉ gọi interface, không gọi trực tiếp mã của từng repo.

```
farsight-lite/
├── farsight/
│   ├── core/                          # dùng chung, không chứa model
│   │   ├── interfaces.py              # BaseDetector, BaseTracker, BaseRestorer, BaseEncoder, BaseFusion
│   │   ├── types.py                   # Tracklet, ModalityOut, Template, SearchResult (mục 3)
│   │   ├── registry.py                # đăng ký model theo tên, nạp từ config
│   │   ├── hooks.py                   # tiện ích forward hook để lấy inter_feat
│   │   └── weights.py                 # tải + kiểm tra sha256 theo manifest.yaml
│   │
│   ├── modules/
│   │   ├── m1_detect_track/
│   │   │   ├── module.py              # ghép các model của M1 thành một bước
│   │   │   ├── bpjdet/                # detector thân + mặt
│   │   │   ├── yolov8_verifier/       # xác minh hộp thân (ngưỡng 0.7)
│   │   │   ├── bytetrack/             # liên kết theo IoU + Kalman
│   │   │   └── psr_appearance/        # ResNet-18 + patch memory (PSR)
│   │   │
│   │   ├── m2_restore/
│   │   │   ├── module.py
│   │   │   ├── quality_gate/          # quyết định có phục hồi hay không
│   │   │   └── datum/                 # DATUM pretrained (dynamic scene)
│   │   │
│   │   ├── m3_encode/
│   │   │   ├── module.py
│   │   │   ├── face/
│   │   │   │   ├── dfa_aligner/       # 5 landmark + căn chỉnh
│   │   │   │   └── kprpe/             # KP-RPE ViT-B, xuất feat + inter_feat
│   │   │   ├── gait/
│   │   │   │   ├── biggait/           # BigGait checkpoint CCPG
│   │   │   │   └── biggergait/        # BiggerGait checkpoint CCPG/CCGR (thử song song)
│   │   │   └── body/
│   │   │       └── aim/               # AIM ResNet-50, chỉ nhánh danh tính
│   │   │
│   │   └── m4_fusion/
│   │       ├── module.py
│   │       ├── zscore/                # fusion v1, không train
│   │       ├── qe/                    # Quality Estimator (2B: train lại trên KP-RPE)
│   │       └── qme/                   # MoE score-fusion + xử lý modality thiếu
│   │
│   ├── io/                            # đọc video (decord/PyAV), HDF5 template store
│   └── pipeline.py                    # enroll(video) / search(video), chỉ gọi module.py
│
├── third_party/                       # mã gốc, git submodule, khoá commit
│   ├── BPJDet/  ultralytics/  ByteTrack/  DATUM/
│   ├── CVLface/  OpenGait/  AIM-CCReID/  QME_ICCV25/
│
├── weights/                           # KHÔNG commit; tải bằng tools/fetch_weights.py
│   ├── m1_detect_track/{bpjdet,yolov8_verifier,psr_appearance}/
│   ├── m2_restore/datum/
│   ├── m3_encode/{face/dfa_aligner,face/kprpe,gait/biggait,gait/biggergait,body/aim}/
│   └── m4_fusion/{qe,qme}/
│
├── configs/
│   └── pipeline/                      # chọn model cho từng module, vd. v1.yaml, 2A.yaml, 2B.yaml
├── splits/                            # ccvid_qme.json, mevid_qme.json
├── eval/                              # protocol MEVID, CCVID, AG-ReID.v2; metrics.py
├── tools/                             # fetch_weights.py, degrade_dataset.py, export_scores.py
└── service/                           # gRPC server (giai đoạn cuối)
```

**Nội dung bắt buộc trong mỗi folder model** (ví dụ `m3_encode/body/aim/`):
```
aim/
├── __init__.py          # đăng ký model vào registry với tên "aim"
├── model.py             # wrapper, kế thừa BaseEncoder; chỉ import từ third_party/AIM-CCReID
├── preprocess.py        # resize/normalize đúng như repo gốc
├── config.yaml          # kích thước vào, feature dim, lớp hook inter_feat, batch size
├── manifest.yaml        # checkpoint: tên file, nguồn (URL), sha256, dữ liệu train, license, ngày tải
├── README.md            # model gốc là gì, kết quả tái hiện, khác biệt so với bài báo
└── tests/
    └── test_smoke.py    # nạp checkpoint, chạy 1 batch giả, kiểm tra shape + tính xác định
```

Ví dụ `manifest.yaml`:
```yaml
name: aim
module: m3_encode/body
source_repo: https://github.com/BoomShakaY/AIM-CCReID
source_commit: <commit hash>
checkpoints:
  - file: ltcc.pth.tar
    url: <link Google Drive trong README AIM>
    sha256: <tính khi tải>
    trained_on: LTCC
    note: chỉ dùng model_state_dict (nhánh danh tính)
license: "không có file LICENSE trong repo gốc"
output_dim: 4096
```

Ví dụ `configs/pipeline/v1.yaml` (chọn model theo tên folder):
```yaml
m1_detect_track: {detector: bpjdet, verifier: yolov8_verifier, tracker: bytetrack, appearance: psr_appearance}
m2_restore:      {enabled: true, gate: quality_gate, restorer: datum}
m3_encode:       {face: [dfa_aligner, kprpe], gait: biggait, body: aim}
m4_fusion:       {method: zscore}
```

**Quy ước kiểm soát model**
- Đổi model chỉ bằng config (`gait: biggait` → `gait: biggergait`), không sửa `pipeline.py`.
- Thêm model mới = thêm một folder mới cùng cấp, không sửa folder model cũ.
- Mỗi kết quả đánh giá ghi kèm tên config pipeline và sha256 của các checkpoint đã dùng, để tái hiện được.
- `tests/test_smoke.py` của mọi model chạy trong CI trước khi merge.

### 6.2 Luồng xử lý
- `enroll(video|images, subject_id)`: M1 → (M2) → M3 → ghi template vào HDF5.
- `search(video)`: M1 → (M2) → M3 → M4 → SearchResult cho từng track.

### 6.3 Tối ưu tốc độ
- Đọc video bằng **decord/PyAV** có tăng tốc GPU (NVDEC) nếu có.
- Chạy M1 theo batch frame. Các bước M3 chạy **bất đồng bộ** theo track, sau khi track kết thúc hoặc đủ 30 frame.
- Khi có nhiều GPU: mỗi GPU một worker xử lý một video, theo cách FarSight làm.
- Cache feature theo `(video_id, track_id)` để chạy lại M4 mà không phải trích feature lại.

### 6.4 Mục tiêu hiệu năng
| Hạng mục | Mục tiêu |
|---|---|
| End-to-end 1080p | ≥ 5 FPS / GPU (FarSight báo cáo khoảng 7 FPS) |
| Kích thước template | ≤ 0.05 MB / người |
| Search 10k gallery | < 100 ms / probe (cosine trên GPU / FAISS) |

---

## 7. Lộ trình triển khai (ưu tiên tốc độ)

Giả định: **3 kỹ sư**, ký hiệu E1 (detection/tracking/hệ thống), E2 (mặt/phục hồi), E3 (gait/body/fusion).

| Tuần | E1 | E2 | E3 | Mốc |
|---|---|---|---|---|
| 1 | Dựng repo, submodule, môi trường; gửi đơn xin dữ liệu | Tải checkpoint KP-RPE, DFA, DATUM | Tải checkpoint BigGait/BiggerGait; chuẩn bị LTCC/PRCC | Môi trường chạy được mọi repo gốc |
| 2 | BPJDet + YOLOv8 wrapper, tiền xử lý GPU dùng chung | Face encoder + aggregation + **hook `inter_feat`**; kiểm tra trên TinyFace | Gait wrapper (+ hook sẵn); kiểm tra trên CCPG | Từng encoder tái hiện được số liệu gốc |
| 3 | PSR-ByteTrack | Protocol đánh giá + metrics; **cố định file split ID** (`splits/`) | Wrapper AIM, nạp checkpoint; tái hiện số LTCC/PRCC | |
| 4 | Tune tracker trên MOT17/MEVID; lọc tracklet | Dựng protocol MEVID/CCVID + biến thể suy giảm | Fusion v1 (z-score); **chạy lộ trình 2A** với dữ liệu .h5 + checkpoint QE của repo QME | **Mốc A: pipeline end-to-end chạy trên MEVID; 2A tái hiện số liệu QME trên CCVID** |
| 5 | Pipeline enroll/search + HDF5 (lưu cả `inter_feat`) | Quality gate + DATUM (M2) | Đánh giá các checkpoint AIM trên CCVID/MEVID (+ bản làm mờ mặt), chọn checkpoint; viết xử lý modality thiếu trong `m4_fusion/qme/` | |
| 6 | Tối ưu tốc độ (batch, async) | Đánh giá M2: giữ hay tắt; **tạo bộ train QME có thiếu modality** | Gắn body encoder đã chọn vào pipeline (+ hook); fine-tune AIM nếu cần | **Mốc B: bảng kết quả đầy đủ cho v1** |
| 7 | Đánh giá AG-ReID.v2 | Phân tích lỗi mặt (chất lượng thấp) | **2B giai đoạn 1:** export score + `inter_feat`, train QE trên KP-RPE; kiểm tra phân phối W | |
| 8 | Profiling | Fine-tune mặt bằng tăng cường suy giảm (chỉ khi cần) | **2B giai đoạn 2:** đóng băng QE, train QME; so v1 / 2A / 2B theo từng điều kiện thiếu modality | **Mốc C: FarSight-Lite v1.0** |
| 9–10 | gRPC service, đóng gói | (Tuỳ chọn) GRTM | (Tuỳ chọn) fine-tune gait với open-set loss | Bản demo / bàn giao |

**Tổng thời gian:** khoảng 8 tuần cho bản v1.0 chạy hoàn chỉnh, thêm 2 tuần cho các nâng cấp tuỳ chọn.

---

## 8. Tài nguyên tính toán (ước tính)

| Hạng mục | Cấu hình gợi ý | Thời gian ước tính |
|---|---|---|
| Phát triển + suy luận | 1 máy 2–4 GPU 24–48 GB | Suốt dự án |
| (Tuỳ chọn) fine-tune AIM | 2 GPU (script mặc định chạy phân tán) | Khoảng 1 ngày mỗi lần chạy |
| Train QE (giai đoạn 1) | 1 GPU | Vài giờ (~6.000 bước theo config mẫu của repo) |
| Train QME (giai đoạn 2) | 1 GPU | Vài giờ, chỉ chạy trên ma trận điểm |
| Export `inter_feat` + ma trận điểm cho CCVID/MEVID | 1–2 GPU | ~1 ngày (chạy lại 3 encoder trên toàn bộ train/val) |
| (Tuỳ chọn) fine-tune BigGait | 4–8 GPU | 1–3 ngày |
| (Tuỳ chọn) GRTM | 4–8 GPU | Vài ngày đến 1–2 tuần |
| Lưu trữ | 3–5 TB (thêm khoảng 2 TB nếu dựng LMDB cho ATSyn) | |

Các con số trên là ước tính để lên kế hoạch, cần điều chỉnh sau lần chạy đầu tiên.

---

## 9. Checklist tải về

### Mã nguồn
- [ ] hnuzhy/BPJDet
- [ ] ultralytics/ultralytics
- [ ] ifzhang/ByteTrack
- [ ] xg416/DATUM
- [ ] mk-minchul/CVLface
- [ ] ShiqiYu/OpenGait
- [ ] BoomShakaY/AIM-CCReID
- [ ] jiezhu23/QME_ICCV25
- [ ] prevso1088/open-set-biometrics (tuỳ chọn)

### Checkpoint
- [ ] BPJDet `ch_face_l_1536_e150_best_mMR.pt` (HF: HoyerChou/BPJDet)
- [ ] YOLOv8x (COCO)
- [ ] DATUM dynamic scene model
- [ ] KP-RPE ViT-Base WebFace12M (HF: minchul/cvlface_adaface_vit_base_kprpe_webface12m)
- [ ] DFA aligner (HF: minchul/cvlface_DFA_mobilenet)
- [ ] BigGait CCPG + BiggerGait CCPG/CCGR (HF: opengait/OpenGait)
- [ ] AIM checkpoint LTCC + PRCC (Google Drive / Baidu trong README repo AIM)
- [ ] Repo QME (Google Drive trong README): checkpoint QE + QME, backbone AdaFace/BigGait/AIM, dữ liệu tiền xử lý `.h5` (vd. `ccvid.h5`, `ccvid_face.h5`), ma trận điểm `test_feats/scoremats_{dataset}.h5`
- [ ] DINOv2 ViT-L/14 pretrain đặt vào `WBModules/BigGait/pretrained_LVMs/` (yêu cầu của repo QME)

### Dữ liệu
| Ưu tiên | Bộ | Dùng cho |
|---|---|---|
| P0 | MEVID, CCVID | Đánh giá chính; **train QE + QME** (chỉ split train, xem M4 2B.1); train body (chỉ split train) |
| P1 | LTCC, PRCC | Tái hiện số liệu checkpoint AIM; fine-tune AIM nếu cần (PRCC tải qua link trong README AIM, LTCC đăng ký với tác giả) |
| P0 | MOT17 | Tune tracker |
| P1 | AG-ReID.v2 | Đánh giá góc nhìn cao |
| P1 | TinyFace | Kiểm tra face encoder |
| P1 | CCPG, CCGR-MINI | Kiểm tra / fine-tune gait |
| P1 | ATSyn (dynamic) | Đánh giá M2, tạo dữ liệu suy giảm |
| P2 | MSU-BRC | So sánh với FarSight Public |
| P2 | DanceTrack, VisDrone, UAV-Human, DroneSURF, SUSTech1K, DroneGait | Đánh giá bổ sung |
| P3 | WebFace4M | Chỉ khi cần fine-tune mặt |

---

## 10. Rủi ro kỹ thuật và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| BPJDet bỏ sót người rất nhỏ hoặc nhìn từ trên cao | Recall thấp trên VisDrone/UAV-Human | Tăng độ phân giải đầu vào, tile ảnh; nếu vẫn kém thì fine-tune với nhãn giả |
| Ghép thân–mặt sai khi đông người | Feature mặt và thân trong cùng track không nhất quán | Siết ngưỡng inner IoU; chỉ nhận mặt có score cao |
| PSR gán nhầm người mặc giống nhau | Số ID switch tăng sau khi bật PSR | Thay ResNet-18 bằng model ReID; thêm điều kiện khoảng cách không gian |
| DATUM làm mất danh tính | Similarity mặt sau phục hồi giảm | Kiểm tra an toàn ở M2; tắt M2 |
| Checkpoint gait lệch miền (CCPG ≠ drone) | Gait kém trên AG-ReID.v2 / DroneGait | Fine-tune BigGait trên CCGR-MINI + DroneGait với open-set loss |
| AIM lệch miền (train trên ảnh tĩnh LTCC/PRCC) | Body kém trên CCVID/MEVID/AG-ReID.v2 | Fine-tune AIM trên CCVID + LTCC + PRCC |
| AIM dựa nhiều vào khuôn mặt | mAP giảm mạnh trên bản làm mờ mặt; điểm body tương quan cao với điểm mặt | Fusion vẫn giữ; dựa vào QE/QME để giảm trọng số body khi trùng tín hiệu với mặt; cân nhắc fine-tune AIM với ảnh làm mờ mặt |
| Feature AIM 4096-d làm template nặng | Template > 0.05 MB/người | Chỉ giữ nửa avg-pool 2048-d nếu mất ≤ 1 điểm mAP |
| QME không tốt hơn z-score | Metric fusion v2 ≤ v1 | Giữ z-score + trọng số chất lượng |
| QE không phân biệt được chất lượng trên KP-RPE | Phân phối W của ảnh sạch và ảnh suy giảm chồng lên nhau | Thử lớp hook khác (nông hơn); hoặc tạm dùng norm feature của KP-RPE làm trọng số chất lượng (bài QME có so sánh phương án dùng norm AdaFace) |
| Rò dữ liệu giữa train QME và test | Kết quả QME trên test cao bất thường so với validation | Kiểm tra lại file `splits/`; đảm bảo encoder (đặc biệt AIM nếu fine-tune trên CCVID) không thấy ID test |
| QME quá khớp vào CCVID/MEVID (ít ID) | Tốt trên CCVID nhưng kém trên AG-ReID.v2 | Giảm số chuyên gia Z; tăng tỉ lệ mẫu thiếu modality/suy giảm; giữ v1 làm phương án dự phòng cho miền mới |
| Chênh lệch lớn giữa dữ liệu public và thực tế | Kết quả tốt trên MEVID nhưng kém trên video thật | Thu thập một tập kiểm thử nội bộ nhỏ theo đúng kịch bản sử dụng; fine-tune có chọn lọc |

---

## 11. Định nghĩa "hoàn thành" cho v1.0

- [ ] `search(video)` chạy end-to-end, trả kết quả cho từng track, có xử lý open-set.
- [ ] Hệ thống vẫn chạy được khi thiếu bất kỳ modality nào.
- [ ] Có bảng kết quả theo mục 5.4 trên MEVID, CCVID và AG-ReID.v2, gồm cả biến thể suy giảm.
- [ ] Fusion tốt hơn modality đơn tốt nhất trên điều kiện "không thấy mặt".
- [ ] Encoder mặt (và tuỳ chọn gait/body) xuất được `inter_feat` trong cùng một lần forward.
- [ ] QE (2B) train xong trên KP-RPE; phân phối W tách rõ giữa ảnh sạch và ảnh suy giảm.
- [ ] File split ID cố định, không có ID nào vừa ở train QME/encoder vừa ở test.
- [ ] Kết quả fusion báo cáo riêng cho ba điều kiện: đủ modality / thiếu mặt / track ngắn.
- [ ] Đạt ≥ 5 FPS end-to-end ở 1080p trên 1 GPU.
- [ ] Mọi bước dựng lại được bằng config và script (`make eval DATASET=mevid`).