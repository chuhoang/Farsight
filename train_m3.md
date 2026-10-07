# Plan phụ: Train lại nhánh mặt (M3a) cho khuôn mặt tầm xa

> Bối cảnh: nhánh mặt hiện dùng KP-RPE ViT-B checkpoint WebFace12M, chạy zero-shot. Mặt đang là modality mạnh nhất trên CCVID sạch (GR-R1 93.3), nhưng tụt mạnh khi ảnh suy giảm (CCVID suy giảm: mặt chỉ còn 15.8) và trên MEVID (29.7).
>
> FarSight train mô hình mặt trên dữ liệu tầm xa BRIAR (3.194 người) **cộng** WebFace12M. Không có BRIAR, plan này thay phần dữ liệu tầm xa bằng **dữ liệu công khai kết hợp suy giảm nhân tạo**, và thêm ràng buộc để feature của ảnh suy giảm bám theo feature của ảnh sạch.

---

## 0. Mục tiêu và tiêu chí thành công

| Hạng mục | Hiện tại | Mục tiêu |
|---|---|---|
| Mặt, CCVID suy giảm, GR-R1 | 15.8 | Tăng rõ rệt (đề xuất ≥ 35) |
| Mặt, MEVID test, GR-R1 | 29.7 | ≥ 35 |
| Mặt, CCVID sạch, GR-R1 | 93.3 | Không giảm quá 1 điểm |
| Benchmark mặt sạch (IJB-B/C hoặc tương đương) | theo checkpoint | Không giảm quá 1 điểm (chống quên) |
| TinyFace Rank-1 | theo checkpoint | Tăng |
| QE mặt (AUC sạch / suy giảm) | 0.98 | Giữ ≥ 0.95 sau khi train lại QE |

Các mục tiêu số là mốc đề xuất để ra quyết định, không phải số liệu đã công bố.

**Nguyên tắc:**
1. Đo kỹ theo kích thước mặt trước, train sau.
2. Khởi tạo từ checkpoint KP-RPE WebFace12M, không train từ đầu.
3. Mọi lựa chọn (checkpoint, siêu tham số) quyết định trên validation, không trên test.
4. Nhánh mặt mới đặt trong folder riêng (`m3_encode/face/kprpe_lr/`), giữ bản cũ để quay lui.

---

## Giai đoạn A – Chẩn đoán theo khoảng cách

### A1. Đo kích thước mặt thực tế
- Với mọi crop mặt trong CCVID test, CCVID suy giảm và MEVID test, tính **khoảng cách giữa hai mắt (IPD)** từ landmark của DFA, và chiều cao crop mặt.
- Vẽ phân bố IPD cho từng tập. Tham chiếu: BRIAR có IPD trong khoảng 15–100 pixel.

### A2. Đánh giá phân tầng
Chia query theo IPD, báo GR-R1 và TAR@0.1%FAR của mặt cho từng nhóm:

| Nhóm IPD | Ý nghĩa |
|---|---|
| < 10 px | Gần như không dùng được |
| 10–20 px | Tầm xa nặng |
| 20–40 px | Tầm xa vừa |
| > 40 px | Cự ly gần |

### A3. Tách lỗi căn chỉnh và lỗi nhận dạng
- Với mỗi nhóm IPD, so hai cấu hình: (a) landmark từ DFA trên ảnh suy giảm; (b) landmark lấy từ ảnh sạch tương ứng (chỉ làm được với CCVID suy giảm, vì có cặp ảnh sạch).
- Nếu (b) tốt hơn (a) nhiều → điểm nghẽn là **landmark/căn chỉnh**, cần làm thêm giai đoạn C3.
- Nếu (a) ≈ (b) → điểm nghẽn là **encoder**, tập trung vào giai đoạn C1–C2.

### A4. Kiểm tra phát hiện mặt (M1)
- Đo tỉ lệ track có `face_box` theo nhóm kích thước người. Nếu BPJDet bỏ sót mặt nhỏ thì cải thiện encoder cũng không giúp được các track đó. Ghi lại để xử lý riêng ở M1 (ngoài phạm vi plan này).

---

## Giai đoạn B – Dữ liệu train và dữ liệu đánh giá

### B1. Danh sách dataset

**Dataset dùng để train**

| Dataset | Quy mô | Đặc điểm | Link | Truy cập |
|---|---|---|---|---|
| WebFace4M / WebFace12M | 4M / 12M ảnh | Mặt rõ, nhiều danh tính; nguồn train của checkpoint KP-RPE gốc. Dùng để chống quên | https://www.face-benchmark.org/download.html | Xin link + mật khẩu, chỉ dùng học thuật |
| QMUL-SurvFace | 463.507 ảnh, 15.573 ID | Mặt giám sát **thật**, độ phân giải thấp, gom từ 17 bộ ReID công khai; có protocol tập mở | https://qmul-survface.github.io/ | Tải trực tiếp (Google Drive / Baidu), kèm code đánh giá |
| TinyFace | 169.403 ảnh, 5.139 ID | Mặt độ phân giải thấp **thật** từ web, trung bình 20×16 px; có split train/test | https://qmul-tinyface.github.io/ | Tải trực tiếp (Google Drive / Baidu) |
| Crop mặt CCVID train | 75 ID | Mặt trong video, đổi quần áo | https://github.com/guxinqian/Simple-CCReID (link tải CCVID trong README) | Tải trực tiếp |
| Crop mặt MEVID train | 84 ID train + 20 ID val | Mặt nhỏ, nhiều camera, ngoài trời | Trang dữ liệu MEVID (link trong bài MEVID, WACV 2023) | Theo điều khoản MEVID |

**Dataset dùng để đánh giá**

| Dataset | Quy mô | Đặc điểm | Link | Truy cập |
|---|---|---|---|---|
| TinyFace test | theo split chính thức | Nhận dạng 1:N mặt độ phân giải thấp | https://qmul-tinyface.github.io/ (có sẵn trong toolkit CVLface) | Tải trực tiếp |
| QMUL-SurvFace test | theo split chính thức | Xác minh và nhận dạng **tập mở** trên mặt giám sát | https://qmul-survface.github.io/ | Tải trực tiếp |
| DroneSURF | 58 người, 200 video, 411K frame, 786K mặt có nhãn | Mặt từ **drone**, hai protocol: active và passive; 34 người train, phần còn lại test; mỗi người có 4 ảnh HR làm gallery | https://iab-rubric.org/index.php/dronesurf | Ký license agreement với nhóm IAB Lab |
| UCCS | ~16.000 mặt, ~1.700 ID | Camera giám sát tầm xa độ phân giải cao, người không biết bị chụp; có protocol **tập mở** (watchlist) | https://vast.uccs.edu/Opensetface/ | Đăng ký với nhóm tác giả; có thời điểm bị tạm ngưng phát hành, cần kiểm tra tình trạng hiện tại |
| IJB-B / IJB-C | | Benchmark mặt chuẩn, dùng để **chống quên** | Có sẵn trong toolkit CVLface (bản đã căn chỉnh) | Qua form của CVLface |
| SCface, IJB-S, LDHF | | Mặt theo khoảng cách có kiểm soát / video giám sát | Cần tìm trang chính thức | Xin quyền, tuỳ chọn |
| CCVID test, MEVID test (crop mặt) | 151 / 54 ID | Đánh giá trong pipeline của hệ thống | như trên | |

Quy tắc chung:
- Đánh lại ID theo tiền tố nguồn (`webface_…`, `survface_…`, `tinyface_…`, `ccvid_…`, `mevid_…`) để không trùng.
- Không ID nào trong tập test (CCVID test, MEVID test, test của TinyFace / QMUL-SurvFace / DroneSURF) xuất hiện trong bất kỳ tập train nào.
- **Kiểm tra trùng nguồn:** QMUL-SurvFace được gom từ 17 bộ ReID công khai. Đối chiếu danh sách nguồn với các bộ dùng để test (ví dụ các bộ ReID video) trước khi dùng, để tránh rò rỉ.
- Ghi điều khoản sử dụng của từng bộ vào `THIRD_PARTY.md`.

### B2. Pipeline suy giảm "tầm xa"
Áp lên ảnh mặt sạch để tạo cặp (sạch, suy giảm). Mỗi ảnh chọn ngẫu nhiên một tổ hợp:

| Phép suy giảm | Khoảng tham số gợi ý | Mô phỏng |
|---|---|---|
| Downsample rồi upsample | Đưa IPD về 8–40 px | Khoảng cách xa |
| Blur Gaussian / motion blur | sigma 0.5–3; motion 3–15 px | Lấy nét kém, chuyển động |
| Nhiễu loạn khí quyển | Simulator P2S/ATSyn, hoặc trường dịch chuyển mượt + blur theo vị trí, D/r₀ 1–6 | Nhiễu loạn |
| Nén | JPEG 10–50, hoặc H.264 bitrate thấp | Video giám sát |
| Nhiễu cảm biến | Gauss + Poisson nhẹ | Ánh sáng yếu |
| Thay đổi màu / độ sáng | gamma, contrast | Điều kiện chiếu sáng |
| Xoay, nghiêng | yaw/pitch nhỏ qua biến đổi affine | Góc nhìn từ cao |

- Chỉ áp lên ảnh **sạch** (WebFace, crop mặt CCVID gần). Không áp thêm lên QMUL-SurvFace / TinyFace vì đã là ảnh độ phân giải thấp thật.
- Tỉ lệ trong batch: khoảng 50% ảnh sạch, 50% ảnh suy giảm (điều chỉnh theo kết quả).
- Lưu tham số suy giảm của từng ảnh để phân tích sau.
- Dùng **cùng pipeline** để tạo CCVID suy giảm cho đánh giá, nhưng **seed khác**.

### B3. Tạo dataset train cho KP-RPE (định dạng CVLface)

CVLface đọc dữ liệu train dạng `.rec/.idx/.tsv` (giống MXNet RecordIO), kèm file landmark `ldmk_5points.csv` cho các model cần landmark như KP-RPE. Mỗi dataset là một thư mục dưới `$DATA_ROOT`, khai báo bằng một file yaml.

**Bước 1 – Chuẩn bị ảnh theo cấu trúc "mỗi ID một thư mục"**
```
$DATA_ROOT/longrange_face_v1/raw/
├── survface_00001/  img_0001.jpg  img_0002.jpg ...
├── tinyface_00001/  ...
├── ccvid_00001/     ...
└── mevid_00001/     ...
```
- Mọi ảnh là mặt **đã căn chỉnh 112×112 RGB**, cùng chuẩn với checkpoint KP-RPE.
- **Crop mặt từ CCVID/MEVID:** chạy đúng pipeline của hệ thống (M1 → crop mặt có nới biên → aligner DFA → 112×112), để dữ liệu train giống dữ liệu lúc chạy. Mỗi tracklet lấy tối đa khoảng 20 frame, cách đều, để tránh trùng lặp.
- **QMUL-SurvFace / TinyFace:** ảnh gốc rất nhỏ. Phóng lên rồi căn chỉnh bằng DFA. Với ảnh mà DFA không cho landmark tin cậy, dùng căn chỉnh tâm cố định và đánh dấu trong metadata.
- Lưu `metadata.csv`: `path, id, source, ipd_px, is_degraded, degradation_params`.

**Bước 2 – Đóng gói thành `.rec`**
```bash
cd cvlface/data_utils/recognition/training_data
python bundle_images_into_rec.py --source_dir $DATA_ROOT/longrange_face_v1/raw
# tạo train.rec / train.idx / train.tsv; nhãn số được gán theo tên thư mục
```

**Bước 3 – Tạo landmark cho KP-RPE**
```bash
python predict_landmark.py --source_dir $DATA_ROOT/longrange_face_v1
# tạo ldmk_5points.csv
```
Với mặt rất nhỏ, kiểm tra ngẫu nhiên vài trăm landmark bằng mắt. Nếu sai nhiều, dùng landmark từ ảnh sạch tương ứng (với ảnh suy giảm nhân tạo) hoặc giữ landmark của căn chỉnh tâm cố định.

**Bước 4 – Khai báo dataset**
```yaml
# cvlface/research/recognition/code/run_v1/dataset/configs/longrange_face_v1.yaml
data_root: ${oc.env:DATA_ROOT}
rec: 'longrange_face_v1'
color_space: 'RGB'
num_classes: <số ID sau khi gộp>
num_image: <số ảnh>
repeated_sampling_cfg: null
semi_sampling_cfg: null
```

**Bước 5 – Trộn với WebFace**
- Cách đơn giản nhất: gộp thư mục ảnh của tập con WebFace4M (đã có landmark) vào cùng `raw/` trước khi đóng gói, với tiền tố `webface_`. Điều chỉnh tỉ lệ trộn bằng số ảnh lấy từ mỗi nguồn.
- Nếu cần lấy mẫu theo tỉ lệ cố định mỗi batch: viết một sampler riêng trong `dataset/` dựa trên cột `source` của metadata.

**Bước 6 – Suy giảm**
- **Online:** CVLface đã có augmenter `basic` với `low_res_augmentation_prob` (mặc định 0.2). Tăng xác suất này và bổ sung một augmenter mới (`longrange`) cho blur, nhiễu loạn, nén theo bảng B2.
- **Offline cặp sạch/suy giảm (cho L_cons ở C2):** dataset trả về cả ảnh sạch và ảnh suy giảm của cùng mẫu, cùng landmark (biến đổi theo phép affine nếu có).

### B4. Tạo dữ liệu đánh giá

| Bộ | Cách chuẩn bị | Protocol / metric |
|---|---|---|
| Toolkit CVLface (LFW, AgeDB, CFP-FP, CPLFW, CALFW, IJB-B, IJB-C, TinyFace) | Tải toolkit (`facerec_val`) theo `README_EVAL_TOOLKIT.md` của CVLface; mật khẩu lấy qua form của tác giả | Chạy sẵn bằng `evaluations/configs/full.yaml`: verification, IJB-B/C TAR@FAR, TinyFace Rank-k |
| QMUL-SurvFace test | Dùng code đánh giá đi kèm dataset; căn chỉnh ảnh như ở B3 | Xác minh (TAR@FAR) và nhận dạng tập mở (TPIR@FPIR) theo protocol chính thức |
| DroneSURF | Crop mặt theo nhãn có sẵn → DFA → 112×112; gallery là 4 ảnh HR mỗi người | Protocol active và passive của dataset (Rank-1, xác minh) |
| UCCS | Theo protocol watchlist của challenge | Nhận dạng tập mở |
| CCVID / MEVID (crop mặt) | Từ pipeline M1 → DFA của hệ thống, lưu template theo track | GR-R1, TAR@0.1%FAR, FNIR@1%FPIR, phân tầng IPD |

- Viết thêm một evaluation type mới trong CVLface (`evaluations/`) cho QMUL-SurvFace và DroneSURF, để chạy cùng lúc với toolkit sẵn có sau mỗi epoch.
- Cố định file protocol (danh sách cặp, gallery/probe) và lưu vào `splits/face_eval/`.

---

## Giai đoạn C – Train

### C1. Fine-tune KP-RPE với AdaFace loss (cấu hình cơ bản)
| Thành phần | Thiết lập |
|---|---|
| Khởi tạo | KP-RPE ViT-B WebFace12M (CVLface) |
| Loss | AdaFace (như checkpoint gốc), head phân loại mới cho tập ID gộp |
| Dữ liệu | B1 + suy giảm B2 |
| Sampler | Trộn nguồn theo tỉ lệ cố định, ví dụ WebFace 50% / QMUL-SurvFace 20% / TinyFace 10% / CCVID 10% / MEVID 10% |
| Learning rate | Nhỏ cho backbone (khoảng 1/10 LR gốc), lớn hơn cho head mới; warmup ngắn, cosine decay |
| Đầu vào | 112×112 + 5 landmark, đúng chuẩn của checkpoint |

AdaFace tự điều chỉnh margin theo chất lượng ảnh (dựa trên norm feature), nên phù hợp sẵn với dữ liệu trộn sạch và suy giảm.

### C2. Ràng buộc nhất quán sạch–suy giảm (teacher–student)
Theo tinh thần DaliID và phần đồng tối ưu của FarSight:
- **Teacher:** KP-RPE gốc, đóng băng, nhận ảnh **sạch**.
- **Student:** KP-RPE đang train, nhận ảnh **suy giảm** của cùng ảnh đó.
- Loss bổ sung: `L_cons = 1 − cos(f_student(ảnh suy giảm), f_teacher(ảnh sạch))`.
- Tổng loss: `L = L_adaface + λ · L_cons`, thử λ ∈ {0.5, 1, 2}.

Mục đích: kéo feature của mặt mờ về gần feature của mặt rõ cùng người, đúng kịch bản gallery sạch, probe tầm xa. Đồng thời giữ không gian feature tương thích với gallery đã enroll bằng checkpoint gốc.

### C3. Landmark cho mặt nhỏ (chỉ làm nếu A3 cho thấy căn chỉnh là điểm nghẽn)
- Fine-tune aligner DFA trên ảnh suy giảm, dùng landmark của ảnh sạch tương ứng làm nhãn.
- Hoặc tăng cường **nhiễu landmark** khi train KP-RPE (dịch ngẫu nhiên 1–3 px), để encoder chịu được landmark sai. KP-RPE vốn được thiết kế để bền với căn chỉnh lệch, nên đây là bổ sung rẻ.

### C4. Thứ tự thí nghiệm (mỗi bước chỉ đổi một yếu tố)

| Run | Thay đổi | Mục đích |
|---|---|---|
| R0 | Checkpoint gốc | Mốc |
| R1 | C1, chỉ WebFace + suy giảm | Hiệu quả của riêng tăng cường suy giảm |
| R2 | R1 + QMUL-SurvFace + TinyFace (mặt nhỏ thật) | Hiệu quả của mặt độ phân giải thấp thật |
| R2b | R2 + crop mặt CCVID/MEVID train | Hiệu quả của dữ liệu đúng miền |
| R3 | R2b + L_cons (C2) | Hiệu quả của ràng buộc sạch–suy giảm |
| R4 | R3 + nhiễu landmark (C3) | Chỉ khi A3 cho thấy cần |

Chọn run tốt nhất theo **validation**: MEVID val (20 ID) + CCVID val (cắt từ ID train) + bản suy giảm của chúng. Chạy 3 seed cho run được chọn.

### C5. Chống quên
- Sau mỗi lần đánh giá: chạy toolkit CVLface (IJB-B/C, TinyFace), QMUL-SurvFace và CCVID sạch.
- Dừng hoặc giảm LR nếu benchmark sạch giảm quá 1 điểm.

---

## Giai đoạn D – Tích hợp lại

1. Đặt model mới vào `farsight/modules/m3_encode/face/kprpe_lr/`, kèm `manifest.yaml` ghi run, dữ liệu train, sha256.
2. **Enroll lại gallery** bằng model mới. Không trộn template cũ và mới vì không gian feature đã thay đổi.
3. **Hook `inter_feat`:** giữ đúng lớp hook như bản cũ, rồi **train lại QE** mặt (feature trung gian đã đổi), kiểm tra AUC sạch/suy giảm.
4. Xuất lại ma trận điểm, chạy fusion v1, **train lại QME** (3 seed).
5. Nếu M2 (DATUM) đang bật: đo lại xem phục hồi còn giúp ích không. Encoder bền hơn có thể làm lợi ích của phục hồi giảm, và M2 có thể tắt được.

---

## Giai đoạn E – Đánh giá

### E1. Bảng kết quả
- Format hiện tại (5 chỉ số × 3 điều kiện) trên CCVID sạch, CCVID suy giảm, MEVID.
- Thêm dòng **mặt đơn lẻ: R0 so với model mới**.

### E2. Phân tầng theo IPD
- Lặp lại bảng A2 với model mới, để thấy cải thiện tập trung ở nhóm IPD nào.

### E3. Benchmark mặt công khai
- TinyFace (Rank-1), QMUL-SurvFace (TAR@FAR, TPIR@FPIR tập mở), DroneSURF (active / passive).
- UCCS và IJB-S nếu có quyền truy cập.
- IJB-B/C và các bộ verification trong toolkit CVLface để kiểm tra không quên.

### E4. Độ tin cậy
- Bootstrap theo ID (5.000 lần) cho chênh lệch R0 so với model mới.
- Báo số query hợp lệ và số query bị bỏ.

---

## Rủi ro

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Quên năng lực trên mặt rõ | CCVID sạch hoặc IJB-B/C giảm > 1 điểm | Tăng tỉ lệ WebFace; giảm LR backbone; tăng λ của L_cons |
| Suy giảm nhân tạo không giống thật | Tốt trên CCVID suy giảm nhưng không cải thiện MEVID | Tăng tỉ lệ QMUL-SurvFace, TinyFace và crop MEVID (mặt nhỏ thật); điều chỉnh pipeline B2 theo phân bố IPD đo ở A1 |
| Quá khớp vào CCVID/MEVID (ít ID) | Validation tốt, test kém | Giảm tỉ lệ dữ liệu đúng miền; dừng sớm theo validation |
| Mặt quá nhỏ, không còn thông tin | Nhóm IPD < 10 px không cải thiện | Chấp nhận; để QE hạ trọng số mặt và dựa vào body |
| Landmark sai trên mặt nhỏ | A3 cho thấy chênh lệch lớn | Làm C3 |
| Không gian feature thay đổi | Template gallery cũ không khớp | Bắt buộc enroll lại gallery (D2) |

---

## Quay lui

```yaml
m3_encode: {face: [dfa_aligner, kprpe],    gait: biggait, body: csci_video}  # cũ
m3_encode: {face: [dfa_aligner, kprpe_lr], gait: biggait, body: csci_video}  # mới
```
