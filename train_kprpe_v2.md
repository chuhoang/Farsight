# Plan: train tiếp KP-RPE tầm xa (kprpe_lr, vòng 2)

> Tiếp nối `train_m3.md`. Code đã có: `eval/face_cache.py`, `eval/face_eval.py`, `farsight/modules/m3_encode/face/kprpe_lr/` (train, degrade, README).
> Vòng 1 cho thấy fine-tune trên ít ID bị **overfit**. Vòng 2 thêm QMUL-SurvFace (nhiều ID mặt giám sát thật) và giữ mô hình gần bản gốc hơn.

---

## 0. Hiện trạng (số đã đo)

### 0.1. Mốc R0: checkpoint gốc KP-RPE WebFace12M, chỉ dùng mặt, GR-R1 theo luật CAL

| Tập | GR-R1 | Ghi chú |
|---|---|---|
| CCVID test | 95.1 | Khớp với số của store, cache tái tạo đúng pipeline |
| CCVID test suy giảm (down:4 turb:2 jpeg:30) | 14.3 | 219/834 track mất mặt hoàn toàn |
| MEVID test | 29.4 | 144/316 probe không có mặt |
| Điểm val (trung bình 3 tập) | 66.48 | CCVID val 97.8 / CCVID val suy giảm 52.2 / MEVID val 49.4 |

### 0.2. Chẩn đoán giai đoạn A

| Hạng mục | Kết quả |
|---|---|
| A1: IPD trung vị theo track | CCVID 7.7 px; CCVID suy giảm khoảng 2.3 px hiệu dụng; MEVID 5.9 px. Gần như mọi track đều dưới 20 px |
| A2: R1 theo IPD (MEVID) | dưới 10 px: 46.0; 10–20 px: 70.5. Mặt càng nhỏ càng kém, như dự kiến |
| A3: căn chỉnh (CCVID suy giảm) | (a) DFA trên crop suy giảm: 15.9; (b) căn chỉnh của ảnh sạch: 24.0 |
| A4: tỉ lệ phát hiện mặt theo chiều cao người | CCVID suy giảm: dưới 100 px chỉ 7.7%, 150–200 px là 66.9%. MEVID: dưới 100 px là 19.8% |

**Đọc A3 cho đúng:** phần lớn chênh lệch 8 điểm đến từ số mặt bị loại, không phải độ chính xác của landmark.
- Ở (a), 246 track mất mặt vì điểm DFA trên ảnh suy giảm rơi dưới `min_face_score = 0.5`. Ở (b) chỉ có 2 track.
- Chỉ xét các track có mặt (nhóm dưới 10 px), (a) đạt 22.5 còn (b) đạt 24.0, chỉ hơn 1.6 điểm.

Kết luận: **ngưỡng điểm DFA đang vứt bỏ mặt mờ**. Cần làm bước 2.4 trước khi đầu tư cho landmark (R4).

### 0.3. Vòng 1 (đã chạy)

| Run | Dữ liệu | Kết quả | Kết luận |
|---|---|---|---|
| R1 (dừng ở step 2000) | chỉ CCVID, có suy giảm | Val 66.5 → 62.1. MEVID val 49 → 31.8 | Học thuộc 60 ID của CCVID |
| R2b (crash CUBLAS ở step 3500, LR 1e-5, lambda_kd 1) | CCVID + MEVID + TinyFace, 2.714 ID | Step 1000: **68.12** (tốt nhất, đã lưu `r2b_s0.pt`). Step 2000: 65.5. Step 3000: 66.0. MEVID val 50 → 41.5 | Overfit sau khoảng 2 epoch |

**Dấu hiệu overfit trong log R2b:**
- Loss cls giảm 12.2 → 2.7, loss kd tăng 0.05 → 0.24, nghĩa là feature trôi khỏi mô hình gốc.
- Trong khi đó MEVID val giảm.

**Nguyên nhân:** MEVID chỉ có 84 ID train (104 ID train chính thức trừ 20 ID val), CCVID có 60 ID, TinyFace có 2.570 ID nhưng ít ảnh mỗi ID. Head AdaFace có quá ít lớp nên backbone học thuộc người thay vì học đặc trưng mặt tầm xa.

### 0.4. Dữ liệu mới

**QMUL-SurvFace** đã có tại `~/datasets/survface/QMUL-SurvFace/`:
- `training_set/`: **5.319 ID, 220.888 ảnh**, kích thước trung vị 27×22 px (p10: 13×10, p90: 33×27). Mặt giám sát thật, đúng miền ta cần.
- `Face_Identification_Test_Set/`: gallery 60.294, mated_probe 60.423, unmated_probe 121.736. Đây là benchmark tập mở theo protocol chính thức.
- Không trùng ID với test của ta: SurvFace (2018) gom từ các bộ ReID có trước cả CCVID (2022) và MEVID (2023).

---

## 1. Mục tiêu vòng 2

| Hạng mục | R0 | Mục tiêu |
|---|---|---|
| Điểm val (chọn checkpoint) | 66.48 | ≥ 70 |
| MEVID val | 49.4 | Không giảm; mục tiêu ≥ 52 |
| CCVID val suy giảm | 52.2 | ≥ 58 |
| CCVID val sạch | 97.8 | Không giảm quá 1 điểm |
| Sau khi chọn (chỉ báo, không dùng để chọn) | | MEVID test ≥ 32, CCVID test suy giảm ≥ 20, CCVID test sạch ≥ 94 |
| TinyFace Rank-1 / SurvFace TPIR@FPIR | theo R0 | Không giảm |

---

## 2. Các bước

### 2.1. Sửa hạ tầng trước khi chạy dài
1. **Chain không chạy tiếp khi run trước lỗi.** Trong `eval/run_kprpe_lr.sh`:
   - kiểm tra `${PIPESTATUS[0]}`;
   - nếu run lỗi và còn file `.last.pt` thì tự resume một lần (lỗi CUBLAS ở R2b là lỗi nhất thời; resume là đủ);
   - nếu vẫn lỗi thì dừng cả chain.
2. **Luôn chạy bằng `Start-Process wsl.exe`**, tách khỏi phiên Claude. Mọi run phải resume được (đã có).
3. **Lưu cả điểm val của checkpoint tốt nhất** vào log JSON (đã có) và in rõ ra log.

### 2.2. Cache SurvFace train (B3)
```bash
bash run.sh eval.face_cache --folder ~/datasets/survface/QMUL-SurvFace/training_set --out survface_train
```
- Khoảng 220k ảnh qua DFA, ước tính 30–45 phút GPU, cache khoảng 8 GB.
- **Đo trước khi dùng:** phân bố điểm DFA trên SurvFace (mặt 13–33 px). Nếu nhiều ảnh dưới 0.5, so hai cách:
  - (a) ngưỡng thấp hơn riêng cho nguồn này (`min_score` theo từng nguồn);
  - (b) với ảnh DFA không tin cậy, dùng căn chỉnh tâm cố định với landmark của template.

  Kiểm tra bằng mắt khoảng 200 ảnh ngẫu nhiên.
- Thêm vào `kprpe_lr/config.yaml`:
  `survface: {cache: survface_train, weight: 0.4, degrade: 0.0}` (đã là ảnh độ phân giải thấp thật, không suy giảm thêm).

### 2.3. Cache thêm MEVID train
- Cache toàn bộ 6.339 tracklet của MEVID train (hiện chỉ có 1.949, lấy mẫu theo ID). Cần store hoặc danh sách tracklet đầy đủ: thêm `--all` cho `face_cache`, lấy tracklet thẳng từ `mevid_tracklets("train")` thay vì từ meta của store.
- Lợi ích: khoảng 3 lần số crop cho cùng 84 người (nhiều camera, khoảng cách hơn). Không thay được việc có thêm ID.
- Giữ nguyên 20 ID val. Chỉ mở rộng crop của 84 ID train.

### 2.4. Ngưỡng điểm DFA cho mặt mờ (từ A3, không cần train)
- Quét `min_face_score` ∈ {0.5, 0.3, 0.2, 0.1, 0} trên **val** (`face_eval --val`, R0), và trên tập `ccvid_val_degraded` để đo coverage.
- Nếu ngưỡng thấp tăng điểm val thì cập nhật `kprpe/config.yaml`, và đo lại mốc R0 với ngưỡng mới cho mọi run sau.
- Rẻ: chỉ chạy lại eval từ cache (khoảng 15 phút).

### 2.5. Cấu hình train vòng 2 (giữ gần bản gốc)

| Tham số | Vòng 1 (R2b) | Vòng 2 | Lý do |
|---|---|---|---|
| `lr` backbone | 1e-5 | **3e-6** | Overfit từ khoảng step 1500 |
| `head_lr` | 1e-3 | 1e-3 | Head mới, nhiều lớp hơn |
| `lambda_kd` | 1 | **5** | kd tăng lên 0.24 ở vòng 1, feature trôi xa bản gốc |
| `lambda_cons` | 0 | **1** | Ràng buộc sạch–suy giảm (C2), đúng mục tiêu của plan gốc |
| Nguồn / trọng số | ccvid .3, mevid .3, tinyface .4 | **survface .4, ccvid .2, mevid .2, tinyface .2** | SurvFace là nguồn ID chính |
| `batch` | 64 | 64 | 3.75 GB với checkpoint theo block |
| `steps` | 16.000 | **16.000** (xem 2.6) | |
| `eval_every` | 1.000 | 1.000 | |
| Giữ nguyên | | AdaFace m=0.4, h=0.333; warmup 500; cosine; bf16; grad clip 5 | |

### 2.6. Số epoch: cần quyết định
Pool vòng 2 khoảng **255k crop** (SurvFace 221k + 34k hiện có), nên 1 epoch ≈ 4.000 step.

| Lựa chọn | Số step | Thời gian (khoảng 1.5 s/step) |
|---|---|---|
| 30 epoch trên toàn pool | khoảng 120.000 | khoảng **50 giờ** mỗi run |
| 30 epoch tính trên dữ liệu đúng miền (CCVID + MEVID), SurvFace là nguồn bổ trợ | 16.000 (≈ 4 epoch toàn pool) | khoảng 7 giờ |

**Đề xuất:** chạy 16.000 step trước. Nếu điểm val vẫn tăng ở cuối run (chưa bão hoà) thì mới kéo dài lên 40–60k step (resume với lịch mới không được vì cosine phụ thuộc tổng số step, nên phải chạy lại). Checkpoint chọn theo val nên run dài không có hại, chỉ tốn thời gian.

### 2.7. Thứ tự run (mỗi run đổi một yếu tố, chọn theo val)

| Run | Thay đổi so với run trước | Câu hỏi trả lời |
|---|---|---|
| S0 | R0 với ngưỡng DFA mới (2.4) | Mốc mới |
| **S1** | Cấu hình 2.5, **không** SurvFace | Ràng buộc chặt hơn (LR, kd, cons) có hết overfit không? |
| **S2** | S1 + SurvFace | Thêm 5.319 ID có giúp không? (run chính) |
| S3 | S2 + MEVID đầy đủ tracklet (2.3) | Thêm crop đúng miền có giúp không? |
| S4 (tuỳ chọn) | S2/S3 + `ldmk_noise=2` | Chỉ làm nếu sau 2.4 nhóm A3 (b) vẫn hơn (a) rõ rệt |
| S5 | Run tốt nhất, 3 seed | Độ ổn định |

Nếu muốn tiết kiệm thời gian: bỏ S1 và chạy thẳng S2. Đổi lại sẽ không tách được tác dụng của ràng buộc với tác dụng của SurvFace.

**Dừng sớm thủ công:** nếu MEVID val giảm 2 lần eval liên tiếp **và** điểm val dưới mức tốt nhất quá 2 điểm thì dừng run, vì checkpoint tốt nhất đã được lưu.

---

## 3. Đánh giá

### 3.1. Sau mỗi run, trên checkpoint tốt nhất
```bash
bash run.sh eval.face_eval --ckpt weights/m3_encode/face/kprpe_lr/<run>_s0.pt --val --a3 --out eval/results/face_eval_<run>.json
```
- GR-R1 / mAP / CC / SC và TAR@FAR trên CCVID test, CCVID test suy giảm, MEVID test, kèm phân tầng IPD.

### 3.2. Benchmark công khai (chống quên và kiểm tra tầm xa)
- **TinyFace:** `eval.tinyface --model kprpe_lr` (cần trỏ `config.yaml` vào checkpoint đang thử).
- **SurvFace test (cần viết):** `eval/survface.py` theo protocol nhận dạng tập mở chính thức: gallery, mated và unmated probe → TPIR@FPIR. Ảnh qua DFA giống lúc train.
- IJB-B/C: không có (cần form CVLface). Chống quên dựa vào CCVID sạch, TinyFace và SurvFace.

### 3.3. Độ tin cậy
- Bootstrap theo ID (5.000 lần) cho chênh lệch R0 so với mô hình mới trên MEVID test (54 ID).

---

## 4. Tích hợp (train_m3 giai đoạn D, sau khi chọn được checkpoint)
1. `kprpe_lr/config.yaml: checkpoint` trỏ tới run được chọn; thêm sha256 vào `manifest.yaml`.
2. Trích lại phần mặt của các store (`eval.regait --mod face`, cần thêm mode `face`), enroll lại gallery.
3. Train lại QE (inter_feat block 8/16 đã đổi), kiểm tra AUC.
4. Xuất lại ma trận điểm, chạy fusion v1 và QME (3 seed), báo cáo theo format General / SC / CC.

---

## 5. Rủi ro

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Vẫn overfit dù có SurvFace | MEVID val giảm sau 2–4k step | Tăng `lambda_kd` lên 10; giảm LR xuống 1e-6; giảm trọng số MEVID/CCVID |
| Học quá ít (kd quá chặt) | Điểm val gần như không đổi so với R0 | Giảm `lambda_kd` xuống 2 |
| DFA sai trên mặt 13–33 px của SurvFace | Điểm DFA thấp, landmark lệch khi xem bằng mắt | Căn chỉnh tâm cố định cho nguồn này (2.2b), hoặc S4 nhiễu landmark |
| Cache SurvFace lớn (khoảng 8 GB RAM khi nạp hết pool) | RAM 15 GB | Nạp ảnh từ h5 theo lô thay vì giữ toàn bộ trong RAM (sửa `Pool`) |
| Lỗi GPU nhất thời (CUBLAS) | Crash giữa run | Tự resume một lần (2.1) |
| Không có cải thiện trên MEVID test | Val tăng, test không | 20 ID val quá ít: báo cáo kèm bootstrap; chấp nhận giới hạn của dữ liệu |

---

## 6. Ước tính thời gian (GPU RTX 5050, chạy tuần tự)

| Việc | Thời gian |
|---|---|
| 2.1 sửa chain + 2.3 cờ `--all` | khoảng 30 phút code |
| 2.2 cache SurvFace | 30–45 phút |
| 2.3 cache MEVID đầy đủ | khoảng 20 phút |
| 2.4 quét ngưỡng DFA | khoảng 15 phút |
| S1, S2, S3 (16k step mỗi run) | khoảng 7 giờ mỗi run, tổng khoảng 21 giờ |
| Eval + SurvFace benchmark | khoảng 1 giờ |
| S5 (thêm 2 seed) | khoảng 14 giờ |
