# Plan phụ: Thay AIM bằng CSCI ở nhánh body (M3c)

> Bối cảnh: nhánh body hiện là điểm nghẽn. AIM (ResNet-50, checkpoint LTCC/PRCC, zero-shot) chỉ đạt GR-R1 80.1 trên CCVID và 20.3 trên MEVID. CSCI (ICCV 2025, backbone EVA02-L) có checkpoint công bố và đã được một nhóm độc lập tái hiện: CCVID general 91.73 / 91.58, CC 91.01 / 90.86; MEVID (checkpoint MEVID) general 79.75 / 56.66; MEVID khi chạy chéo miền từ checkpoint CCVID general 65.51 / 36.82.
>
> Mục tiêu: thay AIM bằng CSCI **không cần train encoder**, giữ nguyên interface của M3, rồi train lại QME trên điểm số mới.

---

## 0. Mục tiêu và tiêu chí thành công

| Hạng mục | Hiện tại (AIM) | Mục tiêu (CSCI) |
|---|---|---|
| Body đơn lẻ, CCVID test, GR-R1 | 80.1 | ≥ 91 (theo số công bố) |
| Body đơn lẻ, CCVID CC-mAP | 71.9 | ≥ 90 |
| Body đơn lẻ, MEVID test, GR-R1 | 20.3 | ≥ 65 (checkpoint CCVID) / ≥ 79 (checkpoint MEVID) |
| Fusion, CCVID, thiếu mặt (GR-R1) | 81.3–82.3 | ≥ body đơn lẻ CSCI |
| Fusion, MEVID, đủ modality (GR-R1) | 32.9–36.1 | ≥ body đơn lẻ CSCI |
| Mọi ô fusion | còn ✗ | fusion ≥ modality đơn tốt nhất |

Các mục tiêu fusion là mốc đề xuất. Mục tiêu body đơn lẻ lấy theo số đã được tái hiện.

**Nguyên tắc:**
1. Xác nhận checkpoint chạy đúng trên một tập con MEVID trước, tích hợp sau.
2. Không train CSCI ở giai đoạn này.
3. AIM giữ lại làm phương án quay lui bằng config, không xoá code.

---

## Giai đoạn A – Chuẩn bị và tái hiện

### A1. Lấy mã và checkpoint
- Repo: `github.com/ppriyank/ICCV-CSCI-Person-ReID` → thêm vào `third_party/CSCI/` (git submodule, khoá commit).
- Tải các checkpoint theo README (Hugging Face):

| Checkpoint | Loại | Dùng cho |
|---|---|---|
| CSCI-V CCVID | Video | Đánh giá CCVID, mặc định cho dữ liệu kiểu CCVID |
| CSCI-V MEVID | Video | Đánh giá MEVID, dữ liệu kiểu CCTV |
| CCVID_IMG | Ảnh | Gallery chỉ có ảnh tĩnh (enroll bằng ảnh) |
| MEVID_IMG | Ảnh | Như trên. Lưu ý file trên Hugging Face có thể khác `MEVID_IMG2` trong script test của tác giả |

- Ghi `manifest.yaml` cho từng checkpoint: tên file, URL, sha256, dữ liệu train, license.

### A2. Tái hiện một phần trên MEVID
- Chỉ chạy trên **một tập con của MEVID test** bằng checkpoint **CSCI-V MEVID**, đủ để xác nhận checkpoint và cách chạy là đúng. Không cần tái hiện toàn bộ CCVID và MEVID.
- Gợi ý tập con: khoảng 100 query (lấy ngẫu nhiên theo ID, cố định seed) cùng toàn bộ gallery của các ID đó; ghi danh sách vào `splits/mevid_csci_subset.json`.
- Chạy **script test gốc của tác giả** trên đúng tập con này, lưu feature và kết quả làm mốc.
- Kiểm tra hợp lý: general Rank-1 / mAP trên tập con phải cùng mức với số đã công bố (79.75 / 56.66 trên toàn bộ test). Tập con nhỏ nên chỉ so mức độ, không yêu cầu khớp từng chữ số.
- Nếu lệch xa, dừng lại và kiểm tra tiền xử lý, cách cắt clip, evaluator trước khi đi tiếp.

### A3. Đo tài nguyên trên máy thật
- Bộ nhớ GPU đỉnh, thời gian trích feature cho mỗi tracklet, số frame/giây.
- Mốc tham chiếu (GPU laptop RTX 5070 bị giới hạn công suất): bản video ~3.8 GB, ~39 frame/giây; bản ảnh ~2.6 GB, ~28 ảnh/giây.
- Đo lại trên GPU sẽ dùng thật trước khi lên kế hoạch tài nguyên.

---

## Giai đoạn B – Viết wrapper

### B1. Cấu trúc thư mục
```
farsight/modules/m3_encode/body/
├── aim/                     # giữ nguyên, phương án quay lui
└── csci/
    ├── __init__.py          # đăng ký "csci_video", "csci_image"
    ├── model.py             # wrapper kế thừa BaseEncoder
    ├── preprocess.py        # resize/normalize đúng như repo gốc
    ├── clips.py             # cắt tracklet thành clip
    ├── config.yaml          # checkpoint, kích thước, số clip, lớp hook
    ├── manifest.yaml
    ├── README.md
    └── tests/test_smoke.py
```

### B2. Đầu vào và tiền xử lý
- Crop thân người từ tracklet của M1, resize về **224×224**, chuẩn hoá đúng như repo gốc.
- **Bản video:** cắt tracklet thành các clip 8 frame, đưa **cách một frame** vào model 4 frame (giữ đúng cách làm của tác giả).
- **Bản ảnh:** chạy từng ảnh, hoặc lấy frame giữa của mỗi clip rồi gộp.

### B3. Gộp theo track
- Mỗi clip cho một vector 1024-d (token ReID). Lấy trung bình các clip rồi L2-normalize → `feat_body (1024,)`.
- **Ngân sách clip:** dùng tối đa **8 clip** mỗi track. Theo đánh giá trên MEVID validation của báo cáo tham chiếu, 8 clip chỉ mất ~0.35 điểm CC mAP so với dùng toàn bộ clip, nhưng nhanh hơn nhiều. Xác nhận lại trên validation của mình.
- Track có ít hơn 4 frame: dùng bản ảnh trên các frame có được.

### B4. Đầu ra theo interface M3
```python
ModalityOut = {
  "feat": float32[1024],      # thay cho 4096-d của AIM
  "quality": float,           # độ phân giải crop × tỉ lệ frame không bị che
  "inter_feat": float32[D_mid] | None,
  "n_frames": int,
}
```
- Template body giảm từ 16 KB (AIM 4096-d) xuống 4 KB (1024-d, float32).

### B5. Hook feature trung gian (cho QE nhánh body, tuỳ chọn)
- Đăng ký forward hook ở một block giữa của EVA02-L (gợi ý block ~16/24), lấy trung bình token không gian.
- Chỉ cần nếu sau này bật QE cho nhánh body.

### B6. Kiểm tra wrapper
- Chạy wrapper trên **cùng tập con MEVID ở A2** → feature phải trùng với feature của script gốc (cosine ≥ 0.999 theo từng tracklet) và Rank-1 / mAP khớp số của A2.
- `test_smoke.py`: nạp checkpoint, chạy một batch giả, kiểm tra shape 1024, kết quả tất định.

---

## Giai đoạn C – Kiểm tra tính bổ sung trước khi fusion

CSCI nhận crop toàn thân nên có thể đã ngầm chứa thông tin mặt khi người ở gần. Cần biết KP-RPE và CSCI bổ sung cho nhau đến đâu trước khi kỳ vọng fusion có lợi.

**Đặt ngưỡng trước khi chạy:**

| Kiểm tra | Cách đo | Ngưỡng đề xuất |
|---|---|---|
| Mặt sửa lỗi của body | % query CSCI sai top-1 mà KP-RPE đúng | ≥ 10% |
| Body sửa lỗi của mặt | % query KP-RPE sai top-1 mà CSCI đúng | ≥ 10% |
| Tương quan điểm | Tương quan điểm cặp cùng người giữa hai model | Càng thấp càng tốt (ghi lại, không đặt ngưỡng) |

- Đo trên CCVID sạch, CCVID suy giảm và MEVID.
- Kỳ vọng: bổ sung **ít** trên CCVID sạch (cả hai đều mạnh), bổ sung **nhiều** trên CCVID suy giảm và MEVID (mặt kém).
- Nếu cả hai ngưỡng đều không đạt ở mọi tập → fusion mặt + body khó có lợi; báo cáo rõ và cân nhắc chỉ dùng model mạnh hơn.

---

## Giai đoạn D – Tích hợp vào fusion

### D1. Chọn checkpoint cho cấu hình chung
Hệ thống cần **một** cấu hình cho mọi dữ liệu. Ba phương án, chọn trên **validation**:

| Phương án | Ưu | Nhược |
|---|---|---|
| D1a. Chỉ CSCI-V CCVID | Đơn giản | Yếu hơn trên dữ liệu kiểu CCTV (65.5 so với 79.75 trên MEVID) |
| D1b. Chỉ CSCI-V MEVID | Mạnh trên CCTV | Cần đo lại trên CCVID; có thể yếu hơn bản CCVID |
| D1c. Chạy cả hai, đưa thành **hai cột điểm body** vào QME | QME tự chọn theo chất lượng | Tốn gấp đôi tính toán nhánh body |

Đề xuất: thử D1a và D1b trước. Chỉ làm D1c nếu chênh lệch giữa hai miền lớn và có dư tài nguyên.

### D2. Xuất lại template và ma trận điểm
- Gallery CCVID và MEVID: enroll lại nhánh body bằng CSCI.
- Xuất ma trận điểm N_G × 3 (mặt, dáng đi, body) cho CCVID sạch, CCVID suy giảm, MEVID.

### D3. Fusion v1
- Chạy lại z-score + trọng số chất lượng với body mới. Ước lượng lại thống kê z-score trên validation (phân phối điểm của CSCI khác AIM).

### D4. Train lại QME
- Dữ liệu: CCVID train + MEVID train (84 ID), giữ tỉ lệ mẫu thiếu modality và suy giảm như trước.
- QE mặt giữ nguyên (đã đạt AUC 0.98). Thêm `gait_quality` nếu đã có từ plan phụ BigGait.
- **Rủi ro cần theo dõi:** checkpoint CSCI-V CCVID được train trên CCVID train, nên trên đúng tập train đó điểm body "quá tốt". QME có thể học cách tin body quá mức. Kiểm tra trên validation và trên CCVID suy giảm; nếu thấy dấu hiệu này, train QME chủ yếu trên MEVID train hoặc trên dữ liệu suy giảm.
- Chạy 3 seed, chọn checkpoint trên validation (CCVID val cắt từ ID train, MEVID val 20 ID).

---

## Giai đoạn E – Đánh giá

### E1. Bảng kết quả
Giữ format hiện tại (CCVID sạch, CCVID suy giảm, MEVID × đủ modality / thiếu mặt / track ngắn × 5 chỉ số), thêm dòng **body đơn lẻ CSCI** và so với dòng AIM cũ.

### E2. So sánh với paper
- Theo protocol QME (Rank-1 / mAP general, TAR@1%FAR, FNIR@1%FPIR trung vị 10 lần chọn non-mated) để so với QME, IDSelect.
- Báo cáo thêm CC Rank-1 / CC mAP để so trực tiếp với CSCI paper và báo cáo tham chiếu.

### E3. Độ tin cậy
- Bootstrap theo ID (5.000 lần) cho chênh lệch giữa cấu hình mới và cũ.
- Ghi số query hợp lệ và số query bị bỏ (query không có positive hợp lệ).

---

## Rủi ro

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không tái hiện được trên tập con MEVID | Kết quả A2 lệch xa số công bố, hoặc wrapper lệch script gốc | Kiểm tra lại tiền xử lý, cách cắt clip, evaluator; so feature với script gốc |
| Chi phí tính toán | Không đạt tốc độ mục tiêu end-to-end | Giảm số clip (8 → 4, chấp nhận mất ~1.8 điểm CC mAP); dùng bản ảnh trên frame giữa clip |
| Bộ nhớ GPU | Chạy cùng KP-RPE và BigGait-L bị tràn | Chạy tuần tự các encoder, hoặc dùng BigGait-S |
| QME tin body quá mức | Validation tốt, CCVID suy giảm kém | Train QME trên MEVID train và dữ liệu suy giảm |
| Mặt và body trùng thông tin | Ngưỡng ở giai đoạn C không đạt trên CCVID sạch | Chấp nhận; giá trị của fusion nằm ở dữ liệu suy giảm và tầm xa |
| Checkpoint ảnh MEVID khác bản của tác giả | Số MEVID_IMG lệch | Ưu tiên bản video; ghi rõ trong manifest |
| License | Dùng ngoài nghiên cứu | Chỉ dùng cho nghiên cứu; ghi vào manifest |

---

## Quay lui

Đổi một dòng trong `configs/pipeline/*.yaml`:
```yaml
m3_encode: {face: [dfa_aligner, kprpe], gait: biggait, body: aim}        # cũ
m3_encode: {face: [dfa_aligner, kprpe], gait: biggait, body: csci_video} # mới
```