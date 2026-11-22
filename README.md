# XuLyAnh – Phần tách dòng (Trang)

Tìm vị trí từng **dòng chữ** trên ảnh tài liệu, rồi lưu ra file JSON để Phát dùng tiếp (OCR).

## Quá trình chạy (từng file làm gì)

```
ảnh từ Việt ──► run_layout.py ──► export_layout.py ──► JSON cho Phát
(stage1/bin)        │
                    ├─ morphology.py  (dọn nhiễu)
                    ├─ rlsa.py        (nối chữ thành dòng)
                    └─ ccl.py         (đếm và khoanh từng khối)

PDF từ Word ──► render_clean.py ──► gen_gt.py ──► đáp án (gt/)
```

| File | Nhận vào (input) | Cho ra (output) |
|---|---|---|
| `morphology.py` | ảnh nhị phân + kernel | ảnh sau erode / dilate / opening / closing. Dùng để xóa đốm nhiễu nhỏ. Tự viết, đã so với cv2 (lệch 0 điểm ảnh). |
| `rlsa.py` | ảnh nhị phân (chữ trắng) | ảnh đã **nối** các chữ cùng hàng thành một vệt dài (một dòng). |
| `ccl.py` | ảnh đã nối | nhãn từng khối, số khối, khung bao `[x0,y0,x1,y1]` và diện tích mỗi khối. Tự viết, đã so với cv2. |
| `run_layout.py` | ảnh trong `data/stage1/bin/` | ảnh có khung đỏ trong `out/layout_demo/` (để nhìn bằng mắt). Chứa hàm chính chạy 3 bước trên, rồi bỏ khung mỏng và vệt bóng. |
| `export_layout.py` | ảnh trong `data/stage1/bin/` | file JSON trong `data/layout/` (mỗi ảnh một file). Dùng lại hàm của `run_layout.py`, thêm xếp thứ tự đọc và gom đoạn. |
| `render_clean.py` | `data/pdf/*.pdf` | ảnh sạch `data/clean_new/clean1..10.png` (1240×1754). |
| `gen_gt.py` | PDF + docx | đáp án trong `gt/` (`layout` = vị trí dòng/đoạn, `structure` = loại đoạn, `mask_text` = mặt nạ chữ, `overlay` = ảnh vẽ khung để kiểm tra). |

## Chạy như thế nào

```
py src/run_layout.py all      # vẽ khung đỏ lên ảnh để xem bằng mắt  -> out/layout_demo/
py src/export_layout.py all   # xuất tọa độ ra JSON                   -> data/layout/
```

Đổi PDF hoặc docx thì chạy lại: `py src/render_clean.py` rồi `py src/gen_gt.py`.

## Đặt tên ảnh

Ảnh để trong `data/stage1/bin/`, tên dạng `<nhóm> <số doc>` hoặc `<nhóm> <số doc>-<số phụ>`:

- `nhàu 5`   → ảnh của doc5
- `nhàu 5-1` → ảnh chụp thêm của doc5

Số đứng trước dấu `-` là số doc. Thêm ảnh mới chỉ cần đặt đúng tên, không phải sửa code.

## Kết quả (cho Phát)

Mỗi ảnh có một file `data/layout/<tên ảnh>.json`, trong đó:

- `doc`: ảnh thuộc doc nào (vd `doc05`)
- `paragraphs` → `lines`: các đoạn và dòng, **đã xếp theo thứ tự đọc**
- mỗi `bbox = [x0, y0, x1, y1]`: góc trái trên và góc phải dưới (pixel, ảnh 1240×1754)

File chỉ có tọa độ, không có chữ. Chữ lấy bằng OCR, đáp án chữ nằm ở `gt/`.
