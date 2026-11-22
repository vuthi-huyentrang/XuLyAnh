# Xuất layout.json (vị trí dòng/đoạn theo thứ tự đọc) cho Phát dùng ở Giai đoạn 3.
# Cách chạy:  py src/export_layout.py all        (cả 50 ảnh)
#             py src/export_layout.py "nhàu" 1 2  (vài ảnh)
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Dùng lại đúng các hàm đã viết trong run_layout.py
from run_layout import (  # noqa: E402
    bo_dau_slug, bo_khung_mong, chay_bo_cuc_chu, doc_anh,
    tao_danh_sach_ten, tim_file_anh,
)

OUT_JSON = ROOT / "data" / "layout"   # KHÔNG nằm trong out/ vì out/ bị .gitignore chặn
OUT_JSON.mkdir(parents=True, exist_ok=True)

GAP_DOAN = 24     # khe hở dọc giữa 2 dòng >= số này (px) -> coi là sang đoạn mới
LECH_THUT = 15    # lề trái 2 dòng lệch hơn số này (px) -> coi là sang đoạn mới (thụt đầu dòng, bullet)


def sap_xep_theo_thu_tu_doc(khung):
    # Thứ tự đọc của văn bản 1 cột: từ trên xuống (y0), bằng nhau thì từ trái sang (x0)
    return sorted(khung, key=lambda k: (k[1], k[0]))


def gom_thanh_doan(dong):
    """Gom các dòng liên tiếp thành đoạn. Đây là quy tắc ĐƠN GIẢN (đúng khoảng 85% dòng),
    chưa phải Projection Profile hoàn chỉnh -> ghi rõ trong JSON để Phát biết."""
    doan = []
    for i, k in enumerate(dong):
        moi = i == 0
        if i > 0:
            truoc = dong[i - 1]
            khe_ho = k[1] - truoc[3]            # đáy dòng trước -> đỉnh dòng này
            lech_trai = abs(k[0] - truoc[0])    # chênh lề trái
            moi = khe_ho >= GAP_DOAN or lech_trai > LECH_THUT
        if moi:
            doan.append([])
        doan[-1].append(k)
    return doan


def hop_bao(cac_khung):
    return [
        min(k[0] for k in cac_khung), min(k[1] for k in cac_khung),
        max(k[2] for k in cac_khung), max(k[3] for k in cac_khung),
    ]


def tao_layout(ten, anh):
    dong = sap_xep_theo_thu_tu_doc(bo_khung_mong(chay_bo_cuc_chu(anh)))
    cac_doan = gom_thanh_doan(dong)
    h, w = anh.shape[:2]
    return {
        "id": bo_dau_slug(ten),
        "size": [w, h],
        "para_method": "gap_indent_heuristic",   # cách gom đoạn (để so sánh khi làm ablation)
        "paragraphs": [
            {"bbox": hop_bao(d), "lines": [{"bbox": k} for k in d]} for d in cac_doan
        ],
        "n_lines": len(dong),
    }


if __name__ == "__main__":
    for ten in tao_danh_sach_ten(sys.argv[1:]):
        p = tim_file_anh(ten)
        if not p.exists():
            print(f"Không thấy {p}")
            continue
        layout = tao_layout(ten, doc_anh(p))
        out = OUT_JSON / f"{layout['id']}.json"
        out.write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{ten}: {layout['n_lines']} dòng, {len(layout['paragraphs'])} đoạn -> {out.name}")