import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

# Thư mục gốc của project (XuLyAnh), tính từ vị trí file này
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ccl import areas, bounding_boxes, connected_components  # tách và đếm khối chữ
from morphology import opening, rect_kernel                  # dọn nhiễu
from rlsa import rlsa                                        # nối chữ thành dòng/khối

BIN_DIR = ROOT / "data" / "stage1" / "bin"  # thư mục chứa ảnh nhị phân đầu vào
OUT_DIR = ROOT / "out" / "layout_demo"      # nơi lưu ảnh kết quả
OUT_DIR.mkdir(parents=True, exist_ok=True)

NHOM_ANH = [
    "ảnh nghiêng",
    "nhàu",
    "thiếu sáng",
    "lộn xộn",
    "khuất",
]


def bo_dau_slug(text):
    """Đổi tên có dấu/khoảng trắng thành dạng slug không dấu để đặt tên file output gọn."""
    s = unicodedata.normalize("NFD", text)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def doc_anh(path):
    # Đọc bằng numpy để tránh lỗi khi đường dẫn có dấu cách hoặc tiếng Việt
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def la_vet_bong(k, rong_anh, ti_le_cao_rong=1.5, le_ngoai=0.07):
    """True nếu khung giống vệt bóng/nhàu ở mép giấy chứ không phải dòng chữ.
    - Dòng chữ thì NGANG (rộng > cao); vệt bóng ở mép thì DỌC (cao > 1.5 x rộng).
    - Chữ nằm trong lề giấy; khung nằm sát mép (7% ngoài cùng mỗi bên) là bóng."""
    x0, y0, x1, y1 = k
    rong, cao = x1 - x0 + 1, y1 - y0 + 1
    if cao > ti_le_cao_rong * rong:
        return True
    return x0 > (1 - le_ngoai) * rong_anh or x1 < le_ngoai * rong_anh


def chay_bo_cuc_chu(anh, t_h=30, t_v=10, t_smooth=20, min_area=150):
    fg = ((anh < 128) * 255).astype(np.uint8)      # đảo màu: chữ đen -> trắng
    fg[:60, :] = 0                                 # xóa mép trên 60px (vệt bóng đen của giấy)
    fg[-60:, :] = 0                                # xóa mép dưới 60px
    fg[:, :12] = 0
    fg[:, -12:] = 0

    sach = opening(fg, rect_kernel(2, 2))          # mở: xóa đốm nhiễu nhỏ
    noi = rlsa(sach, t_h=t_h, t_v=t_v, t_smooth=t_smooth)  # RLSA: nối chữ thành dòng/khối
    nhan, n = connected_components(noi)            # CCL: gán nhãn từng khối chữ
    khung, dien_tich = bounding_boxes(nhan, n), areas(nhan, n)
    rong_anh = anh.shape[1]
    return [
        list(map(int, k)) for k, a in zip(khung, dien_tich)
        if a >= min_area and not la_vet_bong(k, rong_anh)   # bỏ vệt bóng đen ở mép
    ]


def bo_khung_mong(khung, min_cao=12, min_rong=12):
    # Loại bỏ các khung quá mỏng (đường kẻ ngang/dọc, nét mảnh)
    return [
        list(map(int, k))
        for k in khung
        if (k[3] - k[1] + 1) >= min_cao and (k[2] - k[0] + 1) >= min_rong
    ]


def ve_khung_chu(anh, khung_chu):
    # Vẽ khung chữ màu đỏ lên ảnh xám chuyển sang BGR
    mau = cv2.cvtColor(anh, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in khung_chu:
        cv2.rectangle(mau, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return mau


def luu_anh(path, img):
    _, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def tim_file_anh(ten_khong_duoi):
    """Tìm file ảnh theo tên (hỗ trợ .png, .jpg, .jpeg)."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = BIN_DIR / f"{ten_khong_duoi}{ext}"
        if p.exists():
            return p
    return BIN_DIR / f"{ten_khong_duoi}.png"


def phan_tich_ten(ten):
    """Tách tên ảnh thành (nhóm, số doc, số thứ tự ảnh phụ).
    'nhàu 5'   -> ('nhàu', 5, None)   ảnh gốc của doc5
    'nhàu 5-1' -> ('nhàu', 5, 1)      ảnh chụp thêm lần 1 của doc5
    'khuất 1-2'-> ('khuất', 1, 2)     ảnh chụp thêm lần 2 của doc1
    Trả về None nếu tên không đúng dạng '<nhóm> <số doc>[-<số phụ>]'."""
    m = re.match(r"^(.+?)\s+(\d+)(?:-(\d+))?$", ten.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def doc_tuong_ung(ten):
    """Tên ảnh -> mã doc, vd 'nhàu 5-2' -> 'doc05'. Dùng để lấy đúng PDF và gt."""
    kq = phan_tich_ten(ten)
    if kq is None:
        raise ValueError(f"Tên ảnh không đúng dạng '<nhóm> <số doc>[-<số phụ>]': {ten}")
    return f"doc{kq[1]:02d}"


def duong_dan_pdf(ten):
    return ROOT / "data" / "pdf" / f"{doc_tuong_ung(ten)[3:]}.pdf"     # doc05 -> data/pdf/05.pdf


def duong_dan_gt(ten):
    return ROOT / "gt" / "layout" / f"{doc_tuong_ung(ten)}.json"


def _khoa_sap_xep(ten):
    nhom, so_doc, phu = phan_tich_ten(ten)
    thu_tu_nhom = NHOM_ANH.index(nhom) if nhom in NHOM_ANH else 99
    return thu_tu_nhom, so_doc, phu or 0


def liet_ke_ten(nhom=None):
    """Quét thư mục BIN_DIR, lấy MỌI ảnh đúng dạng tên (kể cả 5-1, 5-2...) -> thêm ảnh mới không cần sửa code."""
    ten_list = set()
    for p in BIN_DIR.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and phan_tich_ten(p.stem):
            if nhom is None or phan_tich_ten(p.stem)[0] == nhom:
                ten_list.add(p.stem)
    return sorted(ten_list, key=_khoa_sap_xep)


def tao_danh_sach_ten(args):
    """Sinh danh sách tên file (không gồm phần mở rộng) từ tham số dòng lệnh."""
    if not args:
        return [f"{nhom} {idx}" for nhom in NHOM_ANH for idx in (1, 2)]

    if args == ["all"]:
        return liet_ke_ten()

    # Nếu tham số đầu là một tên nhóm (vd: "nhàu" 1 2 5-1 hoặc "ảnh nghiêng" all)
    if args[0] in NHOM_ANH:
        nhom = args[0]
        so_list = args[1:]
        if not so_list or so_list == ["all"]:
            return liet_ke_ten(nhom)
        return [f"{nhom} {x}" for x in so_list]      # x có thể là "3" hoặc "5-1"

    # Ngược lại coi mỗi tham số là tên ảnh đầy đủ (bỏ .png/.jpg nếu người dùng gõ kèm)
    ket_qua = []
    for item in args:
        p = Path(item)
        ten = p.stem if p.suffix.lower() in {".png", ".jpg", ".jpeg"} else item
        ket_qua.append(ten)
    return ket_qua


if __name__ == "__main__":
    # Ví dụ cách chạy:
    # py src/run_layout.py all                  -> chạy cả 5 nhóm x 10 ảnh (50 ảnh)
    # py src/run_layout.py "ảnh nghiêng" 1 2 3  -> chạy ảnh nghiêng 1, 2, 3
    # py src/run_layout.py "nhàu" all           -> chạy nhàu 1 -> nhàu 10
    # py src/run_layout.py "thiếu sáng 4" "khuất 9"
    danh_sach_ten = tao_danh_sach_ten(sys.argv[1:])

    for ten in danh_sach_ten:
        p = tim_file_anh(ten)
        if not p.exists():
            print(f"Không thấy {p}")
            continue

        anh = doc_anh(p)
        cac_khung = chay_bo_cuc_chu(anh)
        khung_chu = bo_khung_mong(cac_khung, min_cao=12, min_rong=12)

        ten_out = f"{bo_dau_slug(ten)}_khung.png"
        luu_anh(OUT_DIR / ten_out, ve_khung_chu(anh, khung_chu))
        print(f"{ten}: {len(khung_chu)} khung chu -> {ten_out}")