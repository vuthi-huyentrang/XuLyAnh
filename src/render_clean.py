# Render ảnh clean TỪ CHÍNH file PDF -> ảnh và đáp án (gt) luôn khớp tuyệt đối.
import sys
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
W, H = 1240, 1754                      # kích thước chuẩn của cả nhóm
PDF_DIR = ROOT / "data" / "pdf"
OUT = ROOT / "data" / "clean_new"      # ghi ra thư mục MỚI, chưa đè ảnh cũ
OUT.mkdir(parents=True, exist_ok=True)

# Nếu truyền tham số dòng lệnh thì hỗ trợ cả "1", "01", hoặc "doc1"
raw_args = sys.argv[1:]
if raw_args:
    nums = [int("".join(ch for ch in item if ch.isdigit())) for item in raw_args]
else:
    nums = list(range(1, 11))

for n in nums:
    pdf_path = PDF_DIR / f"{n:02d}.pdf"  # PDF tên 01.pdf -> 10.pdf
    if not pdf_path.exists():
        print(f"Bỏ qua: không tìm thấy {pdf_path}")
        continue

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    m = pymupdf.Matrix(W / page.rect.width, H / page.rect.height)   # cùng hệ số với gen_gt
    pix = page.get_pixmap(matrix=m, colorspace=pymupdf.csRGB)       # ảnh màu, nền trắng
    pix.save(str(OUT / f"clean{n}.png"))
    doc.close()
    print(f"{pdf_path.name} -> clean{n}.png ({pix.width} x {pix.height})")