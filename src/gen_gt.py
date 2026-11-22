import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from docx import Document

ROOT = Path(__file__).resolve().parent.parent

W, H = 1240, 1754

MARKER_RE = re.compile(r"^(\uf0b7|\u2022|[-\u2013\u25cf]|\d+[.)])$")


# PHAN 1: HAM TIEN ICH
def imread_vn(path, flags=cv2.IMREAD_COLOR):
    # cv2.imread hay lỗi khi đường dẫn có dấu tiếng Việt (Windows) -> đọc bằng numpy
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite_vn(path, img):
    # tương tự: ghi ảnh an toàn với đường dẫn có dấu
    _, buf = cv2.imencode(Path(path).suffix, img)
    buf.tofile(str(path))


def norm(s):
    # Chuẩn hóa chữ để SO KHỚP: đưa Unicode về dạng NFC và bỏ hết khoảng trắng.
    # Cần vì PDF và docx có thể cắt khoảng trắng / dấu tiếng Việt khác nhau.
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def parse_doc_num(raw_id):
    # Hỗ trợ các dạng đầu vào như "1", "01", "doc1", "doc01" -> trả về số nguyên n
    digits = "".join(ch for ch in str(raw_id) if ch.isdigit())
    if not digits:
        raise ValueError(f"Không tìm thấy số thứ tự trong tham số: {raw_id}")
    return int(digits)


def find_clean(raw_id):
    # Chỉ có thư mục data/clean_new -> tìm clean{n}.png hoặc clean{n:02d}.png tại đây
    n = parse_doc_num(raw_id)
    for name in (f"clean{n}.png", f"clean{n:02d}.png"):
        p = ROOT / "data" / "clean_new" / name
        if p.exists():
            return p
    return None


def resolve_doc_paths(raw_id):
    # Ánh xạ đúng tên file thực tế:
    # - PDF: 01.pdf -> 10.pdf (có fallback sang doc01.pdf / doc1.pdf nếu cần)
    # - DOCX: doc1.docx -> doc10.docx (có fallback sang doc01.docx / 01.docx nếu cần)
    n = parse_doc_num(raw_id)
    doc_id = f"doc{n:02d}"

    pdf_candidates = [
        ROOT / "data" / "pdf" / f"{n:02d}.pdf",
        ROOT / "data" / "pdf" / f"doc{n:02d}.pdf",
        ROOT / "data" / "pdf" / f"doc{n}.pdf",
    ]
    pdf_path = next((p for p in pdf_candidates if p.exists()), pdf_candidates[0])

    docx_candidates = [
        ROOT / "data" / "docx" / f"doc{n}.docx",
        ROOT / "data" / "docx" / f"doc{n:02d}.docx",
        ROOT / "data" / "docx" / f"{n:02d}.docx",
    ]
    docx_path = next((p for p in docx_candidates if p.exists()), docx_candidates[0])

    return doc_id, pdf_path, docx_path


# PHAN 2: DOC PDF -> DANH SACH DONG CHU (BO XU LY HINH)
def read_pdf_lines(pdf_path):
    """Trả về danh sách dòng chữ theo thứ tự đọc, đơn vị pixel."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]  # tài liệu chỉ 1 trang
        # Tọa độ PDF tính bằng "point". Nhân hệ số s để đổi sang pixel của ảnh sạch.
        s = W / page.rect.width
        lines = []
        pending_marker = None  # nhớ ký hiệu đầu dòng đang chờ gộp
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:  # chỉ lấy block chữ, bỏ qua hình
                continue
            for ln in b["lines"]:
                text = "".join(sp["text"] for sp in ln["spans"]).strip()
                if not text:
                    continue
                bbox = [v * s for v in ln["bbox"]]  # [x0, y0, x1, y1] theo pixel
                if MARKER_RE.match(text):  # dòng chỉ chứa "•" hay "1." ...
                    pending_marker = bbox  # chưa lưu, gộp vào dòng chữ kế tiếp
                    continue
                sp0 = ln["spans"][0]  # span đầu tiên quyết định kiểu chữ
                item = {
                    "bbox": bbox,
                    "text": unicodedata.normalize("NFC", text),
                    "size": round(sp0["size"], 1),  # cỡ chữ (pt)
                    "bold": bool(sp0["flags"] & 16),  # cờ bit 16 = in đậm
                    "italic": bool(sp0["flags"] & 2),  # cờ bit 2 = in nghiêng
                    "marker": pending_marker is not None,  # dòng này là mục danh sách?
                }
                if pending_marker is not None:
                    # mở rộng hộp dòng để bao luôn cả ký hiệu "•"
                    item["bbox"] = [
                        min(bbox[0], pending_marker[0]),
                        min(bbox[1], pending_marker[1]),
                        max(bbox[2], pending_marker[2]),
                        max(bbox[3], pending_marker[3]),
                    ]
                    pending_marker = None
                item["bbox"] = [round(v) for v in item["bbox"]]
                lines.append(item)
    return lines


# PHAN 3: GOM DONG THANH DOAN, DUA VAO VAN BAN TRONG DOCX
def group_paragraphs(lines, docx_path, doc_id):
    doc = Document(docx_path)
    paras = [p for p in doc.paragraphs if p.text.strip()]  # bỏ đoạn rỗng
    result, i, warns = [], 0, []
    for p in paras:
        target = norm(p.text)  # văn bản của đoạn trong docx (đã chuẩn hóa)
        acc, got = "", []
        # Lấy lần lượt từng dòng PDF cho đến khi chữ gom được dài bằng đoạn docx
        while i < len(lines) and len(acc) < len(target):
            acc += norm(lines[i]["text"])
            got.append(lines[i])
            i += 1
        # Danh sách bullet / đánh số: PDF có "•" hoặc "1." dính đầu chuỗi, docx thì không
        acc_stripped = re.sub(r"^([\uf0b7\u2022\-\u2013\u25cf]+|\d+[.)])", "", acc)
        numbered = False
        if acc != target and acc_stripped == target:
            acc, numbered = target, True
        # Nếu vẫn không khớp -> in cảnh báo để mình biết đáp án có thể sai
        if acc != target:
            warns.append(f"[{doc_id}] đoạn không khớp: docx='{p.text[:40]}' pdf='{acc[:40]}'")
        if not got:
            continue
        has_numpr = p._p.pPr is not None and p._p.pPr.numPr is not None  # docx đánh dấu danh sách
        is_list = has_numpr or got[0]["marker"] or numbered
        result.append({"lines": got, "is_list": is_list})
    if i < len(lines):
        warns.append(f"[{doc_id}] còn {len(lines) - i} dòng PDF chưa gán vào đoạn nào")
    return result, warns


def union_box(boxes):
    # Hộp nhỏ nhất bao trùm nhiều hộp (hộp của đoạn = bao các dòng của nó)
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def assign_roles(paras):
    # gán vai trò cho từng đoạn: title, subtitle, heading, list_item, paragraph
    sizes = [ln["size"] for p in paras for ln in p["lines"]]
    body = float(np.median(sizes)) if sizes else 12.0  # cỡ chữ thân bài = số trung vị
    for k, p in enumerate(paras):
        ls = p["lines"]
        big = max(ln["size"] for ln in ls)
        all_bold = all(ln["bold"] for ln in ls)
        if p["is_list"]:
            p["role"] = "list_item"
        elif k == 0 and (big > body or all_bold):
            p["role"] = "title"
        elif k == 1 and len(ls) == 1 and ls[0]["italic"]:
            p["role"] = "subtitle"
        elif all_bold and len(ls) == 1:
            p["role"] = "heading"
        else:
            p["role"] = "paragraph"
        p["bbox"] = union_box([ln["bbox"] for ln in ls])


# PHAN 4: TAO MASK DIEM ANH CHO CHU
def make_text_mask(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        m = pymupdf.Matrix(W / page.rect.width, H / page.rect.height)
        pm = page.get_pixmap(matrix=m, colorspace=pymupdf.csGRAY)
        g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
        text_mask = ((g < 160) * 255).astype(np.uint8)  # điểm tối = chữ -> 255
    return text_mask


# PHAN 5: VE OVERLAY CHU DE TU KIEM TRA
def draw_overlay(clean_path, paras, out_path):
    img = imread_vn(clean_path)
    for p in paras:
        for ln in p["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 160, 0), 1)  # DÒNG: xanh lá
        x0, y0, x1, y1 = p["bbox"]
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (255, 0, 0), 2)  # ĐOẠN: xanh dương
        cv2.putText(
            img,
            p["role"],
            (x0, max(12, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
        )  # ghi vai trò
    imwrite_vn(out_path, img)


# PHAN 6: HAM CHINH
def process(raw_id):
    doc_id, pdf_path, docx_path = resolve_doc_paths(raw_id)
    clean_path = find_clean(raw_id)
    for p in (pdf_path, docx_path):
        if not p.exists():
            raise FileNotFoundError(p)  # thiếu file đầu vào thì báo ngay

    # Bước 1: đọc dòng chữ từ PDF
    lines = read_pdf_lines(pdf_path)
    # Bước 2: gom dòng thành đoạn nhờ docx, rồi gán vai trò
    paras, warns = group_paragraphs(lines, docx_path, doc_id)
    assign_roles(paras)

    # Bước 3: đóng gói thành layout.json (đáp án về VỊ TRÍ CHỮ)
    layout = {
        "id": doc_id,
        "size": [W, H],
        "paragraphs": [
            {
                "role": p["role"],
                "bbox": p["bbox"],
                "lines": [
                    {k: ln[k] for k in ("bbox", "text", "size", "bold", "italic")}
                    for ln in p["lines"]
                ],
            }
            for p in paras
        ],
        "n_lines": sum(len(p["lines"]) for p in paras),  # đáp án "đếm dòng"
    }

    # Bước 4: đếm cấu trúc chữ (đáp án về SỐ LƯỢNG)
    roles = [p["role"] for p in paras]
    n_lists, prev = 0, False
    for r in roles:  # một danh sách = các list_item liên tiếp nhau
        if r == "list_item" and not prev:
            n_lists += 1
        prev = (r == "list_item")
    structure = {
        "id": doc_id,
        "n_paragraphs": roles.count("paragraph"),
        "n_titles": roles.count("title"),
        "n_subtitles": roles.count("subtitle"),
        "n_headings": roles.count("heading"),
        "n_lists": n_lists,
        "n_list_items": roles.count("list_item"),
        "n_lines": layout["n_lines"],
    }

    # Bước 5: ghi tất cả ra thư mục gt/
    for sub in ("layout", "structure", "mask_text", "overlay"):
        (ROOT / "gt" / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / "gt/layout" / f"{doc_id}.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (ROOT / "gt/structure" / f"{doc_id}.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tm = make_text_mask(pdf_path)
    imwrite_vn(ROOT / "gt/mask_text" / f"{doc_id}.png", tm)
    if clean_path is not None:
        draw_overlay(clean_path, paras, ROOT / "gt/overlay" / f"{doc_id}.png")
    else:
        warns.append(f"[{doc_id}] không thấy ảnh sạch trong data/clean_new, bỏ qua overlay")
    return structure, warns


# ĐIỂM VÀO: chạy file này sẽ xử lý các tài liệu và in cảnh báo (nếu có)
if __name__ == "__main__":
    ids = sys.argv[1:] or list(range(1, 11))
    all_warns = []
    for item in ids:
        st, w = process(item)
        all_warns += w
        print(st["id"], st)
    print("\nCẢNH BÁO:" if all_warns else "\nKhông có cảnh báo.")
    for w in all_warns:
        print(" ", w)