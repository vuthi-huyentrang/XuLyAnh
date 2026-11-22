import cv2
import numpy as np


def _validate_inputs(img, kernel):
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Ảnh phải là 2 chiều, nhận shape={arr.shape}")
    k = np.asarray(kernel)
    if k.ndim != 2 or k.size == 0:
        raise ValueError(f"Kernel phải là mảng 2 chiều không rỗng, nhận shape={k.shape}")
    if not np.any(k > 0):
        raise ValueError("Kernel phải có ít nhất một phần tử lớn hơn 0")
    return arr.astype(np.uint8, copy=False), k > 0


def _shift_stack_reduce(img, kernel, reducer, border_mode="replicate"):
    arr, mask = _validate_inputs(img, kernel)
    kh, kw = mask.shape
    cy, cx = kh // 2, kw // 2
    h, w = arr.shape
    if h == 0 or w == 0:
        return arr.copy()

    pad_width = ((cy, kh - 1 - cy), (cx, kw - 1 - cx))
    if border_mode == "replicate":
        padded = np.pad(arr, pad_width, mode="edge")
    elif border_mode == "constant_zero":
        padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    else:
        raise ValueError(f"border_mode không hỗ trợ: {border_mode}")

    out = None
    for dy in range(kh):
        for dx in range(kw):
            if not mask[dy, dx]:
                continue
            window = padded[dy : dy + h, dx : dx + w]
            out = window.copy() if out is None else reducer(out, window)
    return out.astype(np.uint8, copy=False)


def erode(img, kernel, border_mode="replicate"):
    """CO: lấy min trong cửa sổ kernel (áp dụng cho cả ảnh nhị phân và ảnh xám)."""
    return _shift_stack_reduce(img, kernel, np.minimum, border_mode=border_mode)


def dilate(img, kernel, border_mode="replicate"):
    """GIÃN: lấy max trong cửa sổ kernel (áp dụng cho cả ảnh nhị phân và ảnh xám)."""
    return _shift_stack_reduce(img, kernel, np.maximum, border_mode=border_mode)


def opening(img, kernel, border_mode="replicate"):
    """MỞ = Co trước, Giãn sau: xóa nhiễu sáng/đốm nhỏ hơn kernel."""
    return dilate(erode(img, kernel, border_mode=border_mode), kernel, border_mode=border_mode)


def closing(img, kernel, border_mode="replicate"):
    """ĐÓNG = Giãn trước, Co sau: lấp khe hở/đứt gãy nhỏ hơn kernel."""
    return erode(dilate(img, kernel, border_mode=border_mode), kernel, border_mode=border_mode)


def rect_kernel(h, w):
    """Kernel hình chữ nhật h x w toàn số 1."""
    if h <= 0 or w <= 0:
        raise ValueError("Kích thước kernel h và w phải lớn hơn 0")
    return np.ones((h, w), dtype=np.uint8)


def compare_with_cv2(img, kernel, border_mode="replicate"):
    """Đối chiếu 4 phép toán với cv2 -> số điểm ảnh khác nhau phải bằng 0."""
    src = np.asarray(img, dtype=np.uint8)
    k = np.asarray(kernel, dtype=np.uint8)
    cv_border = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_CONSTANT

    my_erode = erode(src, k, border_mode=border_mode)
    cv_erode = cv2.erode(src, k, borderType=cv_border)

    my_dilate = dilate(src, k, border_mode=border_mode)
    cv_dilate = cv2.dilate(src, k, borderType=cv_border)

    my_open = opening(src, k, border_mode=border_mode)
    cv_open = cv2.morphologyEx(src, cv2.MORPH_OPEN, k, borderType=cv_border)

    my_close = closing(src, k, border_mode=border_mode)
    cv_close = cv2.morphologyEx(src, cv2.MORPH_CLOSE, k, borderType=cv_border)

    return {
        "erode_diff_px": int(np.count_nonzero(my_erode != cv_erode)),
        "dilate_diff_px": int(np.count_nonzero(my_dilate != cv_dilate)),
        "open_diff_px": int(np.count_nonzero(my_open != cv_open)),
        "close_diff_px": int(np.count_nonzero(my_close != cv_close)),
    }