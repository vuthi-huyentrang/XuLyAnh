import numpy as np


def _as_binary_2d(binary):
    arr = np.asarray(binary)
    if arr.ndim != 2:
        raise ValueError(f"binary phải là ảnh 2 chiều, nhận shape={arr.shape}")
    return (arr > 0).astype(np.uint8) * 255


def rlsa_horizontal(binary, threshold):
    """RLSA theo từng hàng (ngang). Khoảng trống dài <= threshold giữa hai điểm trắng sẽ được lấp."""
    out = _as_binary_2d(binary)
    if threshold <= 0:
        return out

    for y in range(out.shape[0]):
        row = out[y] > 0
        # d = +1 ở đầu mỗi đoạn trắng, -1 ngay sau cuối đoạn trắng
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)          # đầu các đoạn trắng
        ends = np.flatnonzero(d == -1)           # sau cuối các đoạn trắng (loại trừ)
        if len(starts) < 2:
            continue                             # có ít hơn 2 đoạn thì không có khoảng trống ở giữa
        gaps = starts[1:] - ends[:-1]            # độ dài khoảng trống giữa hai đoạn liên tiếp
        for i in np.flatnonzero(gaps <= threshold):
            out[y, ends[i] : starts[i + 1]] = 255  # lấp khoảng trống bằng điểm trắng
    return out


def rlsa_vertical(binary, threshold):
    """RLSA theo từng cột (dọc) = làm RLSA ngang trên ảnh đã chuyển vị rồi chuyển lại."""
    arr = _as_binary_2d(binary)
    return rlsa_horizontal(arr.T, threshold).T


def rlsa(binary, t_h, t_v, t_smooth=None, mode="and"):
    """RLSA cho tài liệu văn bản:
    - mode='and'   : cổ điển (ngang với t_h) AND (dọc với t_v)
    - mode='or'    : (ngang với t_h) OR (dọc với t_v)
    - mode='hv'    : nối ngang trước (t_h), rồi nối dọc (t_v)
    - mode='h_only': chỉ nối ngang (t_h)
    """
    if mode == "h_only":
        combined = rlsa_horizontal(binary, t_h)
    else:
        h = rlsa_horizontal(binary, t_h)
        v = rlsa_vertical(binary, t_v)
        if mode == "and":
            combined = np.minimum(h, v)
        elif mode == "or":
            combined = np.maximum(h, v)
        elif mode == "hv":
            combined = rlsa_vertical(h, t_v)
        else:
            raise ValueError(f"mode không hỗ trợ: {mode}")

    if t_smooth and t_smooth > 0:
        combined = rlsa_horizontal(combined, t_smooth)
    return combined