import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - fallback only
    _distance_transform_edt = None


def _edt(mask: np.ndarray, spacing=None) -> np.ndarray:
    if _distance_transform_edt is not None:
        return _distance_transform_edt(mask, sampling=spacing)

    # Fallback to OpenCV if SciPy is unavailable.
    import cv2

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    if spacing is not None:
        dist = dist * float(np.mean(spacing))
    return dist


def signed_distance_map(one_hot: np.ndarray,
                        spacing=None,
                        class_ids=None,
                        channel_first: bool = False,
                        compact: bool = False,
                        dtype=np.float32) -> np.ndarray:
    if one_hot.ndim != 3:
        raise ValueError(f"Expected 3D one_hot array, got shape {one_hot.shape}")

    if channel_first:
        num_classes, height, width = one_hot.shape
    else:
        height, width, num_classes = one_hot.shape

    if class_ids is None:
        class_ids = list(range(num_classes))
    else:
        class_ids = list(class_ids)

    if compact:
        out_shape = (len(class_ids), height, width) if channel_first else (height, width, len(class_ids))
    else:
        out_shape = (num_classes, height, width) if channel_first else (height, width, num_classes)
    out = np.zeros(out_shape, dtype=dtype)

    for out_idx, class_idx in enumerate(class_ids):
        if channel_first:
            posmask = one_hot[class_idx] > 0.5
        else:
            posmask = one_hot[..., class_idx] > 0.5

        if not posmask.any():
            continue

        negmask = ~posmask
        dist_pos = _edt(posmask, spacing=spacing)
        dist_neg = _edt(negmask, spacing=spacing)
        signed = dist_neg * negmask - (dist_pos - 1) * posmask

        if compact:
            if channel_first:
                out[out_idx] = signed
            else:
                out[..., out_idx] = signed
        else:
            if channel_first:
                out[class_idx] = signed
            else:
                out[..., class_idx] = signed

    return out


def batch_signed_distance_map(one_hot: np.ndarray,
                              spacing=None,
                              class_ids=None,
                              channel_first: bool = False,
                              compact: bool = False,
                              dtype=np.float32) -> np.ndarray:
    if one_hot.ndim != 4:
        raise ValueError(f"Expected 4D one_hot array, got shape {one_hot.shape}")

    batch_size = one_hot.shape[0]
    if channel_first:
        _, num_classes, height, width = one_hot.shape
    else:
        _, height, width, num_classes = one_hot.shape

    if class_ids is None:
        class_ids = list(range(num_classes))
    else:
        class_ids = list(class_ids)

    if compact:
        out_shape = (batch_size, len(class_ids), height, width) if channel_first else (batch_size, height, width, len(class_ids))
    else:
        out_shape = one_hot.shape

    out = np.zeros(out_shape, dtype=dtype)
    for b in range(batch_size):
        out[b] = signed_distance_map(
            one_hot[b],
            spacing=spacing,
            class_ids=class_ids,
            channel_first=channel_first,
            compact=compact,
            dtype=dtype,
        )
    return out
