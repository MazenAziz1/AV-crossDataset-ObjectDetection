from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int

    nominal_scale: float
    actual_scale_x: float
    actual_scale_y: float

    resized_width: int
    resized_height: int

    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int

    def to_dict(self) -> dict:
        return asdict(self)


def round_half_up(value: float) -> int:
    """
    Deterministic positive-number rounding:

        floor(value + 0.5)
    """
    if not math.isfinite(value):
        raise ValueError(
            f"Cannot round non-finite value: {value}"
        )

    return int(
        math.floor(value + 0.5)
    )


def calculate_letterbox_transform(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxTransform:
    """
    Calculate centered, aspect-ratio-preserving letterbox geometry.
    """
    source_width = int(source_width)
    source_height = int(source_height)
    target_width = int(target_width)
    target_height = int(target_height)

    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            "Source dimensions must be positive."
        )

    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            "Target dimensions must be positive."
        )

    nominal_scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = min(
        target_width,
        round_half_up(
            source_width * nominal_scale
        ),
    )

    resized_height = min(
        target_height,
        round_half_up(
            source_height * nominal_scale
        ),
    )

    if resized_width <= 0 or resized_height <= 0:
        raise ValueError(
            "Calculated resized dimensions are not positive."
        )

    actual_scale_x = (
        resized_width / source_width
    )

    actual_scale_y = (
        resized_height / source_height
    )

    remaining_width = (
        target_width - resized_width
    )

    remaining_height = (
        target_height - resized_height
    )

    if remaining_width < 0 or remaining_height < 0:
        raise ValueError(
            "Resized image is larger than the target canvas."
        )

    padding_left = (
        remaining_width // 2
    )

    padding_top = (
        remaining_height // 2
    )

    padding_right = (
        remaining_width - padding_left
    )

    padding_bottom = (
        remaining_height - padding_top
    )

    transform = LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        nominal_scale=float(
            nominal_scale
        ),
        actual_scale_x=float(
            actual_scale_x
        ),
        actual_scale_y=float(
            actual_scale_y
        ),
        resized_width=resized_width,
        resized_height=resized_height,
        padding_left=padding_left,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
    )

    validate_transform_geometry(
        transform
    )

    return transform


def validate_transform_geometry(
    transform: LetterboxTransform,
) -> None:
    if (
        transform.resized_width
        + transform.padding_left
        + transform.padding_right
        != transform.target_width
    ):
        raise ValueError(
            "Horizontal transform geometry is inconsistent."
        )

    if (
        transform.resized_height
        + transform.padding_top
        + transform.padding_bottom
        != transform.target_height
    ):
        raise ValueError(
            "Vertical transform geometry is inconsistent."
        )

    if abs(
        transform.padding_left
        - transform.padding_right
    ) > 1:
        raise ValueError(
            "Horizontal padding is not centered."
        )

    if abs(
        transform.padding_top
        - transform.padding_bottom
    ) > 1:
        raise ValueError(
            "Vertical padding is not centered."
        )

    expected_scale_x = (
        transform.resized_width
        / transform.source_width
    )

    expected_scale_y = (
        transform.resized_height
        / transform.source_height
    )

    if not math.isclose(
        transform.actual_scale_x,
        expected_scale_x,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "actual_scale_x is inconsistent."
        )

    if not math.isclose(
        transform.actual_scale_y,
        expected_scale_y,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "actual_scale_y is inconsistent."
        )


def apply_letterbox_bgr(
    source_image: np.ndarray,
    transform: LetterboxTransform,
    padding_value: int,
) -> tuple[np.ndarray, str]:
    """
    Apply a frozen letterbox transform to an OpenCV BGR image.

    PNG/JPEG files themselves do not encode a BGR/RGB channel label.
    OpenCV decodes and writes using BGR internally while preserving
    the displayed image colors correctly.
    """
    if source_image is None:
        raise ValueError(
            "Source image is None."
        )

    if source_image.ndim != 3:
        raise ValueError(
            "Source image must have three dimensions."
        )

    if source_image.shape[2] != 3:
        raise ValueError(
            "Source image must have three color channels."
        )

    actual_height, actual_width = (
        source_image.shape[:2]
    )

    if (
        actual_width != transform.source_width
        or actual_height != transform.source_height
    ):
        raise ValueError(
            "Source image dimensions do not match the transform."
        )

    padding_value = int(
        padding_value
    )

    if not 0 <= padding_value <= 255:
        raise ValueError(
            "Padding value must be between 0 and 255."
        )

    if (
        transform.resized_width
        == transform.source_width
        and transform.resized_height
        == transform.source_height
    ):
        resized = source_image.copy()
        interpolation_name = "identity"

    else:
        is_downscale = (
            transform.resized_width
            < transform.source_width
            or transform.resized_height
            < transform.source_height
        )

        if is_downscale:
            interpolation = cv2.INTER_AREA
            interpolation_name = (
                "opencv_inter_area"
            )
        else:
            interpolation = cv2.INTER_LINEAR
            interpolation_name = (
                "opencv_inter_linear"
            )

        resized = cv2.resize(
            source_image,
            (
                transform.resized_width,
                transform.resized_height,
            ),
            interpolation=interpolation,
        )

    canvas = np.full(
        (
            transform.target_height,
            transform.target_width,
            3,
        ),
        fill_value=padding_value,
        dtype=np.uint8,
    )

    x1 = transform.padding_left
    y1 = transform.padding_top

    x2 = (
        x1 + transform.resized_width
    )

    y2 = (
        y1 + transform.resized_height
    )

    canvas[
        y1:y2,
        x1:x2,
    ] = resized

    return canvas, interpolation_name


def verify_padding(
    image: np.ndarray,
    transform: LetterboxTransform,
    padding_value: int,
) -> bool:
    """
    Confirm that every padded region contains only the frozen value.
    """
    expected = np.array(
        [
            padding_value,
            padding_value,
            padding_value,
        ],
        dtype=np.uint8,
    )

    checks: list[bool] = []

    if transform.padding_top > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :transform.padding_top,
                        :,
                    ]
                    == expected
                )
            )
        )

    if transform.padding_bottom > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        image.shape[0]
                        - transform.padding_bottom:,
                        :,
                    ]
                    == expected
                )
            )
        )

    if transform.padding_left > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :,
                        :transform.padding_left,
                    ]
                    == expected
                )
            )
        )

    if transform.padding_right > 0:
        checks.append(
            bool(
                np.all(
                    image[
                        :,
                        image.shape[1]
                        - transform.padding_right:,
                    ]
                    == expected
                )
            )
        )

    return all(checks) if checks else True


def calculate_aspect_ratio_error(
    transform: LetterboxTransform,
) -> float:
    source_ratio = (
        transform.source_width
        / transform.source_height
    )

    resized_ratio = (
        transform.resized_width
        / transform.resized_height
    )

    return abs(
        source_ratio - resized_ratio
    )


def maximum_allowed_aspect_ratio_error(
    transform: LetterboxTransform,
) -> float:
    """
    One-pixel dimension rounding may produce a tiny ratio difference.
    """
    return (
        max(
            1.0 / transform.resized_width,
            1.0 / transform.resized_height,
        )
        * 2.0
    )


def transform_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """
    Transform a source absolute XYXY box to processed-image XYXY.
    """
    transformed_xmin = (
        float(xmin)
        * transform.actual_scale_x
        + transform.padding_left
    )

    transformed_ymin = (
        float(ymin)
        * transform.actual_scale_y
        + transform.padding_top
    )

    transformed_xmax = (
        float(xmax)
        * transform.actual_scale_x
        + transform.padding_left
    )

    transformed_ymax = (
        float(ymax)
        * transform.actual_scale_y
        + transform.padding_top
    )

    return (
        transformed_xmin,
        transformed_ymin,
        transformed_xmax,
        transformed_ymax,
    )


def inverse_transform_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """
    Invert a processed-image XYXY box back to source coordinates.
    """
    source_xmin = (
        (
            float(xmin)
            - transform.padding_left
        )
        / transform.actual_scale_x
    )

    source_ymin = (
        (
            float(ymin)
            - transform.padding_top
        )
        / transform.actual_scale_y
    )

    source_xmax = (
        (
            float(xmax)
            - transform.padding_left
        )
        / transform.actual_scale_x
    )

    source_ymax = (
        (
            float(ymax)
            - transform.padding_top
        )
        / transform.actual_scale_y
    )

    return (
        source_xmin,
        source_ymin,
        source_xmax,
        source_ymax,
    )