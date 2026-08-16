from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np


def derive_augmentation_seed(
    base_seed: int,
    epoch: int,
    global_image_id: int,
) -> int:
    """
    Derive a stable 64-bit augmentation seed.

    Python's built-in hash() is intentionally not used because it may
    differ across interpreter sessions.
    """
    payload = (
        f"{int(base_seed)}:"
        f"{int(epoch)}:"
        f"{int(global_image_id)}"
    ).encode("utf-8")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


def validate_xyxy_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    tolerance: float = 1e-8,
) -> None:
    boxes = np.asarray(
        boxes,
        dtype=np.float64,
    )

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            "Boxes must have shape [N, 4] in XYXY format."
        )

    if not np.isfinite(boxes).all():
        raise ValueError(
            "Boxes contain non-finite coordinates."
        )

    if len(boxes) == 0:
        return

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    if np.any(xmax <= xmin):
        raise ValueError(
            "At least one box has non-positive width."
        )

    if np.any(ymax <= ymin):
        raise ValueError(
            "At least one box has non-positive height."
        )

    if np.any(xmin < -tolerance):
        raise ValueError(
            "At least one box has xmin below zero."
        )

    if np.any(ymin < -tolerance):
        raise ValueError(
            "At least one box has ymin below zero."
        )

    if np.any(
        xmax > image_width + tolerance
    ):
        raise ValueError(
            "At least one box exceeds image width."
        )

    if np.any(
        ymax > image_height + tolerance
    ):
        raise ValueError(
            "At least one box exceeds image height."
        )


def clip_xyxy_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    clipped = np.asarray(
        boxes,
        dtype=np.float64,
    ).copy()

    if len(clipped) == 0:
        return clipped.reshape(0, 4)

    clipped[:, [0, 2]] = np.clip(
        clipped[:, [0, 2]],
        0.0,
        float(image_width),
    )

    clipped[:, [1, 3]] = np.clip(
        clipped[:, [1, 3]],
        0.0,
        float(image_height),
    )

    return clipped


def flip_xyxy_horizontally(
    boxes: np.ndarray,
    image_width: int,
) -> np.ndarray:
    flipped = np.asarray(
        boxes,
        dtype=np.float64,
    ).copy()

    if len(flipped) == 0:
        return flipped.reshape(0, 4)

    original_xmin = flipped[:, 0].copy()
    original_xmax = flipped[:, 2].copy()

    flipped[:, 0] = (
        float(image_width)
        - original_xmax
    )

    flipped[:, 2] = (
        float(image_width)
        - original_xmin
    )

    return flipped


def apply_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_limit: float,
    contrast_limit: float,
) -> tuple[np.ndarray, dict[str, float]]:
    contrast_alpha = 1.0 + float(
        rng.uniform(
            -contrast_limit,
            contrast_limit,
        )
    )

    brightness_beta = 255.0 * float(
        rng.uniform(
            -brightness_limit,
            brightness_limit,
        )
    )

    transformed = (
        image.astype(np.float32)
        * contrast_alpha
        + brightness_beta
    )

    transformed = np.clip(
        transformed,
        0.0,
        255.0,
    ).astype(np.uint8)

    return transformed, {
        "contrast_alpha": contrast_alpha,
        "brightness_beta": brightness_beta,
    }


def apply_hsv_adjustment(
    image: np.ndarray,
    rng: np.random.Generator,
    hue_shift_limit: int,
    saturation_minimum: float,
    saturation_maximum: float,
    value_minimum: float,
    value_maximum: float,
) -> tuple[np.ndarray, dict[str, float]]:
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)

    hue_shift = int(
        rng.integers(
            -hue_shift_limit,
            hue_shift_limit + 1,
        )
    )

    saturation_scale = float(
        rng.uniform(
            saturation_minimum,
            saturation_maximum,
        )
    )

    value_scale = float(
        rng.uniform(
            value_minimum,
            value_maximum,
        )
    )

    hsv[:, :, 0] = np.mod(
        hsv[:, :, 0] + hue_shift,
        180.0,
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1]
        * saturation_scale,
        0.0,
        255.0,
    )

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2]
        * value_scale,
        0.0,
        255.0,
    )

    transformed = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    return transformed, {
        "hue_shift_opencv": hue_shift,
        "saturation_scale": saturation_scale,
        "value_scale": value_scale,
    }


def apply_gaussian_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    kernel_choices: list[int],
    sigma_minimum: float,
    sigma_maximum: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid_kernels = [
        int(kernel)
        for kernel in kernel_choices
        if int(kernel) > 0
        and int(kernel) % 2 == 1
    ]

    if not valid_kernels:
        raise ValueError(
            "Gaussian-blur kernels must contain "
            "at least one positive odd integer."
        )

    kernel = int(
        rng.choice(valid_kernels)
    )

    sigma = float(
        rng.uniform(
            sigma_minimum,
            sigma_maximum,
        )
    )

    transformed = cv2.GaussianBlur(
        image,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return transformed, {
        "kernel": kernel,
        "sigma": sigma,
    }


def apply_training_augmentation(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    configuration: dict[str, Any],
    global_image_id: int,
    epoch: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Apply the frozen shared augmentation policy.

    The input image must already be the processed 640x640 letterboxed
    BGR image. Boxes must use absolute XYXY coordinates.
    """
    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "Input image must contain three channels."
        )

    input_configuration = configuration[
        "input"
    ]

    expected_width = int(
        input_configuration["width"]
    )

    expected_height = int(
        input_configuration["height"]
    )

    image_height, image_width = (
        image.shape[:2]
    )

    if (
        image_width != expected_width
        or image_height != expected_height
    ):
        raise ValueError(
            "Input image dimensions differ from "
            "the frozen augmentation policy."
        )

    boxes = np.asarray(
        boxes_xyxy,
        dtype=np.float64,
    ).copy()

    class_ids = np.asarray(
        class_ids,
        dtype=np.int64,
    ).copy()

    if len(boxes) != len(class_ids):
        raise ValueError(
            "Box count and class-ID count differ."
        )

    validate_xyxy_boxes(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )

    base_seed = int(
        configuration[
            "execution"
        ]["base_seed"]
    )

    seed = derive_augmentation_seed(
        base_seed=base_seed,
        epoch=epoch,
        global_image_id=global_image_id,
    )

    rng = np.random.default_rng(seed)

    output_image = image.copy()

    padding_value = int(
        configuration[
            "letterbox_padding"
        ]["value"]
    )

    preserve_padding = bool(
        configuration[
            "letterbox_padding"
        ][
            "preserve_padding_value_after_photometric_transforms"
        ]
    )

    original_padding_mask = np.all(
        output_image
        == np.asarray(
            [
                padding_value,
                padding_value,
                padding_value,
            ],
            dtype=np.uint8,
        ),
        axis=2,
    )

    transforms = configuration[
        "transforms"
    ]

    operations_applied: list[str] = []
    parameters: dict[str, Any] = {}

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    horizontal = transforms[
        "horizontal_flip"
    ]

    horizontal_applied = (
        bool(horizontal["enabled"])
        and rng.random()
        < float(
            horizontal["probability"]
        )
    )

    if horizontal_applied:
        output_image = np.ascontiguousarray(
            output_image[:, ::-1]
        )

        original_padding_mask = (
            np.ascontiguousarray(
                original_padding_mask[
                    :,
                    ::-1,
                ]
            )
        )

        boxes = flip_xyxy_horizontally(
            boxes,
            image_width=image_width,
        )

        operations_applied.append(
            "horizontal_flip"
        )

        parameters[
            "horizontal_flip"
        ] = {
            "applied": True,
        }

    else:
        parameters[
            "horizontal_flip"
        ] = {
            "applied": False,
        }

    # --------------------------------------------------------
    # Brightness and contrast
    # --------------------------------------------------------

    brightness_contrast = transforms[
        "brightness_contrast"
    ]

    brightness_contrast_applied = (
        bool(
            brightness_contrast[
                "enabled"
            ]
        )
        and rng.random()
        < float(
            brightness_contrast[
                "probability"
            ]
        )
    )

    if brightness_contrast_applied:
        (
            output_image,
            operation_parameters,
        ) = apply_brightness_contrast(
            image=output_image,
            rng=rng,
            brightness_limit=float(
                brightness_contrast[
                    "brightness_limit"
                ]
            ),
            contrast_limit=float(
                brightness_contrast[
                    "contrast_limit"
                ]
            ),
        )

        operations_applied.append(
            "brightness_contrast"
        )

        parameters[
            "brightness_contrast"
        ] = {
            "applied": True,
            **operation_parameters,
        }

    else:
        parameters[
            "brightness_contrast"
        ] = {
            "applied": False,
        }

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv_configuration = transforms[
        "hsv_adjustment"
    ]

    hsv_applied = (
        bool(
            hsv_configuration[
                "enabled"
            ]
        )
        and rng.random()
        < float(
            hsv_configuration[
                "probability"
            ]
        )
    )

    if hsv_applied:
        (
            output_image,
            operation_parameters,
        ) = apply_hsv_adjustment(
            image=output_image,
            rng=rng,
            hue_shift_limit=int(
                hsv_configuration[
                    "hue_shift_limit_opencv"
                ]
            ),
            saturation_minimum=float(
                hsv_configuration[
                    "saturation_scale"
                ]["minimum"]
            ),
            saturation_maximum=float(
                hsv_configuration[
                    "saturation_scale"
                ]["maximum"]
            ),
            value_minimum=float(
                hsv_configuration[
                    "value_scale"
                ]["minimum"]
            ),
            value_maximum=float(
                hsv_configuration[
                    "value_scale"
                ]["maximum"]
            ),
        )

        operations_applied.append(
            "hsv_adjustment"
        )

        parameters[
            "hsv_adjustment"
        ] = {
            "applied": True,
            **operation_parameters,
        }

    else:
        parameters[
            "hsv_adjustment"
        ] = {
            "applied": False,
        }

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------

    blur_configuration = transforms[
        "gaussian_blur"
    ]

    blur_applied = (
        bool(
            blur_configuration[
                "enabled"
            ]
        )
        and rng.random()
        < float(
            blur_configuration[
                "probability"
            ]
        )
    )

    if blur_applied:
        (
            output_image,
            operation_parameters,
        ) = apply_gaussian_blur(
            image=output_image,
            rng=rng,
            kernel_choices=list(
                blur_configuration[
                    "kernel_choices"
                ]
            ),
            sigma_minimum=float(
                blur_configuration[
                    "sigma"
                ]["minimum"]
            ),
            sigma_maximum=float(
                blur_configuration[
                    "sigma"
                ]["maximum"]
            ),
        )

        operations_applied.append(
            "gaussian_blur"
        )

        parameters[
            "gaussian_blur"
        ] = {
            "applied": True,
            **operation_parameters,
        }

    else:
        parameters[
            "gaussian_blur"
        ] = {
            "applied": False,
        }

    if preserve_padding:
        output_image[
            original_padding_mask
        ] = padding_value

    boxes = clip_xyxy_boxes(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )

    validate_xyxy_boxes(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )

    trace = {
        "base_seed": base_seed,
        "derived_seed": seed,
        "epoch": int(epoch),
        "global_image_id": int(
            global_image_id
        ),
        "operations_applied": (
            operations_applied
        ),
        "parameters": parameters,
        "padding_pixels_preserved": bool(
            preserve_padding
        ),
    }

    return (
        output_image,
        boxes,
        class_ids,
        trace,
    )