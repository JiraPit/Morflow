# `morphology`

Mathematical morphology operators (dilation, erosion, opening, closing, gradient) for binary and grayscale tensors.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `op` | `string` | `"dilate"` | Morphological operation: `"dilate"`, `"erode"`, `"open"`, `"close"`, `"gradient"` (positional arg 0). |
| `kernel_size` / `ksize` | `int` | `3` | Structuring element size ($K \times K$). |
| `shape` | `string` | `"rect"` | Structuring element shape: `"rect"`, `"cross"`, `"ellipse"`. |
| `iterations` / `iter` | `int` | `1` | Number of sequential application cycles. |

## Behavior
- Dilation evaluates neighborhood local maximum; erosion evaluates neighborhood local minimum.
- Chains primitive operations for compound modes: Open ($\text{Erode} \rightarrow \text{Dilate}$), Close ($\text{Dilate} \rightarrow \text{Erode}$), Gradient ($\text{Dilate} - \text{Erode}$).
- Rows processed concurrently across Rayon worker threads.

## OpenCV backend
Uses shared OpenCV through the versioned `opencv-bridge` plugin. Declare `plugin opencv-bridge/0.1.2` in the pipeline and run `morflow prep`. Shape checking does not load OpenCV. See `actions/image_opencv/README.md` for installation, layout rules and numerical differences.
