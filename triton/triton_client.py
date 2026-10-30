import os
import cv2
import argparse
import numpy as np
import tritonclient.grpc as grpcclient


# DINOv2 normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE  = (518, 518)
MODEL_NAME  = "jade_v3"
INPUT_NAME  = "image"
OUTPUT_NAME = "error_map"


def preprocess(image_path):
    """Load image and preprocess to DINOv2 input tensor."""
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE).astype(np.float32) / 255.0
    normed  = (resized - MEAN) / STD                          # (H, W, 3)
    tensor  = normed.transpose(2, 0, 1)[np.newaxis, ...]      # (1, 3, H, W)
    return tensor.astype(np.float32), raw


def infer(client, tensor):
    """Send tensor to Triton via gRPC and return error_map."""
    inp = grpcclient.InferInput(INPUT_NAME, tensor.shape, "FP32")
    inp.set_data_from_numpy(tensor)

    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)
    result = client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    return result.as_numpy(OUTPUT_NAME)   # (1, H, W)


def postprocess(error_map, raw_img, threshold, min_score=0.3, black_pixel_threshold=2):
    """Normalize, suppress, apply JET colormap, overlay on original."""
    err = error_map[0]   # (H, W)

    normalized = np.clip(err / threshold, 0, 1) if threshold > 0 else np.zeros_like(err)

    # Suppression 1: score below min_score → 0
    normalized[normalized < min_score] = 0

    # Suppression 2: black pixels in original → 0
    raw_resized = cv2.resize(raw_img, IMAGE_SIZE)
    black_mask  = np.max(raw_resized, axis=2) <= black_pixel_threshold
    normalized[black_mask] = 0

    heatmap       = (normalized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed     = cv2.addWeighted(raw_resized, 0.6, heatmap_color, 0.4, 0)
    comparison    = np.concatenate([raw_resized, overlayed], axis=1)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="JADE_v3 Triton gRPC client")
    parser.add_argument("-i", "--image",     required=True,  help="Input image path")
    parser.add_argument("-o", "--output",    required=True,  help="Output image path")
    parser.add_argument("-t", "--threshold", required=True,  help="Path to threshold.npy")
    parser.add_argument("-u", "--url",       default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--min_score",            type=float, default=0.3)
    parser.add_argument("--black_pixel_threshold", type=int,  default=2)
    args = parser.parse_args()

    # Load threshold
    threshold = float(np.load(args.threshold))
    print(f"[Client] Threshold: {threshold:.6f}")

    # Connect
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    print(f"[Client] Connected to Triton: {args.url}")

    # Preprocess
    tensor, raw_img = preprocess(args.image)
    print(f"[Client] Input tensor: {tensor.shape}")

    # Infer
    error_map = infer(client, tensor)
    print(f"[Client] Error map: shape={error_map.shape} min={error_map.min():.4f} max={error_map.max():.4f} p99={np.percentile(error_map,99):.4f}")

    # Postprocess
    result = postprocess(error_map, raw_img, threshold,
                         min_score=args.min_score,
                         black_pixel_threshold=args.black_pixel_threshold)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"[Client] Saved: {args.output}")


if __name__ == "__main__":
    main()
