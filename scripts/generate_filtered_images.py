#!/usr/bin/env python3
"""Generate processed images from transmission map and filter by SSIM threshold."""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import json
from tqdm import tqdm

from chest_xray_sim.inverse.metrics import ssim

import chest_xray_sim.inverse.operators as ops 

# SSIM threshold for filtering generated images
SSIM_THRESHOLD = 0.8

# Parameter ranges for random generation
PARAM_RANGES = {
    "enhance_factor": (0.1, .9),
    "window_center": (0.2, 0.4),
    "window_width": (0.2, 0.5),
    "gamma": (4.0, 10.0),  # for sigmoid windowing
    "low_sigma": (1.0, 4.0),
    # "window_type": ("sigmoid"),
}


def load_image(path: str) -> np.ndarray:
    """Load image from path and normalize to [0, 1]."""
    img = np.array(Image.open(path).convert("L"))
    return img.astype(np.float32) / 255.0


def save_image(img: np.ndarray, path: str):
    """Save image to path."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def generate_random_params(key) -> dict:
    """Generate random parameters for the processing pipeline."""
    keys = jax.random.split(key, len(PARAM_RANGES))
    params = {}

    params['window_type'] = 'sigmoid' 
    for (param_name, (min_val, max_val)), subkey in zip(PARAM_RANGES.items(), keys):
        # if param_name == "window_type":
        #     val = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
        #     params[param_name] = "linear" if val < 0.5 else "sigmoid"

        params[param_name] = jax.random.uniform(
            subkey, minval=min_val, maxval=max_val
        )
        if param_name == "low_sigma":
            params[param_name] = params[param_name].item()

    return params


def process_transmission_map(
    tm: jnp.ndarray, params: dict, windowing_type: str = "sigmoid"
) -> jnp.ndarray:
    x = ops.window(
        tm,
        params["window_center"],
        params["window_width"],
        params.get("gamma", 0.0),
        params['window_type']
    )
    x = ops.negative_log(x)
    x = ops.unsharp_masking(x, params['low_sigma'], params["enhance_factor"])
    x = ops.clipping(x)
    return x

def generate_and_filter_images(
    tm_path: str,
    real_processed_path: str,
    output_dir: str,
    num_images: int,
    seed: int = 42,
    windowing_type: str = "sigmoid",
):
    """Generate processed images and filter by SSIM threshold."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    print("Loading images...")
    tm = load_image(tm_path)
    real_processed = load_image(real_processed_path)

    # Convert to JAX arrays
    tm_jax = jnp.array(tm)

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Track statistics
    accepted_count = 0
    generated_count = 0
    ssim_scores = []
    accepted_params = []

    print(f"Generating {num_images} images with SSIM threshold {SSIM_THRESHOLD}...")

    with tqdm(total=num_images) as pbar:
        while accepted_count < num_images:
            # Generate random parameters
            key, subkey = jax.random.split(key)
            params = generate_random_params(subkey)

            # Process transmission map
            processed = process_transmission_map(tm_jax, params, windowing_type)
            processed_np = np.array(processed)

            # Calculate SSIM
            ssim_score = ssim(real_processed, processed_np).item()
            generated_count += 1
            ssim_scores.append(ssim_score)

            # Check threshold
            if ssim_score >= SSIM_THRESHOLD:
                # Save image
                img_name = f"generated_{accepted_count:04d}.png"
                img_path = os.path.join(output_dir, img_name)
                save_image(processed_np, img_path)

                # Save parameters
                param_dict = {k: float(v) if not isinstance(v, str) else v for k, v in params.items()}
                param_dict["ssim_score"] = float(ssim_score)
                accepted_params.append(param_dict)

                accepted_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "accepted": accepted_count,
                        "generated": generated_count,
                        "accept_rate": f"{accepted_count/generated_count:.2%}",
                        "last_ssim": f"{ssim_score:.4f}",
                    }
                )

    # Save metadata
    metadata = {
        "tm_path": tm_path,
        "real_processed_path": real_processed_path,
        "ssim_threshold": SSIM_THRESHOLD,
        "num_generated": generated_count,
        "num_accepted": accepted_count,
        "acceptance_rate": accepted_count / generated_count,
        "ssim_stats": {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "min": float(np.min(ssim_scores)),
            "max": float(np.max(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        },
        "accepted_parameters": accepted_params,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nGeneration complete!")
    print(f"Generated {generated_count} images to get {accepted_count} accepted images")
    print(f"Acceptance rate: {accepted_count/generated_count:.2%}")
    print(
        f"SSIM stats - Mean: {metadata['ssim_stats']['mean']:.4f}, "
        f"Std: {metadata['ssim_stats']['std']:.4f}, "
        f"Range: [{metadata['ssim_stats']['min']:.4f}, {metadata['ssim_stats']['max']:.4f}]"
    )
    print(f"Results saved to: {output_dir}")


def main():
    num_images = 30

    root = '/Volumes/T7/projs/thesis/'

    tm_path = "data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif"
    tm_path = os.path.join(root, tm_path)
    proc_path = "data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif"
    proc_path = os.path.join(root, proc_path)
    output_dir = "data/generated"
    output_dir = os.path.join(root, output_dir)

    generate_and_filter_images(
        tm_path,
        proc_path,
        output_dir,
        num_images,
    )


if __name__ == "__main__":
    main()
