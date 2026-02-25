import os
import cv2
import numpy as np

# Enable OpenEXR Support 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from CorridorKeyModule.core import color_utils as cu

def test_despill(exr_path, output_dir):
    print(f"Loading '{exr_path}'...")
    
    if not os.path.exists(exr_path):
        print(f"Error: File not found: {exr_path}")
        return
        
    img_linear = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if img_linear is None:
        print(f"Error: Failed to load {exr_path}")
        return
        
    # Standard inference pipeline converts EXRs to RGB and passes them.
    # We will grab just the RGB channels and ensure it's linear float
    if img_linear.ndim == 3 and img_linear.shape[2] > 3:
        img_linear = img_linear[:, :, :3] # keep RGB
    
    # Needs to be RGB, not BGR
    img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)
    
    # In the engine, it processes as sRGB for the despill step.
    img_srgb = cu.linear_to_srgb(np.maximum(img_linear_rgb, 0.0))
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(exr_path))[0]
    
    print("Running despill tests...")
    
    # Save original (sRGB representation for viewing)
    orig_bgr = cv2.cvtColor((np.clip(img_srgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), orig_bgr)
    
    # 1. Test Despill Strength 1.0 (Full Luminance Preserving)
    despill_full = cu.despill(img_srgb, strength=1.0)
    desp_bgr_full = cv2.cvtColor((np.clip(despill_full, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_despill_1.0.png"), desp_bgr_full)
    
    # 2. Test Despill Strength 0.5 (Half)
    despill_half = cu.despill(img_srgb, strength=0.5)
    desp_bgr_half = cv2.cvtColor((np.clip(despill_half, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_despill_0.5.png"), desp_bgr_half)
    
    # 3. Test Old Despill (For comparison - mimicking the old simple clamp)
    r = img_srgb[..., 0]
    g = img_srgb[..., 1]
    b = img_srgb[..., 2]
    limit = (r + b) / 2.0
    g_old = np.minimum(g, limit)
    despill_old = np.stack([r, g_old, b], axis=-1)
    
    desp_bgr_old = cv2.cvtColor((np.clip(despill_old, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_despill_OLD.png"), desp_bgr_old)

    print(f"Saved test renders to {output_dir}")

if __name__ == "__main__":
    target = "/home/corridor/CorridorKey/ClipsForInference/EpicHair/Input/EpicHair00091722.exr"
    out_dir = "/home/corridor/CorridorKey/test_despill_output"
    test_despill(target, out_dir)
