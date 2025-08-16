from rfdetr import RFDETRBase
import torch
import os
import sys

# ----------------------------
# Dataset and output paths
# ----------------------------
DATASET_DIR = r"C:\Users\luisd\OneDrive\Desktop\Differentspanners.v2i.coco"
OUTPUT_DIR = r"C:\Users\luisd\OneDrive\Desktop\Differentspanners.v2i.coco_output"

# Subfolders to check
SUBFOLDERS = ["train", "valid", "test"]
JSON_FILENAME = "_annotations.coco.json"

# Training parameters
EPOCHS = 20
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

# ----------------------------
# Verify dataset exists
# ----------------------------
train_json_path = os.path.join(DATASET_DIR, "train", JSON_FILENAME)
valid_json_path = os.path.join(DATASET_DIR, "valid", JSON_FILENAME)
test_json_path  = os.path.join(DATASET_DIR, "test", JSON_FILENAME)

for path in [train_json_path, valid_json_path, test_json_path]:
    if not os.path.exists(path):
        print(f"ERROR: JSON file not found: {path}")
        sys.exit(1)
    else:
        print(f"Found JSON: {path}")

# ----------------------------
# Main training
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on device: {device}\n")

    # Initialize model with ResNet50 backbone (safe for RTX 5070)
    model = RFDETRBase(backbone="resnet50", device=device)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Enable autograd
    torch.set_grad_enabled(True)
    torch.backends.cudnn.benchmark = True

    # Train the model
    model.train(
        dataset_dir=DATASET_DIR,      # parent folder containing train/valid/test
        dataset_file='roboflow',      # automatically detects _annotations.coco.json
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        output_dir=OUTPUT_DIR
    )

    print(f"\nTraining complete. Model saved in: {OUTPUT_DIR}")
