import pandas as pd
from pathlib import Path
import json
from PIL import Image
import torch
import os
from transformers import AutoProcessor, AutoModelForVision2Seq

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_PATH = BASE_DIR / "data/HWL Materials.xlsx"
IMAGES_ROOT = BASE_DIR / "data/hwl_images"

OUTPUT_DATASET = BASE_DIR / "data/hwl_dataset.json"
OUTPUT_DESCRIPTIONS = BASE_DIR / "data/hwl_descriptions.json"

# ----------------------------
# DEVICE SETUP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"🚀 Using device: {device}")

# ----------------------------
# LOAD MODEL (Qwen2-VL-2B)
# ----------------------------
print("🔄 Loading Qwen2-VL-2B model...")

model_name = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=dtype
).to(device)

# ----------------------------
# CAPTION FUNCTION
# ----------------------------
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        prompt = (
            "You are an expert industrial engineer. "
            "Analyze the given image of a spare part and describe it clearly. "
            "Your answer must include:\n"
            "1. Part type (e.g., connector, valve, sensor, bolt, actuator)\n"
            "2. Material (metal, plastic, rubber, etc.)\n"
            "3. Shape and physical features (size, structure, color, components)\n"
            "4. Possible industrial use\n\n"
            "Keep the description concise, factual, and in one paragraph."
        )

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80)

        result = processor.decode(output[0], skip_special_tokens=True)

        # clean output
        result = result.replace("\n", " ").strip()

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return "No description available"

# ----------------------------
# LOAD EXCEL
# ----------------------------
print("📄 Reading Excel...")
df = pd.read_excel(EXCEL_PATH)

# Clean column names
df.columns = df.columns.str.strip()

dataset = []
descriptions_dict = {}

# ----------------------------
# MAIN LOOP
# ----------------------------
print("⚙️ Processing...")

for idx, row in df.iterrows():

    material_id = str(row["Material"]).strip()
    folder = IMAGES_ROOT / material_id

    image_files = []

    if folder.exists():
        image_files = sorted([
            f for f in folder.glob("*")
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ])

    image_list = [str(f).replace("\\", "/") for f in image_files]

    # ----------------------------
    # DESCRIPTION
    # ----------------------------
    if image_files:
        print(f"🧠 Generating: {material_id}")
        description = generate_caption(image_files[0])
    else:
        description = "No image available"

    # Save for CLIP engine
    descriptions_dict[material_id] = description

    # ----------------------------
    # DATA ENTRY
    # ----------------------------
    data_entry = {
        "material_id": material_id,
        "storage_type": str(row.get("Storage Type", "")),
        "storage_bin": str(row.get("Storage Bin", "")),
        "total_stock": int(row.get("Total Stock", 0)),
        "storage_location": str(row.get("Storage Location", "")),
        "plant": str(row.get("Plant", "")),
        "duration": int(row.get("Duration", 0)),
        "images": image_list,
        "description": description
    }

    dataset.append(data_entry)

    print(f"✅ {material_id}")

# ----------------------------
# SAVE FILES
# ----------------------------
print("💾 Saving files...")

# Save dataset.json
with open(OUTPUT_DATASET, "w") as f:
    json.dump(dataset, f, indent=2)

# Save descriptions.json
with open(OUTPUT_DESCRIPTIONS, "w") as f:
    json.dump(descriptions_dict, f, indent=2)

print("🎉 Done! Generated:")
print(" - hwl_dataset.json")
print(" - hwl_descriptions.json")