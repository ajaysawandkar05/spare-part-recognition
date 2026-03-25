"""
scripts/generate_descriptions.py
----------------------------------
Generates rich text descriptions for all 100 HWL materials
using Qwen2-VL-72B via OpenRouter.

Run from project root:
    set OPENROUTER_API_KEY=sk-or-your-key-here   (Windows)
    python scripts/generate_descriptions.py

After completion, rebuild the CLIP index:
    curl -X POST http://localhost:8000/api/index/rebuild
"""

import os
import base64
import time
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL              = "qwen/qwen3-vl-32b-instruct"
DATASET_JSON       = "data/hwl_dataset.json"
IMAGES_DIR         = "data/hwl_images"
OUTPUT_JSON        = "data/hwl_descriptions.json"   # used by clip_engine.py
OUTPUT_CSV         = "data/hwl_descriptions.csv"    # human-readable backup
DELAY_BETWEEN_REQS = 0.5                            # seconds, increase if rate-limited

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://hwl-spare-parts.local",
        "X-Title":      "HWL Spare Parts Intelligence",
    }
)

PROMPT = """You are an expert industrial spare parts cataloguing system used in a warehouse retrieval application.

Analyze this spare part image and generate a structured description optimized for semantic search.

Output exactly these 6 fields, one per line, no extra text:

PART_TYPE: [component category — e.g. photoelectric sensor, pneumatic valve, cable connector, bracket, relay, actuator]
PHYSICAL: [shape, size estimate, color, material — e.g. cylindrical black plastic housing, 70mm length, M18 thread]
BRAND_MODEL: [any visible brand name, model number, serial markings — write "none visible" if absent]
FUNCTION: [what this part does in one sentence — e.g. detects object presence using infrared beam]
FEATURES: [notable technical features — connectors, LEDs, mounting style, ports, cable type, thread size]
SEARCH_TAGS: [5-8 comma-separated keywords a maintenance engineer would search for]

Rules:
- Only describe what is visually confirmed in the image
- Be specific and technical, not generic
- Never say "the image shows" or "this appears to be"
- If unsure about a detail, omit it rather than guess"""


def encode_image(image_path):
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")

    # Resize if larger than 1024px on any side
    max_size = 1024
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Compress to JPEG — much smaller than PNG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8"), "image/jpeg"


def parse_response(raw, material_id, metadata):
    fields = {k: "" for k in ["part_type", "physical", "brand_model",
                               "function", "features", "search_tags"]}
    key_map = {
        "PART_TYPE":   "part_type",
        "PHYSICAL":    "physical",
        "BRAND_MODEL": "brand_model",
        "FUNCTION":    "function",
        "FEATURES":    "features",
        "SEARCH_TAGS": "search_tags",
    }
    for line in raw.strip().splitlines():
        for prefix, field in key_map.items():
            if line.upper().startswith(f"{prefix}:"):
                fields[field] = line[len(prefix)+1:].strip()

    # Add warehouse metadata as extra context
    meta_tags = []
    if metadata.get("storage_bin"):      meta_tags.append(f"bin {metadata['storage_bin']}")
    if metadata.get("plant"):            meta_tags.append(f"plant {metadata['plant']}")
    if metadata.get("storage_location"): meta_tags.append(metadata["storage_location"])

    # full_description is what CLIP encodes for text search
    parts = []
    if fields["part_type"]:   parts.append(fields["part_type"])
    if fields["physical"]:    parts.append(fields["physical"])
    if fields["brand_model"] and fields["brand_model"].lower() != "none visible":
        parts.append(fields["brand_model"])
    if fields["function"]:    parts.append(fields["function"])
    if fields["features"]:    parts.append(fields["features"])
    if fields["search_tags"]: parts.append(fields["search_tags"])
    if meta_tags:             parts.append(", ".join(meta_tags))

    return {
        "material_id":      material_id,
        "part_type":        fields["part_type"],
        "physical":         fields["physical"],
        "brand_model":      fields["brand_model"],
        "function":         fields["function"],
        "features":         fields["features"],
        "search_tags":      fields["search_tags"],
        "full_description": ". ".join(parts),
        "storage_bin":      metadata.get("storage_bin", ""),
        "plant":            metadata.get("plant", ""),
        "total_stock":      metadata.get("total_stock", ""),
        "raw_response":     raw,
    }


def call_api(image_path, material_id, metadata, retries=3):
    img_b64, mime = encode_image(image_path)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    ],
                }],
            )
            raw = response.choices[0].message.content.strip()
            return parse_response(raw, material_id, metadata)

        except Exception as e:
            print(f"  Warning attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return parse_response(
        f"PART_TYPE: industrial spare part\nSEARCH_TAGS: spare part, {material_id}",
        material_id, metadata
    )


def find_best_image(mat_id):
    mat_dir = Path(IMAGES_DIR) / mat_id
    if not mat_dir.exists():
        return None
    for name in ["image_1.png", "image_1.jpg", "image_1.jpeg"]:
        p = mat_dir / name
        if p.exists():
            return str(p)
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        candidates = sorted(mat_dir.glob(ext))
        if candidates:
            return str(candidates[0])
    return None


def main():
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set.\n"
            "Windows: set OPENROUTER_API_KEY=sk-or-...\n"
            "Get key: https://openrouter.ai/keys"
        )

    with open(DATASET_JSON) as f:
        dataset = json.load(f)
    metadata_map = {item["material_id"]: item for item in dataset}

    # Resume support
    existing = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            existing = json.load(f)
        print(f"[resume] {len(existing)} already described, skipping.\n")

    csv_rows = []
    if os.path.exists(OUTPUT_CSV):
        csv_rows = pd.read_csv(OUTPUT_CSV).to_dict("records")

    total = len(dataset)
    done = skipped = failed = 0

    print(f"[start] {total} materials  |  model: {MODEL}")
    print(f"        images : {IMAGES_DIR}")
    print(f"        output : {OUTPUT_JSON}\n")

    for i, item in enumerate(dataset, 1):
        mat_id = item["material_id"]

        if mat_id in existing:
            skipped += 1
            print(f"[{i:>3}/{total}] skip  {mat_id}")
            continue

        img_path = find_best_image(mat_id)
        if not img_path:
            print(f"[{i:>3}/{total}] WARN  {mat_id} — no image, skipping")
            failed += 1
            continue

        print(f"[{i:>3}/{total}] {mat_id} ... ", end="", flush=True)
        result = call_api(img_path, mat_id, metadata_map.get(mat_id, {}))

        existing[mat_id] = result["full_description"]
        csv_rows.append(result)
        done += 1

        preview = result["full_description"][:110]
        status  = "OK" if result["raw_response"] != "API_FAILURE" else "FALLBACK"
        print(f"[{status}]")
        print(f"         {preview}{'...' if len(result['full_description']) > 110 else ''}")

        # Save after every material
        with open(OUTPUT_JSON, "w") as f:
            json.dump(existing, f, indent=2)
        pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

        time.sleep(DELAY_BETWEEN_REQS)

    # Final save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'─'*60}")
    print(f"  {done} generated  |  {skipped} skipped  |  {failed} failed")
    print(f"  JSON -> {OUTPUT_JSON}   (loaded by clip_engine.py)")
    print(f"  CSV  -> {OUTPUT_CSV}    (human-readable)")
    print(f"{'─'*60}")
    print(f"\nNext step — rebuild CLIP index:")
    print(f"  curl -X POST http://localhost:8000/api/index/rebuild")


if __name__ == "__main__":
    main()  