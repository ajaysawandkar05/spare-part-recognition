"""
scripts/evaluate.py
-------------------
Evaluates retrieval quality using the 10 multi-image materials.

For each material that has image_2.png (and optionally image_3.png):
  - image_1.png is assumed already indexed
  - image_2.png / image_3.png are used as query images
  - Reports Recall@1, Recall@3, Recall@5

Directory structure expected:
    data/hwl_multi_images/
        59-1068431-00529/
            image_1.png   ← in main index (same as xlsx image)
            image_2.png   ← query (different angle)
            image_3.png   ← query (optional)
        ...

Usage:
    python scripts/evaluate.py \
        --multi_dir data/hwl_multi_images \
        --dataset   data/hwl_dataset.json \
        --index     data/hwl_clip_index.npz \
        --top_k     5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Loading {CLIP_MODEL} on {device} ...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()
    return model, processor, device


def encode_image(model, processor, device, path: str) -> np.ndarray:
    img    = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()


def evaluate(
    multi_dir: str,
    dataset_json: str,
    index_path: str,
    top_k: int = 5,
) -> None:
    # Load index
    data          = np.load(index_path, allow_pickle=True)
    embeddings    = data["embeddings"]       # (N, 512)
    material_ids  = data["material_ids"].tolist()

    model, processor, device = load_model()

    results_log = []
    correct_at  = {1: 0, 3: 0, top_k: 0}
    total       = 0

    print(f"\n[eval] Scanning {multi_dir} ...\n")

    for mat_id in sorted(os.listdir(multi_dir)):
        mat_path = Path(multi_dir) / mat_id
        if not mat_path.is_dir():
            continue

        query_images = sorted([
            str(mat_path / f)
            for f in os.listdir(mat_path)
            if f.startswith("image_") and f != "image_1.png"
        ])

        if not query_images:
            continue

        for q_img in query_images:
            q_emb  = encode_image(model, processor, device, q_img)
            scores = embeddings @ q_emb
            top_idx = np.argsort(-scores)

            ranked_ids = [material_ids[i] for i in top_idx[:max(top_k, 5)]]
            rank_of_correct = None
            for rank_pos, rid in enumerate(ranked_ids, start=1):
                if rid == mat_id:
                    rank_of_correct = rank_pos
                    break

            total += 1
            for k in correct_at:
                if rank_of_correct is not None and rank_of_correct <= k:
                    correct_at[k] += 1

            hit  = rank_of_correct == 1
            icon = "✅" if hit else f"❌ (rank {rank_of_correct or '>10'})"
            print(f"  {mat_id:<28}  {Path(q_img).name}  →  {icon}")
            results_log.append({
                "material_id":     mat_id,
                "query_image":     Path(q_img).name,
                "correct_rank":    rank_of_correct,
                "top1_hit":        hit,
                "top_k_hit":       (rank_of_correct is not None and rank_of_correct <= top_k),
                "top1_predicted":  ranked_ids[0],
            })

    if total == 0:
        print(f"[eval] ⚠  No query images found in {multi_dir}")
        print("         Place materials with 2+ images there (image_1.png = index, image_2.png = query).")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  EVALUATION RESULTS  ({total} queries across {len(set(r['material_id'] for r in results_log))} materials)")
    print(f"{'─'*60}")
    for k in sorted(correct_at):
        pct = correct_at[k] / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  Recall@{k:<2}  {bar}  {correct_at[k]}/{total}  ({pct:.1f}%)")
    print(f"{'─'*60}\n")

    # Save results
    out_path = Path(index_path).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_queries": total,
            "recall": {f"@{k}": round(correct_at[k] / total, 4) for k in correct_at},
            "per_query": results_log,
        }, f, indent=2)
    print(f"[eval] Full results saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HWL retrieval on multi-image test set")
    parser.add_argument("--multi_dir", default="data/hwl_multi_images")
    parser.add_argument("--dataset",   default="data/hwl_dataset.json")
    parser.add_argument("--index",     default="data/hwl_clip_index.npz")
    parser.add_argument("--top_k",     type=int, default=5)
    args = parser.parse_args()
    evaluate(args.multi_dir, args.dataset, args.index, args.top_k)


if __name__ == "__main__":
    main()