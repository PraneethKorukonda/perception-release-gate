"""Find scenarios the labels don't describe, with CLIP text search.

BDD100K tells you time of day and weather. It doesn't tell you "a pedestrian
stepping out between parked cars" or "glare from low sun", and those are
exactly the cases a release review wants to see. CLIP puts images and text
in the same embedding space, so a sentence can be used as a query over
thousands of frames.

How I use it:
  1. Embed every 5th frame (1 per second) once, and save the vectors.
  2. Check the miner on something the labels *do* record. For example,
     "a city street at night" should mostly return frames labelled night.
     precision_at_k gives a number for that. If it can't find night, I
     don't trust it to find anything subtler.
  3. Search, look at the top results, keep the set, version it
     (scenarios.py).

Step 3 still needs a person looking at images. CLIP returns plenty of near
misses, and a mined set should be reviewed before it gates a release.

torch and transformers are only imported when you actually embed, so the
rest of the package works without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def images(self, paths, batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                v = self.model.get_image_features(**inputs)
            out.append(v.float().cpu().numpy())
        return normalize(np.concatenate(out))

    def text(self, queries: list[str]) -> np.ndarray:
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            v = self.model.get_text_features(**inputs)
        return normalize(v.float().cpu().numpy())


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def save_embeddings(path, image_ids, vectors) -> None:
    np.savez_compressed(path, image_ids=np.asarray(image_ids, dtype=str), vectors=vectors)


def load_embeddings(path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["image_ids"], z["vectors"]


def search(image_ids, vectors, query_vector, k: int = 200) -> pd.DataFrame:
    sim = vectors @ np.asarray(query_vector).ravel()
    top = np.argsort(-sim)[:k]
    return pd.DataFrame(
        {
            "image_id": np.asarray(image_ids)[top],
            "similarity": sim[top],
            "rank": np.arange(1, len(top) + 1),
        }
    )


def precision_at_k(ranked_ids, positive_ids, ks=(10, 50, 100, 200)) -> dict:
    """Share of the top-k results that the labels say are positives."""
    pos = set(positive_ids)
    hits = np.array([i in pos for i in ranked_ids], float)
    return {f"p@{k}": float(hits[:k].mean()) for k in ks if k <= len(hits)}


def frame_paths(images: pd.DataFrame, image_root: str | Path, stride: int = 5) -> pd.DataFrame:
    """Every `stride`-th frame of each clip, with its path on disk."""
    sub = images[images["frame_index"] % stride == 0][["image_id"]].copy()
    sub["path"] = [str(Path(image_root) / i) for i in sub["image_id"]]
    return sub.reset_index(drop=True)
