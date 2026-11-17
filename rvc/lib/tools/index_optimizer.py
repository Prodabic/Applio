"""Experimental non-destructive FAISS index compression for Applio.

Compresses .index feature vectors only; exported voice .pth files are unchanged.
Use trusted index files only. Numerical checks do NOT establish audio equivalence.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np

METHODS = {"fp16": "SQfp16", "sq8": "SQ8"}


def human_bytes(size: int) -> str:
    size = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def _validate_source(index):
    index = faiss.downcast_index(index)
    if not isinstance(index, (faiss.IndexFlatL2, faiss.IndexIVFFlat)):
        raise ValueError("Only original Applio FlatL2 / IVFFlat indexes are supported")
    if index.metric_type != faiss.METRIC_L2 or not index.is_trained:
        raise ValueError("Source must be a trained L2 index")
    if index.ntotal < 8:
        raise ValueError("At least 8 feature vectors are needed for Applio retrieval")
    if isinstance(index, faiss.IndexIVFFlat):
        # Generated IVFFlat indexes usually lack the direct map required for
        # reconstruct_n(). This changes the in-memory copy, not the source.
        faiss.extract_index_ivf(index).make_direct_map()
    return index


def _read(index, start: int, count: int):
    vectors = np.ascontiguousarray(
        index.reconstruct_n(start, count), dtype=np.float32
    )
    if vectors.shape != (count, index.d) or not np.isfinite(vectors).all():
        raise ValueError("Invalid or non-finite reconstructed features")
    return vectors


def _sample(index, max_rows: int, chunk_size: int):
    stride = max(1, (index.ntotal + max_rows - 1) // max_rows)
    blocks = []
    total = 0
    for start in range(0, index.ntotal, chunk_size):
        count = min(chunk_size, index.ntotal - start)
        chunk = _read(index, start, count)
        subset = chunk[(-start) % stride :: stride]
        if subset.size:
            blocks.append(subset.copy())
            total += len(subset)
        if total >= max_rows:
            break
    return np.ascontiguousarray(
        np.concatenate(blocks, axis=0)[:max_rows], dtype=np.float32
    )


def _make_target(source, method: str, chunk_size: int, train_rows: int):
    if method not in METHODS:
        raise ValueError(f"Unknown compression method: {method}")
    if isinstance(source, faiss.IndexIVFFlat):
        src = faiss.extract_index_ivf(source)
        target = faiss.downcast_index(faiss.index_factory(
            source.d, f"IVF{src.nlist},{METHODS[method]}", faiss.METRIC_L2
        ))
        dst = faiss.extract_index_ivf(target)
        centroids = np.ascontiguousarray(
            src.quantizer.reconstruct_n(0, src.nlist), dtype=np.float32
        )
        dst.quantizer.add(centroids)
        dst.quantizer_trains_alone = 1
        dst.nprobe = src.nprobe
        dst.max_codes = src.max_codes
        target.by_residual = False
    else:
        target = faiss.downcast_index(faiss.index_factory(
            source.d, METHODS[method], faiss.METRIC_L2
        ))
    if not target.is_trained:
        count = train_rows if method == "sq8" else min(1024, train_rows)
        target.train(_sample(source, count, chunk_size))
    if not target.is_trained:
        raise RuntimeError("FAISS quantizer training failed")
    if isinstance(source, faiss.IndexIVFFlat):
        # The original partitions must be kept for consistent IVF search.
        dst.quantizer.reset()
        dst.quantizer.add(centroids)
        if dst.quantizer.ntotal != dst.nlist:
            raise RuntimeError("Failed to restore the original IVF centroids")
    return target


def _quality(source, result, n_rows: int):
    ids = np.unique(np.linspace(
        0, source.ntotal - 1, min(n_rows, source.ntotal), dtype=np.int64
    ))
    a = np.stack([_read(source, int(i), 1)[0] for i in ids])
    b = np.stack([_read(result, int(i), 1)[0] for i in ids])
    rmse = float(np.sqrt(np.mean(np.square(
        a.astype(np.float64) - b.astype(np.float64)
    ))))
    relative = np.linalg.norm(a - b, axis=1) / (
        np.linalg.norm(a, axis=1) + 1e-8
    )
    probes = np.ascontiguousarray(a[:16], dtype=np.float32)
    _, ia = source.search(probes, 8)
    distances, ib = result.search(probes, 8)
    if not np.isfinite(distances).all() or (ib < 0).any():
        raise ValueError("Optimized index did not return 8 valid neighbors")
    overlap = np.mean([
        len(set(x.tolist()) & set(y.tolist())) / 8
        for x, y in zip(ia, ib)
    ])
    return {
        "reconstruction_rmse": rmse,
        "mean_relative_l2": float(np.mean(relative)),
        "p95_relative_l2": float(np.percentile(relative, 95)),
        "mean_top8_id_overlap": float(overlap),
        "note": "Numerical/retrieval metrics are NOT an audio quality guarantee.",
    }


def optimize_file(
    source_path: Path, output_path: Path, method: str,
    chunk_size: int = 4096, train_rows: int = 10000, check_rows: int = 64
):
    """Write a smaller sibling index, preserving the source on disk."""
    source_path, output_path = Path(source_path), Path(output_path)
    if not source_path.is_file() or source_path.suffix.lower() != ".index":
        raise ValueError("Input must be an existing .index file")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if output_path.exists():
        return None
    if chunk_size < 8 or train_rows < 64 or check_rows < 8:
        raise ValueError("chunk >= 8, train_rows >= 64, check_rows >= 8 required")
    source = _validate_source(faiss.read_index(str(source_path)))
    target = _make_target(source, method, chunk_size, train_rows)
    for start in range(0, source.ntotal, chunk_size):
        target.add(_read(source, start, min(chunk_size, source.ntotal - start)))
    if target.ntotal != source.ntotal:
        raise RuntimeError("Vector count mismatch after compression")
    if isinstance(target, faiss.IndexIVFScalarQuantizer):
        faiss.extract_index_ivf(target).make_direct_map()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Temporary output already exists: {partial}")
    try:
        faiss.write_index(target, str(partial))
        result = faiss.read_index(str(partial))
        if result.ntotal != source.ntotal or result.d != source.d:
            raise RuntimeError("Index failed serialization round-trip")
        if isinstance(result, faiss.IndexIVFScalarQuantizer):
            faiss.extract_index_ivf(result).make_direct_map()
        diagnostics = _quality(source, result, check_rows)
        old_size, new_size = source_path.stat().st_size, partial.stat().st_size
        if new_size >= old_size:
            return None
        if output_path.exists():
            raise FileExistsError(f"Output appeared during conversion: {output_path}")
        partial.rename(output_path)
        report = {
            "source": str(source_path), "output": str(output_path),
            "method": method, "source_bytes": old_size, "output_bytes": new_size,
            "saving_percent": round(100 * (1 - new_size / old_size), 3),
            "dimensions": source.d, "vectors": source.ntotal,
            "diagnostics": diagnostics,
        }
        output_path.with_name(output_path.name + ".report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        if partial.exists():
            partial.unlink()


def self_test():
    """Synthetic index tests; no voice recordings or models required."""
    rng = np.random.default_rng(2026)
    vectors = np.ascontiguousarray(
        rng.standard_normal((1600, 64)), dtype=np.float32
    )
    with tempfile.TemporaryDirectory(prefix="applio-index-test-") as folder:
        root = Path(folder)
        for layout in ("Flat", "IVF16,Flat"):
            source = faiss.downcast_index(faiss.index_factory(
                64, layout, faiss.METRIC_L2
            ))
            if not source.is_trained:
                source.train(vectors)
            source.add(vectors)
            filename = root / (layout.replace(",", "_") + ".index")
            faiss.write_index(source, str(filename))
            for mode in METHODS:
                out = root / f"{filename.stem}_{mode}.index"
                report = optimize_file(
                    filename, out, mode,
                    chunk_size=256, train_rows=1024, check_rows=32
                )
                if report is None or not out.is_file():
                    raise AssertionError(f"{layout} / {mode} did not shrink")
                print(f"PASS {layout} -> {mode}: "
                      f"{report['saving_percent']:.1f}% smaller; "
                      f"RMSE {report['diagnostics']['reconstruction_rmse']:.6g}")
    print("Synthetic self-test passed (audio quality not evaluated).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", help=".index file or directory")
    parser.add_argument("--mode", choices=("fp16", "sq8", "both"), default="fp16")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.path is None:
        parser.error("Provide an index file/directory or --self-test")
    files = (
        [args.path] if args.path.is_file() else sorted(
            p for p in args.path.rglob("*.index")
            if not p.stem.endswith(("_fp16", "_sq8"))
        )
    )
    failures = 0
    for source in files:
        for mode in (METHODS if args.mode == "both" else (args.mode,)):
            output = source.with_name(source.stem + f"_{mode}.index")
            try:
                report = optimize_file(source, output, mode)
                print(f"{source.name} {mode}: " + (
                    f"{report['saving_percent']:.1f}% smaller"
                    if report else "skipped"
                ))
            except Exception as exc:
                failures += 1
                print(f"ERROR {source}: {exc}", file=sys.stderr)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
