#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=4.0.0", "Pillow>=10.0.0"]
# ///

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


def find_pdfs(paths: Iterable[Path], recursive: bool) -> List[Path]:
    pdfs: List[Path] = []
    for p in paths:
        if p.is_dir():
            globber = p.rglob if recursive else p.glob
            pdfs.extend(sorted(globber("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
    return pdfs


def extract_pdf_images(
    pdf_path: Path,
    output_root: Optional[Path],
    fmt: str,
    overwrite: bool,
) -> None:
    fmt = fmt.lower()
    if fmt == "jpg":
        fmt = "jpeg"

    reader = PdfReader(str(pdf_path))

    # Default output directory: "<pdf_stem>" (same name as PDF without extension)
    if output_root is None:
        out_dir = pdf_path.parent / pdf_path.stem
    else:
        # Put each PDF in its own subdir under output_root
        out_dir = output_root / pdf_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for page_index, page in enumerate(reader.pages, start=1):
        images = getattr(page, "images", [])
        for image_index, image_file in enumerate(images, start=1):
            pil_img = image_file.image

            if fmt in ("jpeg", "jpg") and pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            filename = f"{pdf_path.stem}_p{page_index:04d}_i{image_index:02d}.{fmt}"
            out_path = out_dir / filename

            if out_path.exists() and not overwrite:
                continue

            pil_img.save(out_path, format=fmt.upper())
            total += 1

    print(f"{pdf_path}: wrote {total} images to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split PDFs (assembled from JPEGs) back into per-page image files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input PDF files and/or directories containing PDFs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. "
            "Default: a folder named after each PDF (without extension) next to the PDF."
        ),
    )
    parser.add_argument(
        "--fmt",
        choices=["jpeg", "jpg", "png"],
        default="jpeg",
        help="Output image format (default: jpeg).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when inputs include directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    pdfs = find_pdfs(input_paths, recursive=args.recursive)

    if not pdfs:
        print("No PDFs found in the given inputs.")
        return

    for pdf_path in pdfs:
        extract_pdf_images(
            pdf_path=pdf_path,
            output_root=args.output_dir,
            fmt=args.fmt,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
