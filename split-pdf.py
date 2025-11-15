#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=5.0.0"]
# ///

import argparse
import math
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def parse_pages_spec(spec: str, total_pages: int) -> list[list[int]]:
    """
    Parse a pages spec like '1-5,3,4,6-9,9-' into a list of page lists.
    Pages are 1-based in the spec.
    """
    parts: list[list[int]] = []
    for segment in spec.split(","):
        seg = segment.strip()
        if not seg:
            continue

        if "-" in seg:
            start_str, end_str = seg.split("-", 1)
            if not start_str:
                raise ValueError(f"Invalid range '{seg}': start is missing")
            start = int(start_str)

            if end_str.strip() == "":
                end = total_pages
            else:
                end = int(end_str)

            if not (1 <= start <= total_pages):
                raise ValueError(
                    f"Start page {start} in '{seg}' is out of bounds (1-{total_pages})")
            if not (1 <= end <= total_pages):
                raise ValueError(
                    f"End page {end} in '{seg}' is out of bounds (1-{total_pages})")
            if start > end:
                raise ValueError(
                    f"Start page {start} is greater than end page {end} in '{seg}'")

            parts.append(list(range(start, end + 1)))
        else:
            page = int(seg)
            if not (1 <= page <= total_pages):
                raise ValueError(
                    f"Page {page} is out of bounds (1-{total_pages})")
            parts.append([page])

    if not parts:
        raise ValueError("No valid page specifications found")
    return parts


def split_by_pages(reader: PdfReader, parts_pages: list[list[int]], out_dir: Path, stem: str) -> None:
    for i, pages in enumerate(parts_pages, start=1):
        writer = PdfWriter()
        for p in pages:
            writer.add_page(reader.pages[p - 1])  # p is 1-based
        out_path = out_dir / f"{stem}_part{i}.pdf"
        with out_path.open("wb") as f:
            writer.write(f)
        print(f"Wrote {out_path}")


def split_by_pages_per(reader: PdfReader, pages_per: int, out_dir: Path, stem: str) -> None:
    if pages_per <= 0:
        raise ValueError("--pages-per must be a positive integer")

    total = len(reader.pages)
    part_index = 1
    current_start = 1

    while current_start <= total:
        current_end = min(current_start + pages_per - 1, total)
        writer = PdfWriter()
        for p in range(current_start, current_end + 1):
            writer.add_page(reader.pages[p - 1])
        out_path = out_dir / f"{stem}_part{part_index}.pdf"
        with out_path.open("wb") as f:
            writer.write(f)
        print(f"Wrote {out_path}")
        part_index += 1
        current_start = current_end + 1


def split_by_num_parts(reader: PdfReader, num_parts: int, out_dir: Path, stem: str) -> None:
    if num_parts <= 0:
        raise ValueError("--num-parts must be a positive integer")

    total = len(reader.pages)
    if num_parts > total:
        raise ValueError(
            f"--num-parts ({num_parts}) cannot exceed total pages ({total})")

    # Use equal-sized chunks as much as possible, last one may have fewer pages.
    base_size = math.ceil(total / num_parts)
    current_start = 1
    part_index = 1

    while current_start <= total and part_index <= num_parts:
        current_end = min(current_start + base_size - 1, total)
        writer = PdfWriter()
        for p in range(current_start, current_end + 1):
            writer.add_page(reader.pages[p - 1])
        out_path = out_dir / f"{stem}_part{part_index}.pdf"
        with out_path.open("wb") as f:
            writer.write(f)
        print(f"Wrote {out_path}")

        part_index += 1
        current_start = current_end + 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Split a PDF into multiple parts in different ways."
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the input PDF file.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pages",
        help="Comma-separated list of page ranges like '1-5,3,4,6-9,9-'. Each segment becomes its own output PDF.",
    )
    group.add_argument(
        "--pages-per",
        type=int,
        help="Number of pages per output PDF.",
    )
    group.add_argument(
        "--num-parts",
        type=int,
        help="Split the PDF into this many parts, as evenly as possible.",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input_pdf)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    print(f"Input PDF: {input_path} ({total_pages} pages)")

    # Output directory: sibling directory named after the PDF filename (without extension)
    out_dir = input_path.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    stem = input_path.stem

    if args.pages is not None:
        parts_pages = parse_pages_spec(args.pages, total_pages)
        split_by_pages(reader, parts_pages, out_dir, stem)
    elif args.pages_per is not None:
        split_by_pages_per(reader, args.pages_per, out_dir, stem)
    elif args.num_parts is not None:
        split_by_num_parts(reader, args.num_parts, out_dir, stem)
    else:
        raise SystemExit("No splitting mode selected")


if __name__ == "__main__":
    main()
