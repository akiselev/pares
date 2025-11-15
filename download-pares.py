#!/usr/bin/env -S uv run

# /// script
# dependencies = [
#   "playwright>=1.47.0",
#   "pillow>=10.0.0",
# ]
# ///

import asyncio
import argparse
import logging
import re
import urllib.parse
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set

from PIL import Image  # type: ignore
from playwright.async_api import (  # type: ignore
    APIRequestContext,
    Error as PlaywrightError,
    async_playwright,
)

logger = logging.getLogger("pares_downloader")


BASE_URL = "https://pares.mcu.es/ParesBusquedas20"


def slugify(text: str) -> str:
    """
    Make a filesystem-safe name from arbitrary text.
    """
    text = text.strip()
    # Replace path separators and illegal filename characters
    text = re.sub(r"[\\\/|:*?\"<>]", "_", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove trailing dots/spaces (Windows-safe)
    text = text.strip(" .")
    # Limit length
    return text[:180] or "document"


def existing_pdf_for_description(collection_dir: Path, desc_id: str) -> Optional[Path]:
    """
    Return the path to an already-downloaded PDF for the given description id,
    if it exists inside collection_dir. Filenames produced by this script
    always start with '[{desc_id}] '.
    """
    if not collection_dir.exists():
        return None

    prefix = f"[{desc_id}] "
    for candidate in collection_dir.glob("*.pdf"):
        if candidate.name.startswith(prefix):
            return candidate
    return None


def is_url(text: str) -> bool:
    """
    Heuristically determine whether the input looks like an HTTP(S) URL.
    """
    parsed = urllib.parse.urlparse(text)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def classify_pares_url(url: str) -> str:
    """
    Determine whether a PARES URL points to a description page or a search results page.
    Returns one of: 'description', 'search_results', or 'unknown'.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()

    description_markers = (
        "/catalogo/description/",
        "/catalogo/show/",
    )
    search_markers = (
        "/catalogo/find",
        "/catalogo/contiene/",
    )

    if any(marker in path for marker in description_markers):
        return "description"
    if any(marker in path for marker in search_markers):
        return "search_results"
    return "unknown"


def extract_description_id_from_url(url: str) -> str:
    """
    Given a PARES URL that points at a specific description, extract its numeric id.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Not a valid HTTP(S) URL: {url}")

    patterns = [
        r"/(?:description|show)/(\d+)",
        r"/contiene/(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, parsed.path)
        if match:
            return match.group(1)

    match = re.search(r"/(\d+)(?:/)?$", parsed.path)
    if match:
        return match.group(1)

    for values in urllib.parse.parse_qs(parsed.query).values():
        for value in values:
            if value.isdigit():
                return value

    raise ValueError(
        f"Could not find a numeric description id inside the URL: {url}"
    )


async def find_description_id_from_reference(page, reference_code: str) -> str:
    """
    Resolve a PARES reference code (e.g. 'ES.41091.AGI//PATRONATO,30')
    to an internal numeric description id by calling the catalog 'find'
    endpoint with the reference number (signatura).

    If the input is already a numeric id, it is returned unchanged.
    """
    if reference_code.isdigit():
        return reference_code

    # Use the part after the // as the reference number tail, e.g.:
    #   ES.41091.AGI//PATRONATO,30     -> PATRONATO,30
    #   ES.41091.AGI//PATRONATO,30,R.1 -> PATRONATO,30,R.1
    tail = reference_code.split("//")[-1].strip() or reference_code.strip()

    # Build direct URL to the search results:
    # The advanced search form you dumped posts to action="find" with method="get",
    # so /catalogo/find?signatura=...&signaturaCompleta=1 is the results page.
    params = {
        "nm": "",
        "signatura": tail,
        "signaturaCompleta": "1",  # treat as full reference number
    }
    url = f"{BASE_URL}/catalogo/find?{urllib.parse.urlencode(params)}"

    await page.goto(url, wait_until="networkidle", timeout=120000)

    # Look for the standard results table:
    table = page.locator("table#resultados")
    if not await table.count():
        raise RuntimeError(
            f"No results table found after searching for reference number '{tail}'. "
            f"URL was: {url}"
        )

    rows = table.locator("tbody tr")
    n_rows = await rows.count()
    target_id = None

    # Prefer an exact match on the reference number text in <p class="signatura">
    # Example inner HTML:
    #   <p class="signatura"><em class="destacado">Reference number: </em>PATRONATO,31,R.2</p>
    exact_pattern = re.compile(
        rf"Reference number:\s*{re.escape(tail)}\s*$"
    )

    for i in range(n_rows):
        row = rows.nth(i)
        signatura_loc = row.locator("p.signatura")
        if not await signatura_loc.count():
            continue

        signatura_text = (await signatura_loc.first.inner_text()).strip()

        # First try exact match "Reference number: PATRONATO,30"
        if not exact_pattern.search(signatura_text):
            # Fallback: loose match if HTML is slightly different
            if tail not in signatura_text:
                continue

        # Grab the description link in this row
        link = row.locator("p.titulo a[href*='/catalogo/description/']").first
        href = await link.get_attribute("href")
        if not href:
            continue

        m = re.search(r"/description/(\d+)", href)
        if m:
            target_id = m.group(1)
            break

    if not target_id:
        raise RuntimeError(
            f"Could not resolve description id for reference code '{reference_code}' "
            f"(tail '{tail}') from the search results at {url}."
        )

    return target_id


async def extract_description_ids_from_results_page(page, results_url: str) -> List[str]:
    """
    Given a PARES search results URL, return all description ids present in the table.
    """
    await page.goto(results_url, wait_until="networkidle", timeout=120000)

    table = page.locator("table#resultados")
    if not await table.count():
        raise RuntimeError(
            f"No results table found at '{results_url}'. "
            "Ensure the URL points to a catalog search results page."
        )

    rows = table.locator("tbody tr")
    n_rows = await rows.count()
    found_ids: List[str] = []

    for i in range(n_rows):
        row = rows.nth(i)
        link = row.locator("p.titulo a[href*='/catalogo/description/']").first
        href = await link.get_attribute("href")
        if not href:
            continue
        match = re.search(r"/description/(\d+)", href)
        if match:
            found_ids.append(match.group(1))

    if not found_ids:
        logger.warning(
            "Search results page at '%s' did not contain any description links.", results_url
        )

    return found_ids


async def get_children_description_ids(page, desc_id: str) -> List[str]:
    """
    For a given description id, return a list of direct child description ids
    by following the 'Contains' link (if present).
    """
    await page.goto(f"{BASE_URL}/catalogo/description/{desc_id}", wait_until="networkidle")

    # Prefer the explicit 'Contains:' / 'Contiene:' info block
    contains_block = page.locator(
        "div.info:has(h4.aviso:has-text('Contains'))"
    )
    link = contains_block.locator("a[href*='/catalogo/contiene/']").first

    # Fallback: any /catalogo/contiene/ link anywhere on the page
    if not await link.count():
        link = page.locator("a[href*='/catalogo/contiene/']").first

    if not await link.count():
        # No children for this node
        print(f"  No 'Contains' link found for {desc_id}")
        return []

    href = await link.get_attribute("href")
    if not href:
        print(f"  'Contains' link for {desc_id} has no href")
        return []

    children_url = urllib.parse.urljoin(BASE_URL, href)
    await page.goto(children_url, wait_until="networkidle")

    table = page.locator("table#resultados")
    if not await table.count():
        print(
            f"  No results table found in children page for {desc_id} ({children_url})")
        return []

    rows = table.locator("tbody tr")
    n_rows = await rows.count()
    children: List[str] = []

    for i in range(n_rows):
        row = rows.nth(i)
        link = row.locator("p.titulo a[href*='/catalogo/description/']").first
        href = await link.get_attribute("href")
        if not href:
            continue
        m = re.search(r"/description/(\d+)", href)
        if m:
            children.append(m.group(1))

    print(
        f"  Found {len(children)} direct children for {desc_id}: {', '.join(children) if children else '(none)'}")
    return children


async def gather_all_description_ids(page, root_desc_id: str) -> List[str]:
    """
    Recursively walk 'Contains' relationships to collect all descendant
    description ids, including the root.
    """
    seen: Set[str] = set()
    order: List[str] = []

    async def dfs(desc_id: str) -> None:
        if desc_id in seen:
            return
        seen.add(desc_id)
        order.append(desc_id)
        children = await get_children_description_ids(page, desc_id)
        for child in children:
            await dfs(child)

    await dfs(root_desc_id)
    return order


async def fetch_with_retries(
    request: APIRequestContext,
    url: str,
    *,
    label: Optional[str] = None,
    max_attempts: int = 5,
    base_delay: float = 1.5,
) -> Optional[bytes]:
    """
    Fetch the given URL using Playwright's APIRequestContext with simple
    exponential backoff retries. Returns the body bytes on success, or None
    if all attempts failed.
    """
    target = label or url
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await request.get(url)
        except PlaywrightError as exc:  # type: ignore[catching-anything]
            last_error = exc
            logger.warning(
                "Request for %s failed on attempt %d/%d: %s",
                target,
                attempt,
                max_attempts,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Unexpected error fetching %s on attempt %d/%d: %s",
                target,
                attempt,
                max_attempts,
                exc,
            )
        else:
            try:
                if response.status == 200:
                    body = await response.body()
                else:
                    body = None
                    logger.warning(
                        "Request for %s returned HTTP %s on attempt %d/%d",
                        target,
                        response.status,
                        attempt,
                        max_attempts,
                    )
            finally:
                await response.dispose()

            if body is not None:
                return body

        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1))
            logger.info(
                "Retrying %s in %.1f seconds (attempt %d/%d)",
                target,
                delay,
                attempt + 1,
                max_attempts,
            )
            await asyncio.sleep(delay)

    if last_error:
        logger.error(
            "Failed to fetch %s after %d attempts: %s", target, max_attempts, last_error
        )
    return None


async def fetch_metadata_for_description(context, desc_id: str) -> dict:
    """
    Return basic metadata (title, reference number, reference code) for a description.
    """
    page = await context.new_page()
    try:
        await page.goto(f"{BASE_URL}/catalogo/description/{desc_id}", wait_until="networkidle")
        title = (await page.locator("#tituloUD").inner_text()).strip()

        ref_locator = page.locator(
            "div.info:has(h4.aviso:has-text('Reference number:')) p"
        )
        ref_number = ""
        if await ref_locator.count():
            ref_number = (await ref_locator.first.inner_text()).strip()

        ref_code_locator = page.locator(
            "div.info:has(h4.aviso:has-text('Reference code:')) p"
        )
        ref_code = ""
        if await ref_code_locator.count():
            ref_code = (await ref_code_locator.first.inner_text()).strip()

        return {
            "title": title,
            "reference_number": ref_number,
            "reference_code": ref_code,
        }
    finally:
        await page.close()


async def download_document_as_pdf(context, desc_id: str, collection_dir: Path) -> None:
    """
    For a given description id, open its image viewer, download all pages as JPEGs,
    and assemble them into a single PDF inside collection_dir.
    """
    meta = await fetch_metadata_for_description(context, desc_id)
    title = meta.get("title") or f"Description {desc_id}"
    ref_number = meta.get("reference_number") or f"id_{desc_id}"
    safe_name = slugify(f"[{desc_id}] {title}")
    pdf_path = collection_dir / f"{safe_name}.pdf"

    if pdf_path.exists():
        logger.info(
            f"Skipping {desc_id} (PDF already exists: {pdf_path.name})")
        return

    page = await context.new_page()
    try:
        viewer_url = f"{BASE_URL}/catalogo/show/{desc_id}"
        await page.goto(viewer_url, wait_until="networkidle")

        # Check that the viewer exists (document is digitized)
        if not await page.locator("#viewer").count():
            logger.info(f"No image viewer for {desc_id}; skipping.")
            return

        total_loc = page.locator("input#txt_totalImagenes")
        dbcode_loc = page.locator("input#dbCode").first
        brillo_loc = page.locator("input#txt_brillo").first
        contrast_loc = page.locator("input#txt_contrast").first

        if not await total_loc.count():
            logger.warning(
                f"Could not find total image count for {desc_id}; skipping.")
            return

        total_str = await total_loc.get_attribute("value")
        db_code = await dbcode_loc.get_attribute("value") if await dbcode_loc.count() else None
        brillo = await brillo_loc.get_attribute("value") if await brillo_loc.count() else "10.0"
        contrast = (
            await contrast_loc.get_attribute("value") if await contrast_loc.count() else "1.0"
        )

        if not total_str or not db_code:
            logger.warning(
                f"Missing total image count or dbCode for {desc_id}; skipping.")
            return

        total = int(total_str)
        logger.info(
            f"Downloading {total} pages for {desc_id} -> {pdf_path.name}")

        request = context.request
        images: List[bytes] = []

        for i in range(1, total + 1):
            params = {
                "accion": "42",
                "txt_descarga": "1",
                "dbCode": db_code,
                "txt_id_imagen": str(i),
                "txt_zoom": "10",
                "txt_contraste": "0",
                "txt_polarizado": "",
                "txt_brillo": str(brillo),
                "txt_contrast": str(contrast),
                "txt_totalImagenes": str(total),
                "txt_transformacion": "-1",
            }
            url = f"{BASE_URL}/ViewImage.do?{urllib.parse.urlencode(params)}"
            data = await fetch_with_retries(
                request, url, label=f"{desc_id} image {i}"
            )
            if data is None:
                logger.warning(
                    "Skipping image %d for %s after repeated request failures.",
                    i,
                    desc_id,
                )
                continue
            images.append(data)
            # Periodic progress updates (about 10 steps)
            step = max(1, total // 10)
            if i % step == 0 or i == total:
                logger.info(f"Downloaded {i}/{total} images for {desc_id}")

        if not images:
            logger.warning(
                f"No images downloaded for {desc_id}; skipping PDF creation.")
            return

        pil_images = []
        for idx, data in enumerate(images, start=1):
            try:
                im = Image.open(BytesIO(data))  # type: ignore[name-defined]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error opening image {idx} for {desc_id}: {e}")
                continue
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            pil_images.append(im)

        if not pil_images:
            logger.warning(
                f"All downloaded images for {desc_id} failed to open; skipping.")
            return

        first, rest = pil_images[0], pil_images[1:]
        collection_dir.mkdir(parents=True, exist_ok=True)
        first.save(pdf_path, "PDF", save_all=True, append_images=rest)
        logger.info(f"Saved PDF: {pdf_path}")
    finally:
        await page.close()


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PARES documents (and their hierarchical children) "
            "as per-document PDFs using Playwright."
        )
    )
    parser.add_argument(
        "reference",
        help=(
            "PARES reference code (e.g. 'ES.41091.AGI//PATRONATO,31'), "
            "numeric description id (e.g. '122096'), a document metadata URL, "
            "or a catalog search results URL."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("pares_downloads"),
        help="Base output directory (default: ./pares_downloads)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only download the specified description, not its children.",
    )
    parser.add_argument(
        "--browser",
        choices=["firefox", "chromium", "webkit"],
        default="chromium",
        help="Playwright browser engine to use (default: firefox).",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Starting PARES download")

    collection_dir = args.output_dir / slugify(args.reference)

    async with async_playwright() as p:
        browser_type = getattr(p, args.browser)
        browser = await browser_type.launch(headless=True)
        context = await browser.new_context()
        nav_page = await context.new_page()

        try:
            reference_input = args.reference.strip()
            root_candidates: List[str] = []

            if is_url(reference_input):
                url_type = classify_pares_url(reference_input)
                if url_type == "search_results":
                    logger.info(
                        "Detected search results URL; extracting description ids..."
                    )
                    root_candidates = await extract_description_ids_from_results_page(
                        nav_page, reference_input
                    )
                    if not root_candidates:
                        raise RuntimeError(
                            "No descriptions were found in the provided search results URL."
                        )
                    logger.info(
                        "Found %d description(s) in search results.",
                        len(root_candidates),
                    )
                else:
                    root_desc_id = extract_description_id_from_url(
                        reference_input)
                    root_candidates = [root_desc_id]
                    logger.info(
                        "Using description id %s extracted from provided URL.",
                        root_desc_id,
                    )
            else:
                logger.info(
                    f"Resolving reference '{reference_input}' to description id...")
                root_desc_id = await find_description_id_from_reference(nav_page, reference_input)
                logger.info(
                    f"Resolved reference '{reference_input}' -> description id {root_desc_id}"
                )
                root_candidates = [root_desc_id]

            seen: Set[str] = set()
            all_ids: List[str] = []
            for root_id in root_candidates:
                if args.no_recursive:
                    new_ids = [root_id]
                else:
                    new_ids = await gather_all_description_ids(nav_page, root_id)
                for desc_id in new_ids:
                    if desc_id not in seen:
                        seen.add(desc_id)
                        all_ids.append(desc_id)
            if not all_ids:
                raise RuntimeError(
                    "No description ids to process after resolving input.")

            logger.info(
                f"Will process {len(all_ids)} description(s): {', '.join(all_ids)}")

            await nav_page.close()

            for idx, desc_id in enumerate(all_ids, start=1):
                logger.info(
                    f"[{idx}/{len(all_ids)}] Processing description {desc_id}")

                existing_pdf = existing_pdf_for_description(
                    collection_dir, desc_id)
                if existing_pdf:
                    logger.info(
                        "Skipping %s (PDF already exists: %s)",
                        desc_id,
                        existing_pdf.name,
                    )
                    continue

                try:
                    await download_document_as_pdf(context, desc_id, collection_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to download description %s: %s",
                        desc_id,
                        exc,
                        exc_info=True,
                    )

        finally:
            await context.close()
            await browser.close()
            logger.info("Finished PARES download")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
