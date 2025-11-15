Just a small set of scripts to improve the workflow for scraping Spanish archive documents from PARES and prepare them for LLMs. Vibe coded so don't expect any miracles.

These can be run directly if you have UV installed.

# Scripts

## download-pares

# PARES PDF Downloader

A small command-line tool for downloading digitized documents from the Spanish archives portal [PARES](https://pares.mcu.es) as per-description PDFs.

Given a PARES reference code, description ID, or catalog URL, the script:

* Resolves it to one or more internal description IDs
* Recursively walks “Contains / Contiene” relationships (unless disabled)
* Downloads all available page images via PARES’ viewer
* Assembles them into a single multi-page PDF per description

---

## Features

* Accepts multiple forms of input:

  * Reference codes, e.g. `ES.41091.AGI//PATRONATO,31`
  * Numeric description IDs, e.g. `122096`
  * Description URLs, e.g. `https://pares.mcu.es/ParesBusquedas20/catalogo/description/122096`
  * Search result URLs, e.g. `https://pares.mcu.es/ParesBusquedas20/catalogo/find?...`
* Automatically resolves reference codes to internal description IDs
* Recursively traverses child descriptions via “Contains / Contiene”
* Uses Playwright for robust navigation and downloading
* Retries image downloads with exponential backoff
* Converts all downloaded JPEGs into a single PDF per description
* Idempotent and resume-friendly:

  * Skips PDFs that already exist
  * Skips descriptions with no digitized viewer
* Filenames are slugified and Windows-safe:

  * `"[{desc_id}] {title}.pdf"`

---

## Requirements

```bash
uvx run playwright install chromium # or whatever  browser you  will use
```

---

## Installation

1. Save the script as `pares_downloader.py` (or similar) and make it executable:

```bash
chmod +x pares_downloader.py
```

2. Ensure `uv` is installed:

3. Install Playwright browsers (once):

```bash
uvx run playwright install chromium firefox webkit
```

No additional `pip install` step is needed; dependencies are declared in the script header and automatically resolved by `uv run`.

---

## Command-line usage

Basic syntax:

```bash
./pares_downloader.py REFERENCE_OR_URL [options]
```

### Positional argument

* `reference`
  One of:

  * PARES reference code, e.g. `ES.41091.AGI//PATRONATO,31`
  * Numeric description ID, e.g. `122096`
  * Description metadata URL
  * Catalog search-results URL

### Options

* `-o, --output-dir PATH`
  Base output directory.
  Default: `./pares_downloads`

* `--no-recursive`
  Only download the given description(s) and do not traverse “Contains” children.

* `--browser {firefox,chromium,webkit}`
  Playwright browser engine to use.
  Default: `chromium`.

---

## Output layout

For an input reference (or URL) like:

```bash
./pares_downloader.py "ES.41091.AGI//PATRONATO,31"
```

The script:

1. Slugifies the input reference and creates a subdirectory inside the output dir:

```text
pares_downloads/
  ES.41091.AGI__PATRONATO,31/
    [122091] Some description title.pdf
    [122092] Child description.pdf
    ...
```

2. For each description ID it processes:

* Fetches metadata (title, reference number, reference code)
* Builds a filename: `"[{desc_id}] {title}.pdf"` (slugified)
* Downloads all viewer pages as JPEGs
* Converts to a single multi-page PDF

3. If a PDF already exists for a given description ID (based on filename prefix `[{desc_id}] `), it is skipped.

---

## Examples

### 1. Download a single reference and all its descendants

```bash
./pares_downloader.py "ES.41091.AGI//PATRONATO,31"
```

* Resolves the reference code to its root description ID
* Recursively gathers all “Contains” descriptions
* Creates one PDF per description in:

```text
./pares_downloads/ES.41091.AGI__PATRONATO,31/
```

### 2. Download only a single description (no recursion)

```bash
./pares_downloader.py "ES.41091.AGI//PATRONATO,31,R.3" --no-recursive
```

Downloads only that specific description, if digitized.

### 3. Use a numeric description ID directly

```bash
./pares_downloader.py 122096
```

This bypasses reference resolution and starts traversal from `122096`.

### 4. Use a catalog description URL

```bash
./pares_downloader.py "https://pares.mcu.es/ParesBusquedas20/catalogo/description/122096"
```

The script extracts the ID (`122096`) and proceeds as above.

### 5. Use a search results URL

```bash
./pares_downloader.py "https://pares.mcu.es/ParesBusquedas20/catalogo/find?signatura=PATRONATO,31,R.2&signaturaCompleta=1"
```

* Parses all description links in the results table
* Treats each description as a root
* Traverses children (unless `--no-recursive` is used)
* Produces multiple PDFs in one run

### 6. Custom output directory and browser

```bash
./pares_downloader.py "ES.41091.AGI//PATRONATO,31" \
  --output-dir /path/to/my_archive \
  --browser firefox
```

Output will be under:

```text
/path/to/my_archive/ES.41091.AGI__PATRONATO,31/
```

## Known limitations

* Only works with documents that have an image viewer in PARES (`#viewer` element). If a description is not digitized (e.g., microfilm only), it is skipped.
* Assumes PARES HTML structure and field names remain broadly stable.
* Very large hierarchies or documents with hundreds of pages may take significant time and disk space.
