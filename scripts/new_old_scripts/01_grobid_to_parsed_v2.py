#!/usr/bin/env python3
"""
Script 1 (Revised): PDF → GROBID TEI → Parsed Sections JSON

Key Improvements:
1. Recursive subsection extraction (captures nested structure)
2. Footnote and table extraction (methods info often lives there)
3. Better sentence splitting (handles abbreviations)
4. Metadata extraction (authors, journal, year - useful for MCA supplementary variables)
5. Section-level statistics (word count, paragraph count) for quality checks
6. Robust error handling with detailed logging

Dependencies:
  pip install requests lxml

Usage:
  python 01_grobid_to_parsed_v2.py \
    --pdf_dir "./data/raw" \
    --tei_dir "./data/tei/raw" \
    --parsed_dir "./data/tei/parsed" \
    --grobid_url "http://localhost:8070"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from lxml import etree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Sentence splitter - we use a simpler approach that works reliably
# Instead of complex lookbehinds, we split and then merge false positives
SENT_END_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Abbreviations that shouldn't trigger sentence splits (used in post-processing)
ABBREVS = {'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'jr', 'sr', 
           'vs', 'etc', 'vol', 'pp', 'no', 'fig', 'eq', 'al', 'ed', 'eds',
           'e.g', 'i.e', 'cf', 'et', 'st', 'ave', 'blvd'}

# Major section break indicators (prevent container inheritance)
MAJOR_BREAK_PATTERNS = [
    r"\bdiscussion\b",
    r"\bconclud",  # conclusion, conclusions, concluding
    r"\breference",
    r"\bbibliograph",
    r"\bappendix\b",
    r"\bannex\b",
    r"\backnowledg",
]

# =============================================================================
# GROBID Client
# =============================================================================

def grobid_process_fulltext(
    pdf_path: Path,
    grobid_url: str,
    timeout_sec: int = 180,
    max_retries: int = 3,
    retry_sleep_sec: float = 3.0,
) -> str:
    """Send PDF to GROBID and return TEI XML."""
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    
    for attempt in range(1, max_retries + 1):
        try:
            with pdf_path.open("rb") as f:
                files = {"input": (pdf_path.name, f, "application/pdf")}
                data = {
                    "consolidateHeader": "1",  # Enable for better metadata
                    "consolidateCitations": "0",
                    "includeRawAffiliations": "1",
                    "teiCoordinates": "0",  # Disable for smaller output
                }
                resp = requests.post(
                    endpoint, 
                    files=files, 
                    data=data, 
                    timeout=timeout_sec
                )
            
            resp.raise_for_status()
            tei_xml = resp.text or ""
            
            if not tei_xml.strip().startswith("<"):
                raise RuntimeError(f"GROBID returned non-XML for {pdf_path.name}")
            
            return tei_xml
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}/{max_retries} for {pdf_path.name}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt}/{max_retries}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt}/{max_retries}: {e}")
        
        if attempt < max_retries:
            time.sleep(retry_sleep_sec * attempt)
    
    raise RuntimeError(f"GROBID failed after {max_retries} attempts for {pdf_path}")

# =============================================================================
# Text Utilities
# =============================================================================

def normalize_whitespace(s: str) -> str:
    """Collapse whitespace and strip."""
    return " ".join((s or "").split()).strip()


def normalize_header(header: str) -> str:
    """Normalize section header for matching."""
    h = (header or "").strip().lower()
    # Remove numbering: "1.", "1.1", "I.", "II)", "A."
    h = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", h)
    h = re.sub(r"^\s*[ivxlc]+[\.\)]\s*", "", h, flags=re.I)
    h = re.sub(r"^\s*[a-z][\.\)]\s*", "", h, flags=re.I)
    # Remove special chars, collapse whitespace
    h = re.sub(r"[^a-z0-9\s]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with abbreviation handling."""
    text = normalize_whitespace(text)
    if not text:
        return []
    
    # Initial split on sentence boundaries
    raw_sentences = SENT_END_PATTERN.split(text)
    
    # Merge back sentences that were incorrectly split on abbreviations
    sentences = []
    buffer = ""
    
    for part in raw_sentences:
        part = part.strip()
        if not part:
            continue
            
        if buffer:
            # Check if previous part ended with an abbreviation
            buffer_lower = buffer.lower()
            is_abbrev = any(buffer_lower.endswith(abbr + '.') for abbr in ABBREVS)
            
            if is_abbrev:
                # Merge with current part
                buffer = buffer + " " + part
            else:
                # Previous was a complete sentence
                sentences.append(buffer)
                buffer = part
        else:
            buffer = part
    
    if buffer:
        sentences.append(buffer)
    
    return sentences


def first_n_sentences(text: str, n: int = 3) -> str:
    """Extract first n sentences."""
    sents = split_sentences(text)
    return " ".join(sents[:n])


def last_n_sentences(text: str, n: int = 3) -> str:
    """Extract last n sentences."""
    sents = split_sentences(text)
    return " ".join(sents[-n:]) if sents else ""


def is_major_break(header_norm: str) -> bool:
    """Check if header indicates a major section boundary."""
    if not header_norm:
        return False
    for pattern in MAJOR_BREAK_PATTERNS:
        if re.search(pattern, header_norm, re.I):
            return True
    return False


def count_words(text: str) -> int:
    """Count words in text."""
    return len(normalize_whitespace(text).split())

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SectionBlock:
    """Represents a parsed section from the TEI document."""
    paper_id: str
    sec_index: int
    depth: int  # 0 = top-level, 1 = subsection, etc.
    header_raw: str
    header_norm: str
    container_raw: Optional[str]
    container_norm: Optional[str]
    is_container: bool
    paragraphs: List[str]
    footnotes: List[str] = field(default_factory=list)
    
    # Computed properties
    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs).strip()
    
    @property
    def text_with_footnotes(self) -> str:
        """Text including footnotes (useful for methods extraction)."""
        parts = self.paragraphs.copy()
        if self.footnotes:
            parts.append("\n[FOOTNOTES]\n" + "\n".join(self.footnotes))
        return "\n\n".join(parts).strip()
    
    @property
    def preview_head(self) -> str:
        if not self.paragraphs:
            return ""
        return first_n_sentences(self.paragraphs[0], n=3)
    
    @property
    def preview_tail(self) -> str:
        if not self.paragraphs:
            return ""
        return last_n_sentences(self.paragraphs[-1], n=3)
    
    @property
    def word_count(self) -> int:
        return count_words(self.text)
    
    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)


@dataclass
class PaperMetadata:
    """Bibliographic metadata extracted from TEI header."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    keywords: List[str] = field(default_factory=list)

# =============================================================================
# TEI Parsing
# =============================================================================

def extract_metadata(root: etree._Element) -> PaperMetadata:
    """Extract bibliographic metadata from TEI header."""
    meta = PaperMetadata()
    
    # Title
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", namespaces=TEI_NS)
    if title_el is not None:
        meta.title = normalize_whitespace("".join(title_el.itertext()))
    
    # Authors
    for author in root.findall(".//tei:teiHeader//tei:author", namespaces=TEI_NS):
        persname = author.find(".//tei:persName", namespaces=TEI_NS)
        if persname is not None:
            forename = persname.find("tei:forename", namespaces=TEI_NS)
            surname = persname.find("tei:surname", namespaces=TEI_NS)
            name_parts = []
            if forename is not None:
                name_parts.append(normalize_whitespace("".join(forename.itertext())))
            if surname is not None:
                name_parts.append(normalize_whitespace("".join(surname.itertext())))
            if name_parts:
                meta.authors.append(" ".join(name_parts))
    
    # Journal
    journal_el = root.find(".//tei:teiHeader//tei:monogr/tei:title", namespaces=TEI_NS)
    if journal_el is not None:
        meta.journal = normalize_whitespace("".join(journal_el.itertext()))
    
    # Year
    date_el = root.find(".//tei:teiHeader//tei:publicationStmt/tei:date", namespaces=TEI_NS)
    if date_el is not None:
        year_text = date_el.get("when", "") or normalize_whitespace("".join(date_el.itertext()))
        match = re.search(r"\b(19|20)\d{2}\b", year_text)
        if match:
            meta.year = match.group(0)
    
    # DOI
    for idno in root.findall(".//tei:teiHeader//tei:idno", namespaces=TEI_NS):
        if idno.get("type", "").lower() == "doi":
            meta.doi = normalize_whitespace("".join(idno.itertext()))
            break
    
    # Keywords
    for kw in root.findall(".//tei:teiHeader//tei:keywords//tei:term", namespaces=TEI_NS):
        kw_text = normalize_whitespace("".join(kw.itertext()))
        if kw_text:
            meta.keywords.append(kw_text)
    
    return meta


def extract_abstract(root: etree._Element) -> str:
    """Extract abstract text from TEI."""
    abs_el = root.find(".//tei:teiHeader/tei:profileDesc//tei:abstract", namespaces=TEI_NS)
    if abs_el is None:
        return ""
    
    # Try to get structured paragraphs first
    paras = abs_el.findall(".//tei:p", namespaces=TEI_NS)
    if paras:
        texts = [normalize_whitespace("".join(p.itertext())) for p in paras]
        texts = [t for t in texts if t]
        return "\n\n".join(texts)
    
    # Fall back to all text
    return normalize_whitespace("".join(abs_el.itertext()))


def extract_footnotes(div: etree._Element) -> List[str]:
    """Extract footnotes from a div element."""
    footnotes = []
    for note in div.findall(".//tei:note", namespaces=TEI_NS):
        note_type = note.get("type", "").lower()
        if note_type in ("foot", "footnote", ""):
            text = normalize_whitespace("".join(note.itertext()))
            if text and len(text) > 10:  # Filter trivial notes
                footnotes.append(text)
    return footnotes


def extract_sections_recursive(
    parent: etree._Element,
    paper_id: str,
    depth: int = 0,
    sec_counter: List[int] = None,
    container_raw: Optional[str] = None,
    container_norm: Optional[str] = None
) -> List[SectionBlock]:
    """
    Recursively extract sections from TEI body.
    
    This handles nested subsections properly, which is important for
    papers with complex structure (e.g., "3. Methods" > "3.1 Data" > "3.2 Model").
    """
    if sec_counter is None:
        sec_counter = [0]
    
    sections: List[SectionBlock] = []
    
    # Find divs at current level
    divs = parent.findall("./tei:div", namespaces=TEI_NS)
    
    current_container_raw = container_raw
    current_container_norm = container_norm
    
    for div in divs:
        sec_counter[0] += 1
        sec_index = sec_counter[0]
        
        # Extract header
        head = div.find("./tei:head", namespaces=TEI_NS)
        header_raw = normalize_whitespace("".join(head.itertext())) if head is not None else ""
        header_raw = header_raw if header_raw else "NO_HEADER"
        header_norm = normalize_header(header_raw) if header_raw != "NO_HEADER" else ""
        
        # Extract direct paragraphs only (not from nested divs)
        paras = div.findall("./tei:p", namespaces=TEI_NS)
        paragraphs = []
        for p in paras:
            text = normalize_whitespace("".join(p.itertext()))
            if text:
                paragraphs.append(text)
        
        # Extract footnotes
        footnotes = extract_footnotes(div)
        
        # Determine if this is a container (header-only section)
        is_container = (len(paragraphs) == 0) and (header_raw != "NO_HEADER")
        
        # Handle container inheritance
        if is_container:
            current_container_raw = header_raw
            current_container_norm = header_norm
            sections.append(SectionBlock(
                paper_id=paper_id,
                sec_index=sec_index,
                depth=depth,
                header_raw=header_raw,
                header_norm=header_norm,
                container_raw=None,
                container_norm=None,
                is_container=True,
                paragraphs=[],
                footnotes=[],
            ))
        else:
            # Major breaks reset container
            if is_major_break(header_norm):
                effective_container_raw = None
                effective_container_norm = None
                current_container_raw = None
                current_container_norm = None
            else:
                effective_container_raw = current_container_raw
                effective_container_norm = current_container_norm
            
            sections.append(SectionBlock(
                paper_id=paper_id,
                sec_index=sec_index,
                depth=depth,
                header_raw=header_raw,
                header_norm=header_norm,
                container_raw=effective_container_raw,
                container_norm=effective_container_norm,
                is_container=False,
                paragraphs=paragraphs,
                footnotes=footnotes,
            ))
        
        # Recursively process nested divs
        nested = extract_sections_recursive(
            div,
            paper_id=paper_id,
            depth=depth + 1,
            sec_counter=sec_counter,
            container_raw=header_raw if not is_major_break(header_norm) else None,
            container_norm=header_norm if not is_major_break(header_norm) else None,
        )
        sections.extend(nested)
    
    return sections


def extract_all_sections(root: etree._Element, paper_id: str) -> List[SectionBlock]:
    """Extract all sections from TEI body."""
    body = root.find(".//tei:text/tei:body", namespaces=TEI_NS)
    if body is None:
        return []
    
    return extract_sections_recursive(body, paper_id=paper_id)


def parse_tei_file(tei_path: Path, paper_id: Optional[str] = None) -> Dict[str, Any]:
    """Parse TEI XML file into structured JSON."""
    xml_bytes = tei_path.read_bytes()
    root = etree.fromstring(xml_bytes)
    
    pid = paper_id or tei_path.stem.replace(".tei", "")
    
    # Extract components
    metadata = extract_metadata(root)
    abstract = extract_abstract(root)
    sections = extract_all_sections(root, paper_id=pid)
    
    # Build output
    return {
        "paper_id": pid,
        "tei_path": str(tei_path),
        "metadata": asdict(metadata),
        "abstract": abstract,
        "sections": [
            {
                **asdict(s),
                "text": s.text,
                "text_with_footnotes": s.text_with_footnotes,
                "preview_head": s.preview_head,
                "preview_tail": s.preview_tail,
                "word_count": s.word_count,
                "paragraph_count": s.paragraph_count,
            }
            for s in sections
        ],
        "stats": {
            "total_sections": len(sections),
            "content_sections": len([s for s in sections if not s.is_container]),
            "total_words": sum(s.word_count for s in sections),
            "has_abstract": bool(abstract),
        }
    }

# =============================================================================
# Bundle Construction (for downstream extraction)
# =============================================================================

def build_opening_bundle(parsed: Dict[str, Any], max_intro_sentences: int = 10) -> str:
    """
    Build opening context bundle: Abstract + Introduction.
    Used for Prompt 1 (Rationale extraction).
    """
    parts = []
    
    # Abstract
    abstract = (parsed.get("abstract") or "").strip()
    if abstract:
        parts.append(f"### ABSTRACT ###\n{abstract}")
    
    # First content section (usually Introduction)
    for sec in parsed.get("sections", []):
        if sec.get("is_container"):
            continue
        text = sec.get("text", "").strip()
        if text:
            header = sec.get("header_raw", "FIRST_SECTION")
            # Truncate to first N sentences for efficiency
            truncated = first_n_sentences(text, n=max_intro_sentences)
            parts.append(f"### FIRST SECTION: {header} ###\n{truncated}")
            break
    
    return "\n\n".join(parts).strip()

# =============================================================================
# Main
# =============================================================================

VALID_DOMAINS = [
    "sociology", "political_science", "psychology",
    "social_sciences_interdisciplinary", "communication",
]


def process_domain(
    domain: str,
    data_root: Path,
    grobid_url: str,
    timeout: int,
    overwrite: bool,
    limit: int,
    skip_grobid: bool,
) -> Tuple[int, int]:
    """Process all papers for a single domain. Returns (success_count, error_count)."""
    pdf_dir = data_root / "raw" / domain
    tei_dir = data_root / "tei" / "raw" / domain
    parsed_dir = data_root / "tei" / "parsed" / domain

    tei_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    if skip_grobid:
        tei_files = sorted(tei_dir.glob("*.tei.xml"))
        if not tei_files:
            logger.warning(f"[{domain}] No TEI files found in: {tei_dir}")
            return 0, 0
        logger.info(f"[{domain}] Found {len(tei_files)} TEI files to parse")
        if limit > 0:
            tei_files = tei_files[:limit]

        for tei_path in tei_files:
            paper_id = tei_path.stem.replace(".tei", "")
            parsed_path = parsed_dir / f"{paper_id}.parsed.json"

            if parsed_path.exists() and not overwrite:
                logger.info(f"[{domain}] [SKIP] {paper_id} (exists)")
                continue

            try:
                parsed = parse_tei_file(tei_path, paper_id=paper_id)
                parsed_path.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                success_count += 1
                logger.info(
                    f"[{domain}] [OK] {paper_id}: "
                    f"{parsed['stats']['content_sections']} sections, "
                    f"{parsed['stats']['total_words']} words"
                )
            except Exception as e:
                error_count += 1
                logger.error(f"[{domain}] [ERROR] {paper_id}: {e}")
    else:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"[{domain}] No PDFs found in: {pdf_dir}")
            return 0, 0
        logger.info(f"[{domain}] Found {len(pdfs)} PDFs to process")
        if limit > 0:
            pdfs = pdfs[:limit]

        for pdf_path in pdfs:
            paper_id = pdf_path.stem
            tei_path = tei_dir / f"{paper_id}.tei.xml"
            parsed_path = parsed_dir / f"{paper_id}.parsed.json"

            if parsed_path.exists() and not overwrite:
                logger.info(f"[{domain}] [SKIP] {paper_id} (exists)")
                continue

            try:
                if not tei_path.exists() or overwrite:
                    logger.info(f"[{domain}] Processing {paper_id}...")
                    tei_xml = grobid_process_fulltext(
                        pdf_path=pdf_path,
                        grobid_url=grobid_url,
                        timeout_sec=timeout,
                    )
                    tei_path.write_text(tei_xml, encoding="utf-8")

                parsed = parse_tei_file(tei_path, paper_id=paper_id)
                parsed_path.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                success_count += 1
                logger.info(
                    f"[{domain}] [OK] {paper_id}: "
                    f"{parsed['stats']['content_sections']} sections, "
                    f"{parsed['stats']['total_words']} words"
                )
            except Exception as e:
                error_count += 1
                logger.error(f"[{domain}] [ERROR] {paper_id}: {e}")

    return success_count, error_count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert PDFs to parsed JSON via GROBID"
    )
    ap.add_argument(
        "--domain", required=True,
        help="Domain to process (sociology, political_science, psychology, "
             "social_sciences_interdisciplinary, communication) or 'all'"
    )
    ap.add_argument("--data_root", default="./data",
                    help="Root data directory (default: ./data)")
    ap.add_argument("--grobid_url", default="http://localhost:8070",
                    help="GROBID server URL")
    ap.add_argument("--timeout", type=int, default=180,
                    help="HTTP timeout per PDF (seconds)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N PDFs per domain (0 = no limit)")
    ap.add_argument("--skip_grobid", action="store_true",
                    help="Skip GROBID, parse existing TEI files only")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    # Resolve domains
    if args.domain.lower() == "all":
        domains = VALID_DOMAINS
    else:
        if args.domain not in VALID_DOMAINS:
            raise SystemExit(
                f"Invalid domain: {args.domain!r}. "
                f"Choose from: {VALID_DOMAINS} or 'all'"
            )
        domains = [args.domain]

    total_success = 0
    total_errors = 0

    for domain in domains:
        logger.info(f"{'='*60}")
        logger.info(f"Processing domain: {domain}")
        logger.info(f"{'='*60}")

        s, e = process_domain(
            domain=domain,
            data_root=data_root,
            grobid_url=args.grobid_url,
            timeout=args.timeout,
            overwrite=args.overwrite,
            limit=args.limit,
            skip_grobid=args.skip_grobid,
        )
        total_success += s
        total_errors += e

    logger.info(f"[DONE] Total — Success: {total_success}, Errors: {total_errors}")


if __name__ == "__main__":
    main()
