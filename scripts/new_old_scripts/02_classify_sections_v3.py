#!/usr/bin/env python3
"""
Script 2 (v3): Section Classification with Multi-Tag Support

Key Changes from v2:
1. Removed NARRATIVE_ANALYSIS label (merged into RESULTS)
2. Multi-tag classification: sections can have multiple labels with weights
3. Updated heuristics and LLM prompt for 9 labels
4. Better handling of merged sections via multi-tagging
5. Container validation: orphaned containers demoted to regular sections
6. Random sampling mode for pilot testing

Input:  ./data/tei/parsed/{domain}/*.json
Output: ./data/tei/processed/{domain}/*.processed.json

Usage:
  # Process a single domain
  python scripts/02_classify_sections_v3.py --domain sociology

  # Process all domains
  python scripts/02_classify_sections_v3.py --domain all

  # Random sample of 20 papers across all domains (for pilot testing)
  python scripts/02_classify_sections_v3.py --domain all --sample 20 --seed 42

  # Later: run all, skipping already-processed papers
  python scripts/02_classify_sections_v3.py --domain all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError("Install openai: pip install openai") from e


# =============================================================================
# Constants and Label Definitions (NARRATIVE_ANALYSIS removed)
# =============================================================================

MAJOR_LABELS = [
    "INTRO",           # Introduction, overview, framing
    "THEORY_LIT",      # Literature review, theoretical framework, hypotheses
    "METHODS",         # Data, measures, analytical approach
    "RESULTS",         # Findings, analysis output (includes narrative analysis)
    "DISCUSSION",      # Interpretation, implications
    "CONCLUSION",      # Summary, future work
    "REFERENCES",      # Bibliography
    "APPENDIX",        # Supplementary material
    "OTHER",           # Acknowledgements, etc.
]

LabelLiteral = Literal[
    "INTRO", "THEORY_LIT", "METHODS", "RESULTS", "DISCUSSION",
    "CONCLUSION", "REFERENCES", "APPENDIX", "OTHER"
]


# =============================================================================
# Pydantic Models for Multi-Tag Classification
# =============================================================================

class TagAssignment(BaseModel):
    """A single tag assignment with weight."""
    label: LabelLiteral = Field(description="Functional section label")
    weight: float = Field(
        ge=0.0, le=1.0,
        description="Weight/proportion for this label (0-1)"
    )
    reason: str = Field(
        default="",
        max_length=100,
        description="Brief justification for this tag"
    )


class SectionClassification(BaseModel):
    """Classification result for a single section with multi-tag support."""
    sec_index: int = Field(description="Section index from the input")
    primary_label: LabelLiteral = Field(description="Primary functional label")
    tags: List[TagAssignment] = Field(
        description="All applicable tags with weights (should sum to ~1.0)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall classification confidence (0-1)"
    )


class BatchClassificationResponse(BaseModel):
    """Response containing classifications for multiple sections."""
    classifications: List[SectionClassification] = Field(
        description="List of section classifications"
    )


# =============================================================================
# Heuristic Patterns (Social Science Specific)
# =============================================================================

HEURISTIC_PATTERNS: Dict[str, List[str]] = {
    "REFERENCES": [
        r"^\s*references?\s*$",
        r"^\s*bibliograph(y|ies)\s*$",
        r"^\s*works?\s+cited\s*$",
        r"^\s*cited\s+literature\s*$",
    ],
    "APPENDIX": [
        r"^\s*appendix",
        r"^\s*appendices",
        r"^\s*annex",
        r"^\s*supplement(ary|al)?\s+(material|information|data|appendix)",
        r"^\s*online\s+(appendix|supplement)",
        r"^\s*supporting\s+information",
    ],
    "CONCLUSION": [
        r"^\s*conclusion(s)?\s*$",
        r"^\s*concluding\s+remarks?\s*$",
        r"^\s*summary\s+and\s+conclusion",
        r"^\s*final\s+remarks?\s*$",
        r"^\s*closing\s+remarks?\s*$",
        r"^\s*summary\s*$",
        r"^\s*future\s+(research|work|directions?|stud)",
    ],
    "DISCUSSION": [
        r"^\s*discussion\s*$",
        r"^\s*general\s+discussion",
        r"^\s*discussion\s+and\s+(?!conclusion)",
        r"^\s*implications?\s*$",
        r"^\s*interpretation\s*$",
        r"^\s*theoretical\s+implications?",
        r"^\s*practical\s+implications?",
        r"^\s*limitations?\s*$",
        r"^\s*limitations?\s+and\s+future",
    ],
    "RESULTS": [
        r"^\s*results?\s*$",
        r"^\s*findings?\s*$",
        r"^\s*empirical\s+results?",
        r"^\s*main\s+(results?|findings?)",
        r"^\s*descriptive\s+(results?|statistics?)",
        r"^\s*regression\s+(results?|analysis)",
        r"^\s*robustness\s+(checks?|analysis|tests?)",
        r"^\s*sensitivity\s+analysis",
        r"^\s*supplementary\s+(results?|analysis)",
        r"^\s*additional\s+(results?|analysis|findings?)",
        r"^\s*results?\s+and\s+analysis",
        # Qualitative results — require "results" or "findings" in header
        r"^\s*qualitative\s+(findings?|results?)",
    ],
    "METHODS": [
        r"\bmethod(s|ology)?\s*$",
        r"^\s*materials?\s+and\s+methods?",
        r"^\s*data\s+and\s+method",
        r"^\s*research\s+(design|method|approach)",
        r"^\s*study\s+design",
        r"^\s*empirical\s+(strategy|approach|design)",
        r"^\s*data\s*$",
        r"^\s*data\s+(collection|sources?|description)",
        r"^\s*sample\s*$",
        r"^\s*sample\s+(selection|description|and\s+data)",
        r"^\s*corpus\s*$",
        r"^\s*dataset\s*$",
        r"^\s*(dependent|independent|control)\s+variables?",
        r"^\s*variables?\s+and\s+measures?",
        r"^\s*measures?\s*$",
        r"^\s*measurement\s*$",
        r"^\s*operationalization",
        r"^\s*coding\s+(scheme|procedure)",
        r"^\s*analytical\s+(strategy|approach|framework|method)",
        r"^\s*estimation\s+(strategy|approach|method)",
        r"^\s*identification\s+strategy",
        r"^\s*model\s+(specification|setup)",
        r"^\s*statistical\s+(method|approach|analysis|model)",
        r"^\s*topic\s+model",
        r"^\s*text\s+(analysis|processing|preprocessing)",
        r"^\s*preprocessing",
    ],
    "THEORY_LIT": [
        r"^\s*literature\s+review",
        r"^\s*related\s+(work|literature|research)",
        r"^\s*prior\s+(work|research|literature)",
        r"^\s*previous\s+(work|research|stud)",
        r"^\s*background\s*$",
        r"^\s*theoretical\s+(background|framework|context)",
        r"^\s*conceptual\s+(background|framework)",
        r"^\s*theory\s*$",
        r"^\s*hypothes[ie]s",
        r"^\s*research\s+questions?",
        r"^\s*state\s+of\s+(the\s+)?art",
        r"^\s*framing\s+",
        r"^\s*cultural\s+.*\s+theory",
        # Headers that are often lit review but were previously ambiguous
        r"^\s*analysis\s+of\s+(the\s+)?(literature|existing|prior|previous)",
        r"^\s*thematic\s+(overview|review|background)",
        r"^\s*case\s+study\s+(literature|review|background)",
        r"^\s*review\s+of\s+(the\s+)?(literature|research|evidence)",
    ],
    "INTRO": [
        r"^\s*introduction\s*$",
        r"^\s*overview\s*$",
        r"^\s*preface\s*$",
        r"^\s*motivation\s*$",
    ],
}

# Priority order for resolving conflicts
HEURISTIC_PRIORITY = [
    "REFERENCES",
    "APPENDIX", 
    "CONCLUSION",
    "DISCUSSION",
    "RESULTS",
    "METHODS",
    "THEORY_LIT",
    "INTRO",
]

# Patterns for merged sections - now returns multi-tags
MERGED_SECTION_PATTERNS = [
    (r"results?\s+and\s+discussion", [("RESULTS", 0.6), ("DISCUSSION", 0.4)]),
    (r"discussion\s+and\s+conclusion", [("DISCUSSION", 0.6), ("CONCLUSION", 0.4)]),
    (r"findings?\s+and\s+discussion", [("RESULTS", 0.6), ("DISCUSSION", 0.4)]),
    (r"summary\s+and\s+conclusion", [("CONCLUSION", 1.0)]),
    (r"discussion\s+and\s+implications?", [("DISCUSSION", 1.0)]),
    (r"implications?\s+and\s+conclusion", [("DISCUSSION", 0.5), ("CONCLUSION", 0.5)]),
    (r"limitations?\s+and\s+future", [("DISCUSSION", 0.5), ("CONCLUSION", 0.5)]),
    (r"data\s+and\s+(method|analysis)", [("METHODS", 1.0)]),
    (r"method(s)?\s+and\s+data", [("METHODS", 1.0)]),
    (r"theory\s+and\s+hypothes", [("THEORY_LIT", 1.0)]),
    (r"literature\s+and\s+hypothes", [("THEORY_LIT", 1.0)]),
    (r"results?\s+and\s+analysis", [("RESULTS", 1.0)]),
]


# =============================================================================
# Heuristic Matching
# =============================================================================

def normalize_header(header: str) -> str:
    """Normalize header for pattern matching."""
    h = (header or "").strip().lower()
    h = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", h)
    h = re.sub(r"^\s*[ivxlc]+[\.\)]\s*", "", h, flags=re.I)
    h = re.sub(r"^\s*[a-z][\.\)]\s*", "", h, flags=re.I)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def check_merged_section(header_norm: str) -> Optional[List[Tuple[str, float]]]:
    """Check if header indicates a merged section. Returns list of (label, weight) tuples."""
    for pattern, tags in MERGED_SECTION_PATTERNS:
        if re.search(pattern, header_norm, re.I):
            return tags
    return None


def apply_heuristics(header_raw: str) -> Tuple[Optional[str], List[Tuple[str, float]], float, str]:
    """
    Apply heuristic patterns to classify section header.
    
    Returns: (primary_label, tags_list, confidence, reason)
    """
    if not header_raw or header_raw == "NO_HEADER":
        return None, [], 0.0, "no_header"
    
    header_norm = normalize_header(header_raw)
    if not header_norm:
        return None, [], 0.0, "empty_after_norm"
    
    # Check for merged sections first (returns multi-tags)
    merged = check_merged_section(header_norm)
    if merged:
        primary = merged[0][0]  # First tag is primary
        return primary, merged, 0.85, f"merged_section:{'+'.join(t[0] for t in merged)}"
    
    # Try each label in priority order
    for label in HEURISTIC_PRIORITY:
        patterns = HEURISTIC_PATTERNS.get(label, [])
        for pattern in patterns:
            if re.search(pattern, header_norm, re.I):
                is_exact = pattern.endswith(r"\s*$") or pattern.endswith("$")
                conf = 0.95 if is_exact else 0.85
                # Single tag with full weight
                return label, [(label, 1.0)], conf, f"pattern:{pattern[:30]}"
    
    return None, [], 0.0, "no_match"


# =============================================================================
# Content-Based Cues
# =============================================================================

CONTENT_CUES = {
    "INTRO": [
        r"\b(this\s+(paper|study|article|research)\s+(examines?|investigates?|explores?|analyzes?))\b",
        r"\b(we\s+(examine|investigate|explore|analyze|study|aim|seek))\b",
        r"\b(the\s+(purpose|aim|goal|objective)\s+of\s+this)\b",
        r"\b(in\s+this\s+(paper|study|article))\b",
    ],
    "THEORY_LIT": [
        r"\b([A-Z][a-z]+\s+(?:and\s+[A-Z][a-z]+\s+)?\(\d{4}\))",  # Author (2020)
        r"\b(prior\s+(research|work|studies?|literature))\b",
        r"\b(previous\s+(research|work|studies?|scholars?))\b",
        r"\b(existing\s+(research|literature|work|scholarship))\b",
        r"\b(scholars?\s+(have\s+)?(argued?|shown|found|demonstrated|noted|suggested))\b",
        r"\b(the\s+literature\s+(on|suggests?|shows?|indicates?))\b",
        r"\b(according\s+to)\b",
        r"\b(building\s+on|drawing\s+on)\s+\w+",
        r"\b(theoretical(ly)?|conceptual(ly)?)\s+(framework|perspective|lens|approach|model)\b",
        r"\b(hypothes[ie]s\b|we\s+hypothesize|we\s+expect\s+that)\b",
    ],
    "METHODS": [
        r"\b(we\s+(collected?|gathered?|scraped?|downloaded|obtained|sampled))\b",
        r"\b(our\s+(data|corpus|sample|dataset)\s+(consist|comprise|include|contain))\b",
        r"\b(we\s+(use|employ|apply|implement|run|fit|estimate)\s+(a\s+)?(topic\s+model|LDA|STM|regression|model))\b",
        r"\b(preprocessing|tokeniz|lemmatiz|stopword|stemm)\b",
        r"\b(k\s*=\s*\d+|number\s+of\s+topics?\s*(is|was|=))\b",
    ],
    "RESULTS": [
        r"\b(table\s+\d+|figure\s+\d+)\s+(shows?|presents?|displays?|illustrates?|reports?)\b",
        r"\b(we\s+(find|found|show|observe|identify|discover))\b",
        r"\b(results?\s+(indicate|suggest|show|reveal|demonstrate))\b",
        r"\b(significant(ly)?|coefficient|p\s*[<>=]|β\s*=|odds\s+ratio)\b",
        r"\b(topic\s+\d+|the\s+(first|second|third|largest|smallest)\s+topic)\b",
    ],
    "DISCUSSION": [
        r"\b(these\s+(findings?|results?)\s+(suggest|indicate|imply|support))\b",
        r"\b(we\s+(interpret|argue|contend|suggest|propose))\b",
        r"\b(implications?\s+(for|of))\b",
        r"\b(consistent\s+with|in\s+line\s+with|contrary\s+to)\b",
        r"\b(limitation|caveat|shortcoming|weakness)\b",
    ],
    "CONCLUSION": [
        r"\b(in\s+(conclusion|sum|summary)|to\s+(conclude|summarize))\b",
        r"\b(this\s+(paper|study)\s+(has\s+)?(shown|demonstrated|contributed))\b",
        r"\b(future\s+(research|work|studies?)\s+(should|could|might|may))\b",
        r"\b(overall|taken\s+together|in\s+closing)\b",
    ],
}


def detect_content_cues(text: str) -> Dict[str, int]:
    """Count content cues for each label in given text."""
    counts = {label: 0 for label in CONTENT_CUES}
    text_lower = (text or "").lower()
    
    for label, patterns in CONTENT_CUES.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.I)
            counts[label] += len(matches)
    
    return counts


def get_dominant_cue(text: str) -> Tuple[Optional[str], int]:
    """Get the label with most content cues."""
    counts = detect_content_cues(text)
    if not any(counts.values()):
        return None, 0
    
    best_label = max(counts, key=counts.get)
    return best_label, counts[best_label]


# =============================================================================
# LLM Classification
# =============================================================================

SYSTEM_PROMPT = """You are an expert Research Archivist classifying academic paper sections by their FUNCTION.

CATEGORIES (9 total):
- INTRO: Opening section framing the research problem, contributions, and roadmap
- THEORY_LIT: Prior work, theoretical background, hypotheses; describes what OTHERS did
- METHODS: What THIS paper did - data collection, measures, preprocessing, models, estimation
- RESULTS: Reports findings - statistics, effects, patterns, "we find/show" (includes qualitative findings)
- DISCUSSION: Interprets meaning, implications, connects to theory, limitations
- CONCLUSION: Final wrap-up, summary, future research, closing remarks
- REFERENCES / APPENDIX / OTHER: As named

MULTI-TAG CLASSIFICATION:
Many social science sections serve MULTIPLE functions. For example:
- "Results and Discussion" → RESULTS (0.6) + DISCUSSION (0.4)
- "Data and Methods" → METHODS (1.0)
- "Discussion and Conclusion" → DISCUSSION (0.6) + CONCLUSION (0.4)

For each section:
1. Assign a PRIMARY label (the dominant function)
2. Provide ALL applicable tags with weights (should sum to ~1.0)
3. Single-function sections get one tag with weight 1.0

CRITICAL DISAMBIGUATION:
1) RESULTS vs DISCUSSION:
   - RESULTS: Emphasizes reporting (numbers, effects, what was found)
   - DISCUSSION: Emphasizes interpretation, meaning, implications

2) METHODS vs THEORY_LIT:
   - METHODS: "we collected", "we use", "our sample" - actions by authors
   - THEORY_LIT: Citations, "X (2020) found", "prior research shows" - describing others

3) RESULTS vs THEORY_LIT (common misclassification!):
   - RESULTS: Reports THIS paper's findings - "we find", "Table 1 shows", statistical values
   - THEORY_LIT: Reviews what OTHER scholars found - dense citations "(Author, 2020)", "prior research shows", "scholars have argued"
   - A section heavy with citations and "Author (year)" patterns is almost certainly THEORY_LIT, not RESULTS
   - THEORY_LIT typically appears in the first half of the paper, before METHODS

4) Position matters:
   - INTRO is typically section 1
   - THEORY_LIT typically appears in sections 2-3 (before METHODS)
   - RESULTS typically appears AFTER METHODS
   - CONCLUSION is typically near the end"""


@dataclass
class LLMConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.0
    max_tokens: int = 4000
    max_items_per_batch: int = 12
    retries: int = 3
    retry_delay: float = 2.0


def build_classification_prompt(
    paper_id: str,
    items: List[Dict[str, Any]],
    total_sections: int
) -> str:
    """Build prompt for batch section classification."""
    lines = [
        f"PAPER: {paper_id}",
        f"TOTAL SECTIONS: {total_sections}",
        "",
        "Classify each section with multi-tag support. Weights should sum to ~1.0.",
        ""
    ]
    
    for item in items:
        sec_idx = item["sec_index"]
        pos_frac = sec_idx / max(total_sections, 1)
        
        lines.append(f"--- SECTION {sec_idx} ---")
        lines.append(f"Position: {sec_idx}/{total_sections} ({pos_frac:.1%})")
        lines.append(f"Header: {item.get('header_raw', '')!r}")
        
        prev_info = item.get("prev_label", "?")
        next_info = item.get("next_label", "?")
        lines.append(f"Neighbors: prev=[{prev_info}] next=[{next_info}]")
        
        head = (item.get("preview_head", "") or "")[:300]
        tail = (item.get("preview_tail", "") or "")[:200]
        if head:
            lines.append(f"HEAD: {head!r}")
        if tail and tail != head:
            lines.append(f"TAIL: {tail!r}")
        
        lines.append("")
    
    return "\n".join(lines)


def call_llm_classification(
    client: OpenAI,
    cfg: LLMConfig,
    prompt: str
) -> Dict[str, Any]:
    """Call OpenAI API for section classification using structured outputs."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    last_error = None
    for attempt in range(1, cfg.retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                messages=messages,
                response_format=BatchClassificationResponse,
            )
            
            parsed_response = completion.choices[0].message.parsed
            
            if parsed_response is None:
                refusal = completion.choices[0].message.refusal
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")
                raise ValueError("Parsed response is None")
            
            return parsed_response.model_dump()
            
        except ValidationError as e:
            last_error = e
            logger.warning(f"Validation error on attempt {attempt}: {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"API error on attempt {attempt}: {e}")
        
        if attempt < cfg.retries:
            time.sleep(cfg.retry_delay * attempt)
    
    raise RuntimeError(f"LLM classification failed: {last_error}")


# =============================================================================
# Guardrails and Post-Processing
# =============================================================================

def apply_guardrails(
    primary_label: str,
    tags: List[Dict[str, Any]],
    header_norm: str,
    preview_head: str,
    preview_tail: str,
    sec_index: int,
    total_sections: int
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Apply guardrails to correct common misclassifications.
    Returns: (corrected_primary, corrected_tags, adjustment_note)
    """
    note = ""
    
    is_first = sec_index <= 1
    is_near_end = sec_index >= total_sections - 1
    position_frac = sec_index / max(total_sections, 1)
    
    # CONCLUSION guardrail
    if primary_label == "CONCLUSION":
        has_conclusion_cues = any(
            re.search(p, preview_tail, re.I) 
            for p in CONTENT_CUES.get("CONCLUSION", [])
        )
        header_is_conclusion = re.search(r"\bconclud", header_norm, re.I)
        
        if not (is_near_end or has_conclusion_cues or header_is_conclusion):
            primary_label = "DISCUSSION"
            tags = [{"label": "DISCUSSION", "weight": 1.0, "reason": "guardrail_correction"}]
            note = "conclusion_not_at_end"
    
    # INTRO guardrail
    if primary_label == "INTRO" and position_frac > 0.3:
        primary_label = "THEORY_LIT"
        tags = [{"label": "THEORY_LIT", "weight": 1.0, "reason": "guardrail_correction"}]
        note = "intro_too_late"
    
    # RESULTS → THEORY_LIT guardrail
    # A section in the first 40% of the paper labeled RESULTS that reads like
    # literature review is almost certainly misclassified.
    if primary_label == "RESULTS" and position_frac < 0.4:
        combined_text = (preview_head or "") + " " + (preview_tail or "")
        lit_cues = sum(
            len(re.findall(p, combined_text, re.I))
            for p in CONTENT_CUES.get("THEORY_LIT", [])
        )
        result_cues = sum(
            len(re.findall(p, combined_text, re.I))
            for p in CONTENT_CUES.get("RESULTS", [])
        )
        
        # If lit cues dominate and there are barely any results cues, reclassify
        if lit_cues >= 3 and lit_cues > result_cues * 2:
            primary_label = "THEORY_LIT"
            tags = [{"label": "THEORY_LIT", "weight": 1.0, "reason": "guardrail_lit_over_results"}]
            note = f"results_looks_like_lit:lit={lit_cues},res={result_cues},pos={position_frac:.0%}"
    
    # Also catch RESULTS in first 40% even without strong lit cues,
    # if there are zero results cues — likely an ambiguous section
    if primary_label == "RESULTS" and position_frac < 0.3:
        combined_text = (preview_head or "") + " " + (preview_tail or "")
        result_cues = sum(
            len(re.findall(p, combined_text, re.I))
            for p in CONTENT_CUES.get("RESULTS", [])
        )
        if result_cues == 0:
            # No results evidence at all in a very early section — suspicious
            # Check if header itself contains "result" or "finding" explicitly
            header_has_result = re.search(r"\b(results?|findings?)\b", header_norm, re.I)
            if not header_has_result:
                primary_label = "THEORY_LIT"
                tags = [{"label": "THEORY_LIT", "weight": 1.0, "reason": "guardrail_early_no_evidence"}]
                note = f"early_results_no_evidence:pos={position_frac:.0%}"
    
    # RESULTS/DISCUSSION disambiguation
    if primary_label in ("RESULTS", "DISCUSSION"):
        combined_text = (preview_head or "") + " " + (preview_tail or "")
        result_cues = sum(
            len(re.findall(p, combined_text, re.I))
            for p in CONTENT_CUES.get("RESULTS", [])
        )
        disc_cues = sum(
            len(re.findall(p, combined_text, re.I))
            for p in CONTENT_CUES.get("DISCUSSION", [])
        )
        
        if result_cues > disc_cues + 2 and primary_label == "DISCUSSION":
            primary_label = "RESULTS"
            tags = [{"label": "RESULTS", "weight": 1.0, "reason": "strong_results_cues"}]
            note = f"strong_results_cues:{result_cues}>{disc_cues}"
        elif disc_cues > result_cues + 2 and primary_label == "RESULTS":
            primary_label = "DISCUSSION"
            tags = [{"label": "DISCUSSION", "weight": 1.0, "reason": "strong_discussion_cues"}]
            note = f"strong_discussion_cues:{disc_cues}>{result_cues}"
    
    return primary_label, tags, note


def propagate_labels_from_neighbors(
    sections: List[Dict[str, Any]],
    max_iterations: int = 2
) -> None:
    """Propagate labels to fill gaps using neighbor information."""
    for iteration in range(max_iterations):
        changes = 0
        
        for i, sec in enumerate(sections):
            if sec.get("primary_label") is not None:
                continue
            if sec.get("is_container", False):
                continue
            
            prev_label = sections[i-1].get("primary_label") if i > 0 else None
            next_label = sections[i+1].get("primary_label") if i < len(sections)-1 else None
            
            if prev_label and prev_label == next_label:
                sec["primary_label"] = prev_label
                sec["tags"] = [{"label": prev_label, "weight": 1.0, "reason": "neighbor_propagation"}]
                sec["label_source"] = "neighbor_propagation"
                sec["confidence"] = 0.6
                changes += 1
        
        if changes == 0:
            break


# =============================================================================
# Container Validation
# =============================================================================

# Major section labels that should NEVER be nested under each other.
# If a child's header resolves to one of these, it contradicts being nested
# under a container that resolves to a DIFFERENT one.
MAJOR_SECTION_LABELS = {"INTRO", "THEORY_LIT", "METHODS", "RESULTS",
                        "DISCUSSION", "CONCLUSION", "REFERENCES", "APPENDIX"}


def _quick_label(header_raw: str) -> Optional[str]:
    """
    Try to resolve a header to a major label using heuristics.
    Returns None if no confident match. This is a lightweight check —
    it reuses apply_heuristics but only trusts high-confidence matches.
    """
    primary, _, conf, _ = apply_heuristics(header_raw)
    if primary and conf >= 0.85 and primary in MAJOR_SECTION_LABELS:
        return primary
    return None


def validate_containers(sections: List[Dict[str, Any]], paper_id: str = "") -> List[Dict[str, Any]]:
    """
    Validate container–child relationships and fix two types of GROBID errors:
    
    1. ORPHANED CONTAINERS: A container has no children referencing it.
       → Demote to regular section so it enters classification.
    
    2. MIS-NESTED CHILDREN: A child's header clearly belongs to a different
       major section than its container (e.g., "Results" nested under "Methods").
       → Strip the container assignment so the child is classified independently.
    
    Both fixes are conservative and only trigger on high-confidence heuristic
    matches — ambiguous cases are left unchanged for LLM classification.
    """
    if not sections:
        return sections

    # --- Pass 1: Find orphaned containers ---
    # Build set of container headers that are actually referenced by children
    referenced_containers = set()
    for sec in sections:
        cr = sec.get("container_raw")
        if cr:
            referenced_containers.add(cr)

    # --- Pass 2: Fix both error types ---
    orphan_count = 0
    unnest_count = 0
    result = []

    for sec in sections:
        sec = dict(sec)  # shallow copy

        # Fix 1: Orphaned containers
        if sec.get("is_container", False):
            header_raw = sec.get("header_raw", "")
            is_orphan = header_raw not in referenced_containers

            if is_orphan:
                sec["is_container"] = False
                sec["_container_demoted"] = True
                orphan_count += 1

        # Fix 2: Mis-nested children
        # Only check non-container sections that have a container assignment
        elif sec.get("container_raw") and not sec.get("is_container", False):
            child_label = _quick_label(sec.get("header_raw", ""))
            container_label = _quick_label(sec.get("container_raw", ""))

            if (child_label and container_label
                    and child_label != container_label):
                # Child's header clearly belongs to a different major section
                # than its container → strip the container assignment
                sec["_original_container_raw"] = sec["container_raw"]
                sec["_original_container_norm"] = sec.get("container_norm")
                sec["container_raw"] = None
                sec["container_norm"] = None
                sec["_container_unnested"] = True
                unnest_count += 1

        result.append(sec)

    if orphan_count > 0 or unnest_count > 0:
        logger.info(
            f"{paper_id}: Container fixes — "
            f"demoted {orphan_count} orphan(s), "
            f"unnested {unnest_count} mis-nested child(ren)"
        )

    return result


# =============================================================================
# Main Processing
# =============================================================================

def process_paper(
    parsed: Dict[str, Any],
    client: OpenAI,
    cfg: LLMConfig,
    cache_dir: Path,
    overwrite_cache: bool = False
) -> Dict[str, Any]:
    """Process a single paper's sections for classification."""
    paper_id = parsed.get("paper_id", "unknown")
    sections = parsed.get("sections", [])
    
    if not sections:
        logger.warning(f"{paper_id}: No sections found")
        return parsed
    
    # Validate containers before classification
    sections = validate_containers(sections, paper_id=paper_id)
    
    total_sections = len(sections)
    processed_sections = []
    unmapped_items = []
    stats = {"heuristic": 0, "llm": 0, "propagated": 0}
    
    # First pass: Apply heuristics
    for i, sec in enumerate(sections):
        sec_copy = dict(sec)
        header_raw = sec.get("header_raw", "")
        
        if sec.get("is_container", False):
            sec_copy["primary_label"] = None
            sec_copy["tags"] = []
            sec_copy["label_source"] = "container"
            processed_sections.append(sec_copy)
            continue
        
        primary, tags, conf, reason = apply_heuristics(header_raw)
        
        if primary:
            sec_copy["primary_label"] = primary
            sec_copy["tags"] = [{"label": t[0], "weight": t[1], "reason": reason} for t in tags]
            sec_copy["label_source"] = "heuristic"
            sec_copy["confidence"] = conf
            sec_copy["label_reason"] = reason
            stats["heuristic"] += 1
        else:
            sec_copy["primary_label"] = None
            sec_copy["tags"] = []
            unmapped_items.append({
                "list_index": i,
                "sec_index": sec.get("sec_index", i + 1),
                "header_raw": header_raw,
                "preview_head": sec.get("preview_head", ""),
                "preview_tail": sec.get("preview_tail", ""),
            })
        
        processed_sections.append(sec_copy)
    
    # Add neighbor context
    for item in unmapped_items:
        idx = item["list_index"]
        prev_sec = processed_sections[idx - 1] if idx > 0 else {}
        next_sec = processed_sections[idx + 1] if idx < len(processed_sections) - 1 else {}
        
        item["prev_label"] = prev_sec.get("primary_label", "?")
        item["next_label"] = next_sec.get("primary_label", "?")
    
    # Second pass: LLM classification
    if unmapped_items:
        cache_path = cache_dir / f"{paper_id}.llm_cache.json"
        
        if cache_path.exists() and not overwrite_cache:
            try:
                cached = json.loads(cache_path.read_text())
                llm_results = {
                    c["sec_index"]: c 
                    for c in cached.get("classifications", [])
                }
            except Exception as e:
                logger.warning(f"{paper_id}: Cache read failed: {e}")
                llm_results = {}
        else:
            llm_results = {}
            
            for batch_start in range(0, len(unmapped_items), cfg.max_items_per_batch):
                batch = unmapped_items[batch_start:batch_start + cfg.max_items_per_batch]
                
                try:
                    prompt = build_classification_prompt(paper_id, batch, total_sections)
                    response = call_llm_classification(client, cfg, prompt)
                    
                    for item in response.get("classifications", []):
                        llm_results[item["sec_index"]] = item
                        
                except Exception as e:
                    logger.error(f"{paper_id}: LLM batch failed: {e}")
                    for item in batch:
                        llm_results[item["sec_index"]] = {
                            "sec_index": item["sec_index"],
                            "primary_label": "OTHER",
                            "tags": [{"label": "OTHER", "weight": 1.0, "reason": "llm_error"}],
                            "confidence": 0.3,
                        }
            
            cache_path.write_text(json.dumps({
                "classifications": list(llm_results.values())
            }, indent=2))
        
        # Apply LLM results with guardrails
        for item in unmapped_items:
            idx = item["list_index"]
            sec_idx = item["sec_index"]
            sec = processed_sections[idx]
            
            llm_result = llm_results.get(sec_idx, {})
            raw_primary = llm_result.get("primary_label", "OTHER")
            raw_tags = llm_result.get("tags", [{"label": "OTHER", "weight": 1.0, "reason": ""}])
            raw_conf = llm_result.get("confidence", 0.5)
            
            final_primary, final_tags, guardrail_note = apply_guardrails(
                primary_label=raw_primary,
                tags=raw_tags,
                header_norm=normalize_header(item.get("header_raw", "")),
                preview_head=item.get("preview_head", ""),
                preview_tail=item.get("preview_tail", ""),
                sec_index=sec_idx,
                total_sections=total_sections
            )
            
            sec["primary_label"] = final_primary
            sec["tags"] = final_tags
            sec["label_source"] = "llm"
            sec["confidence"] = raw_conf * (0.9 if guardrail_note else 1.0)
            sec["label_reason"] = guardrail_note if guardrail_note else "llm"
            stats["llm"] += 1
    
    # Third pass: Neighbor propagation
    propagate_labels_from_neighbors(processed_sections)
    for sec in processed_sections:
        if sec.get("label_source") == "neighbor_propagation":
            stats["propagated"] += 1
    
    # Build output - maintain backward compatibility with major_label field
    for sec in processed_sections:
        sec["major_label"] = sec.get("primary_label")  # Alias for compatibility
    
    result = dict(parsed)
    result["sections"] = processed_sections
    result["classification_stats"] = {
        "total": total_sections,
        "by_source": stats,
        "model": cfg.model,
        "multi_tag_enabled": True,
    }
    
    return result


VALID_DOMAINS = [
    "sociology", "political_science", "psychology",
    "social_sciences_interdisciplinary", "communication",
]


def collect_files_for_domain(
    data_root: Path, domain: str
) -> List[Tuple[Path, Path, Path]]:
    """
    Collect (input_file, output_file, cache_dir) tuples for a domain.
    Returns list of (parsed_json_path, processed_json_path, cache_dir).
    """
    parsed_dir = data_root / "tei" / "parsed" / domain
    processed_dir = data_root / "tei" / "processed" / domain
    cache_dir = data_root / "tei" / "unmapped_llm" / domain

    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parsed_dir.glob("*.json"))
    return [(fp, processed_dir, cache_dir) for fp in files]


def main():
    ap = argparse.ArgumentParser(description="Classify paper sections with multi-tag support")
    ap.add_argument(
        "--domain", required=True,
        help="Domain to process (sociology, political_science, psychology, "
             "social_sciences_interdisciplinary, communication) or 'all'"
    )
    ap.add_argument("--data_root", default="./data",
                    help="Root data directory (default: ./data)")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N papers per domain (0 = no limit)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Randomly sample N papers PER DOMAIN for pilot testing. "
                         "E.g., --sample 30 with --domain all draws 30 from each of 5 domains.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for --sample (default: 42)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing processed files")
    args = ap.parse_args()

    if load_dotenv:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    cfg = LLMConfig(model=args.model, temperature=args.temperature)

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

    # Collect all candidate files across selected domains
    # Each item: (parsed_path, processed_dir, cache_dir, domain)
    all_items: List[Tuple[Path, Path, Path, str]] = []
    for domain in domains:
        for fp, proc_dir, c_dir in collect_files_for_domain(data_root, domain):
            all_items.append((fp, proc_dir, c_dir, domain))

    if not all_items:
        raise SystemExit("No parsed JSON files found in any selected domain")

    # Apply sampling or limit per domain
    if args.sample > 0:
        rng = random.Random(args.seed)
        sampled_items = []
        from collections import Counter
        domain_counts = Counter()
        for domain in domains:
            domain_items = [it for it in all_items if it[3] == domain]
            n = min(args.sample, len(domain_items))
            picked = rng.sample(domain_items, n)
            sampled_items.extend(picked)
            domain_counts[domain] = n
        all_items = sampled_items
        logger.info(
            f"Sampled {len(all_items)} papers — {args.sample} per domain (seed={args.seed})"
        )
        for d, c in sorted(domain_counts.items()):
            logger.info(f"  {d}: {c} papers")
    elif args.limit > 0:
        # Apply per-domain limit
        limited_items = []
        for domain in domains:
            domain_items = [it for it in all_items if it[3] == domain]
            limited_items.extend(domain_items[:args.limit])
        all_items = limited_items

    total = len(all_items)
    logger.info(f"Processing {total} papers with multi-tag classification")

    success = 0
    skipped = 0
    errors = 0

    for i, (fp, processed_dir, cache_dir, domain) in enumerate(all_items, 1):
        paper_id = fp.stem.replace(".parsed", "")
        out_path = processed_dir / f"{paper_id}.processed.json"

        if out_path.exists() and not args.overwrite:
            logger.info(f"[{domain}] [SKIP] {paper_id} (exists)")
            skipped += 1
            continue

        try:
            parsed = json.loads(fp.read_text())
            result = process_paper(
                parsed, client, cfg, cache_dir,
                overwrite_cache=args.overwrite
            )
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

            stats = result.get("classification_stats", {}).get("by_source", {})
            demoted = sum(
                1 for s in result.get("sections", [])
                if s.get("_container_demoted")
            )
            unnested = sum(
                1 for s in result.get("sections", [])
                if s.get("_container_unnested")
            )
            fix_str = ""
            if demoted or unnested:
                fix_str = f" fixes=demoted:{demoted},unnested:{unnested}"
            logger.info(
                f"[{domain}] [{i}/{total}] [OK] {paper_id}: "
                f"h={stats.get('heuristic',0)} "
                f"llm={stats.get('llm',0)} "
                f"prop={stats.get('propagated',0)}"
                f"{fix_str}"
            )
            success += 1

        except Exception as e:
            logger.error(f"[{domain}] [ERROR] {paper_id}: {e}")
            errors += 1

    logger.info(
        f"[DONE] Success: {success}, Skipped: {skipped}, Errors: {errors}"
    )


if __name__ == "__main__":
    main()