#!/usr/bin/env python3
"""
Script 3 (v4): Content Extraction — Revised Schema

Changes from v3:
1. Prompt 1 (Rationale): Extracts CANDIDATE SENTENCES for downstream supervised
   classification. Loose, inclusive extraction — grabs any sentence where authors
   justify or rationalize using computational text analysis methods.
2. Prompt 2 (Data Pipeline): Simplified to core fields — source type, source name,
   corpus size, unit of analysis, temporal structure, preprocessing reported.
3. Prompt 3 (Modeling + Interpretation): MERGED prompt covering the full topic
   modeling workflow — algorithm, K selection, validation, labeling, downstream.

Bundle routing (two bundles only):
  Bundle 2 (Technical): METHODS + RESULTS       → P2 Data, P3 Modeling & Interpretation

Requirements:
  pip install openai>=1.40.0 pydantic>=2.0.0 python-dotenv

Input:  ./data/tei/processed/{domain}/*.processed.json
Output: ./data/outputs/{domain}/{domain}.extractions.jsonl

Usage:
  python scripts/new_old_scripts/03_extract_v3.py --domain sociology
  python scripts/new_old_scripts/03_extract_v3.py --domain all
"""
from __future__ import annotations
import argparse, json, logging, os, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError("pip install openai") from e


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA — Slim definitions, no decision logic
# ═══════════════════════════════════════════════════════════════════════

class DataExtraction(BaseModel):
    data_source: List[Literal[
        "Social_Media", "News_Media",
        "Government_Or_Legal", "Organizational_Documents",
        "Academic_Text", "Survey_Or_Interview",
        "Existing_Dataset", "Other", "NOT_REPORTED",
    ]] = Field(default_factory=list, description="Genre of text data. Select ALL that apply.")
    data_source_detail: str = Field(default="", description="Specific source name.")
    corpus_size: Optional[int] = Field(default=None, description="Document count in primary corpus.")
    corpus_size_raw: str = Field(default="", description="Size as stated by authors.")
    unit_of_analysis: Literal[
        "Long_Form_Document", "Short_Form_Post",
        "Speech_Or_Interview", "Open_Ended_Response",
        "Sub_Document_Segment", "Abstract_Or_Title",
        "Other", "NOT_REPORTED",
    ] = Field(description="PRIMARY unit of text fed into the topic model.")
    preprocessing_reported: Literal["Yes", "No"] = Field(
        description="Whether any preprocessing steps are described.",
    )


class ModelExtraction(BaseModel):
    algorithm: Literal[
        "LDA", "STM", "NMF", "BERTopic", "Top2Vec",
        "BTM", "LSA", "CTM", "DTM",
        "Other_Topic_Model", "NOT_REPORTED",
    ] = Field(description="Algorithm family of the FINAL topic model.")
    algorithm_detail: str = Field(default="", description="Algorithm name as stated.")
    software: Literal[
        "stm_R", "MALLET", "gensim", "topicmodels_R",
        "scikit_learn", "BERTopic_pkg", "lda_R",
        "Other", "NOT_REPORTED",
    ] = Field(description="Topic modeling package.")
    software_detail: str = Field(default="", description="Software name if Other.")
    K_final: List[int] = Field(default_factory=list, description="Final K as list. [] if not reported.")
    K_evidence: str = Field(default="", description="Verbatim quote ≤60 words with final K.")
    metadata_in_model: Literal[
        "Structural", "Post_Hoc", "None", "NOT_REPORTED",
    ] = Field(
        description=(
            "Whether document-level metadata is used with the topic model.\n"
            "Structural=metadata enters the model itself (STM prevalence/content "
            "covariates, DMR prior, supervised TM, metadata as model input).\n"
            "Post_Hoc=metadata used only in downstream analysis after model "
            "(LDA then regression with covariates, topic proportions by group).\n"
            "None=no metadata used.\n"
            "NOT_REPORTED=unclear."
        ),
    )


class KSelectionExtraction(BaseModel):
    selection_strategy: Literal[
        "Named_Metric_Only", "Named_Metric_Plus_Human",
        "Human_Judgment_Only", "Theory_Or_Design_Prior",
        "No_Justification_Or_Not_Reported",
    ] = Field(description="Primary K selection strategy.")
    metrics_used: List[Literal[
        "Coherence", "Exclusivity_FREX", "Perplexity",
        "Held_Out_Likelihood", "Residuals",
        "Multi_Metric_Package", "Other_Metric",
    ]] = Field(default_factory=list, description="Named metrics. Empty if none.")
    human_judgment_described: Literal["Yes", "No"] = Field(
        description="Whether human review of outputs is described.",
    )
    selection_evidence: str = Field(default="", description="Verbatim quote ≤80 words.")


class LabelingExtraction(BaseModel):
    labeling_basis: Literal[
        "Top_Words_Only", "Top_Words_Plus_FREX",
        "Words_And_Docs", "Docs_Only", "NOT_REPORTED",
    ] = Field(description="Evidence used to assign labels.")
    labeling_process: Literal[
        "Informal", "Structured_Multi_Coder",
        "No_Labeling", "NOT_REPORTED",
    ] = Field(description="Who labeled and how rigorous.")
    label_validated: Literal["Yes", "No"] = Field(
        description="Any formal validation of labels described?",
    )
    labeling_evidence: str = Field(default="", description="Verbatim quote ≤80 words.")
    label_validation_evidence: str = Field(default="", description="Verbatim quote ≤60 words if Yes.")


class DownstreamExtraction(BaseModel):
    has_downstream_analysis: bool = Field(
        description="Any analysis beyond presenting/describing topics?",
    )
    topics_as_findings_or_inputs: Literal[
        "Findings", "Inputs", "Both", "Neither",
    ] = Field(description="Are topics the endpoint or inputs to further analysis?")
    epistemic_function: Literal[
        "Measurement_Tool", "Descriptive_Mapping",
        "Exploratory_For_Theory", "Data_Reduction",
        "Classification_Proxy", "NOT_REPORTED",
    ] = Field(description="What TM is FOR in the paper.")
    epistemic_function_evidence: str = Field(default="", description="Verbatim quote ≤60 words.")
    topic_model_position: Literal[
        "Primary_Method", "One_Of_Several",
        "Preliminary_Step", "Validation_Or_Robustness",
        "NOT_REPORTED",
    ] = Field(description="Where TM sits in analytical architecture.")
    topic_output_form: List[Literal[
        "Full_Proportions", "Reduced_Or_Assigned",
        "Derived_Measure", "Predictive_Feature",
    ]] = Field(default_factory=list, description="How outputs are transformed. Select ALL.")
    analytical_families: List[Literal[
        "Descriptive_Presentation",
        "Statistical_Modeling",
        "Computational_Analysis",
        "Qualitative_Engagement",
    ]] = Field(default_factory=list, description="Method families applied. Select ALL.")
    temporal_analysis: bool = Field(description="Any analysis of change over time?")
    group_comparison: bool = Field(description="Any comparison across named groups?")
    strongest_inferential_claim: Literal[
        "Descriptive", "Associational",
        "Causal_Without_Design", "Causal_With_Design",
        "NOT_APPLICABLE",
    ] = Field(description="Strongest claim. NOT_APPLICABLE if no downstream.")
    validation_approach: Literal[
        "Internal_Metrics_Only", "Discipline_Standards",
        "Both", "None_Reported",
    ] = Field(description="How TM outputs are evaluated for downstream use.")
    validation_evidence: str = Field(default="", description="Verbatim quote ≤60 words.")
    methodological_integration: Literal[
        "Standalone", "Sequential_Pipeline",
        "Mixed_Methods", "Triangulation",
        "NOT_REPORTED",
    ] = Field(description="How TM relates to other methods in the paper.")
    downstream_description: str = Field(
        default="", description="1-3 sentence summary of downstream use.",
    )
    downstream_evidence: str = Field(default="", description="Verbatim quote ≤80 words.")


class TechnicalExtraction(BaseModel):
    data: DataExtraction
    model: ModelExtraction
    K_selection: KSelectionExtraction
    labeling: LabelingExtraction
    downstream: DownstreamExtraction


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — RULES (stable)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_RULES = """You are an expert Research Assistant extracting topic modeling methodology from social science papers.

CORE RULES
1. Extract ONLY what is explicitly stated in the paper text provided.
2. Evidence fields MUST be verbatim quotes from the paper and MUST contain the warrant for the coded value.
3. Do NOT infer, do NOT “best guess”, and do NOT use general domain knowledge.
4. If the paper is silent/unclear for a field, code NOT_REPORTED (or the schema’s specified empty default).

DEFAULTS & EVIDENCE GATING (STRICT)
A) Defaults:
- Strings: "" (empty) unless the schema uses explicit NOT_REPORTED categories; prefer NOT_REPORTED categories when available.
- Lists: []
- Integers: None
- For tri-state fields: use NOT_REPORTED when not explicitly stated.

B) Evidence gating:
- If a categorical field is coded as NOT_REPORTED, its evidence string MUST be "".
- If a categorical field is coded as a specific non-NOT_REPORTED value, its evidence string MUST be non-empty and include the warrant.
- For list fields: include evidence if (and only if) at least one list value is non-empty (e.g., K_final has values; metrics_used not empty).

C) Tri-state requirement for “absence vs silence”:
For these fields, code NOT_REPORTED if the paper does not explicitly mention the issue:
- preprocessing_reported: Yes/No/NOT_REPORTED
- human_judgment_described: Yes/No/NOT_REPORTED
- label_validated: Yes/No/NOT_REPORTED
- has_downstream_analysis: Yes/No/NOT_REPORTED
- temporal_analysis: Yes/No/NOT_REPORTED
- group_comparison: Yes/No/NOT_REPORTED

DATA SOURCE — by genre (what the text IS), not collection method:
- Social_Media = tweets, posts, comments, Reddit, forums, blogs.
- News_Media = newspapers, broadcast transcripts, online news.
- Government_Or_Legal = legislative records, judicial opinions, policy documents, regulations.
- Organizational_Documents = party manifestos, corporate reports, press releases, NGO reports.
- Academic_Text = journal articles, abstracts, conference papers, scholarly corpora.
- Survey_Or_Interview = open-ended survey responses, interview transcripts, focus groups.
- Existing_Dataset = named curated dataset or archive used as the corpus.
- Other / NOT_REPORTED as needed.

UNIT OF ANALYSIS — code PRIMARY unit fed into the topic model (apply in order):
1) Researchers segmented larger text into smaller units → Sub_Document_Segment
   (e.g., sentences, paragraphs, 1,000-word chunks, “speech segments”, “documents split into sections”)
2) Abstract/title/metadata-only fields → Abstract_Or_Title
3) Open-ended survey responses → Open_Ended_Response
4) Transcribed speech/interview text → Speech_Or_Interview
5) Platform communicative acts (tweet, post, comment) → Short_Form_Post
6) Standalone documents (articles, speeches as whole docs, reports) → Long_Form_Document
7) Other / NOT_REPORTED

Special case: If they aggregate many posts into one “document” (e.g., all tweets by user/day/thread),
unit fed into model is Long_Form_Document (constructed), while data_source remains Social_Media.

ALGORITHM
- Code the algorithm family of the FINAL topic model.
- Other_Topic_Model = named topic model not in list (e.g., ETM, SeededLDA, hLDA, TopSBM).
- NOT_REPORTED if it only says “topic modeling” without naming an algorithm or is not a topic model.
- If algorithm_detail would be “topic modeling”, then set algorithm=NOT_REPORTED and algorithm_detail="".

SOFTWARE
- Only code topic-model-estimation packages (not general preprocessing/plotting libraries).
- Examples of non-software: pandas, dplyr, tidyverse, ggplot2.
- If package is not in the list but clearly a TM package, code software=Other and fill software_detail.

K_final
- K_final is a list of FINAL chosen topic numbers ONLY.
- Do NOT include candidate ranges unless a final K is explicitly selected.
- Every K value must appear in K_evidence as a digit or written number.

METADATA_IN_MODEL
Structural = metadata enters the model itself:
- STM prevalence/content covariates, DMR priors, supervised topic models, time as model structure in DTM, etc.
Post_Hoc = metadata used only AFTER modeling:
- LDA/other TM then regression/comparison of topic proportions across groups/covariates.
None = no metadata use described in model or posthoc.
NOT_REPORTED = unclear.

K SELECTION STRATEGY — decision tree (explicit-only)
STEP 1 — Is there explicit evidence of diagnostics/metrics for choosing K?
YES if evidence includes any named metric/diagnostic, such as:
- coherence (c_v, UMass, semantic coherence), exclusivity, FREX, perplexity, held-out likelihood,
  residuals, log-likelihood, AIC/BIC, CaoJuan, Deveaud, Griffiths2004, Arun2010,
  “ldatuning” or “stm built-in diagnostics/fit statistics/model diagnostics” (even if not individually named).
NO if only vague phrases appear:
- “best fit”, “goodness of fit” (unnamed), “model diagnostics” with no implication of quantitative diagnostics,
  “coherent topics” (as an outcome only).

IMPORTANT: If they say “built-in diagnostics/fit statistics/model diagnostics” as the method for choosing K,
treat as metric evidence EVEN IF individual metrics are not listed:
→ selection_strategy = Named_Metric_Only
→ metrics_used should include Multi_Metric_Package (or Other_Metric if more appropriate).

STEP 2 — Is there explicit evidence of human inspection/judgment?
YES if evidence includes:
- “reviewed/inspected topics”, “interpretable/interpretability”, “qualitative assessment”,
  “manual evaluation”, “trial and error” with interpretability reasoning.
NO if it only says “selected/chose” without inspection language.

DECISION:
- metric YES + human YES → Named_Metric_Plus_Human
- metric YES + human NO  → Named_Metric_Only
- metric NO  + human YES → Human_Judgment_Only
- metric NO  + human NO  → Theory_Or_Design_Prior (ONLY if explicit theory/design rationale)
                           else No_Justification_Or_Not_Reported

If a metric is only mentioned as a general possibility but not used, code Human_Judgment_Only
ONLY if human judgment is explicitly described; otherwise No_Justification_Or_Not_Reported.

LABELING (topic interpretation and naming)
labeling_basis (explicit-only):
- Top_Words_Only = labels based only on top/highest-probability words.
- Top_Words_Plus_FREX = mentions FREX/exclusivity words in labeling.
- Words_And_Docs = uses both words and representative documents/excerpts to label.
- Docs_Only = uses documents/excerpts only.
- NOT_REPORTED = labeling basis not described.

labeling_process (explicit-only):
- No_Labeling = explicitly says no labels OR topics referred only by number/top words with no naming.
- Structured_Multi_Coder = 2+ independent coders AND a reported reliability metric (kappa/alpha/% agreement).
- Informal = labeling occurred but without independent-coding + reliability (single author, team discussion, expert consult without reliability).
- NOT_REPORTED = unclear whether/how labeling happened.

label_validated (explicit-only, tri-state):
- Yes only if explicit validation of labels is reported (IRR, expert review, formal evaluation, documented spot-check procedure).
- No only if explicit statement of no validation is made.
- NOT_REPORTED if silent.

labeling_evidence must be authors’ OWN actions, not a literature citation about “best practice”.

DOWNSTREAM (use of topic outputs AFTER modeling)
Definition of “downstream” in this schema:
Downstream includes ANY substantive use of topic outputs beyond merely listing topics/top words without interpretation.
This includes descriptive mapping (plots, prevalence summaries), statistical modeling on topic outputs,
computational analyses using topic vectors, and qualitative engagement with exemplar documents.

has_downstream_analysis (tri-state):
- Yes = any downstream use as defined above is explicitly described.
- No = explicit statement that no downstream use/analysis is done OR only minimal listing with no interpretation.
- NOT_REPORTED = unclear.

topics_as_findings_or_inputs:
- Findings = topics are the main empirical results (described/mapped/interpreted as outcomes).
- Inputs = topic outputs are used as IV/features/filters in other analyses.
- Both = both Findings and Inputs occur.
- Neither = no downstream use (consistent with has_downstream_analysis=No).

epistemic_function (choose what is explicitly claimed; apply priority order):
Priority order if multiple are mentioned:
1) Classification_Proxy (explicit replacement of manual coding)
2) Measurement_Tool (explicit operationalization of a named construct)
3) Data_Reduction (explicit dimensionality reduction/features for another model)
4) Exploratory_For_Theory (explicit theory-building/inductive theorizing)
5) Descriptive_Mapping (default if only “identify themes/topics” without stronger claims)
NOT_REPORTED if no statement of purpose is provided.

topic_model_position:
- Primary_Method = TM is central method driving results.
- One_Of_Several = TM is one method among multiple with comparable weight.
- Preliminary_Step = TM is used mainly to prepare/filter/structure data for other analyses.
- Validation_Or_Robustness = TM used as robustness check or validation tool.
- NOT_REPORTED if unclear.

topic_output_form (select ALL that are explicitly described):
- Full_Proportions = mixture/topic proportions used.
- Reduced_Or_Assigned = dominant topic, binary assignment, thresholded presence.
- Derived_Measure = computed indices from topics (entropy, diversity, polarization, similarity).
- Predictive_Feature = used as features in ML/prediction tasks.

analytical_families (select ALL that apply):
- Descriptive_Presentation = descriptive plots/tables/summaries without formal inference tests.
- Statistical_Modeling = regression, hypothesis tests, time-series inference with coefficients/p-values.
- Computational_Analysis = ML, clustering, PCA/MCA, network analysis using topic outputs.
- Qualitative_Engagement = close reading/case studies using exemplar documents beyond labeling.

temporal_analysis (tri-state):
- Yes only if explicitly analyzing change over time using topic outputs.
- No only if explicitly stating no temporal analysis.
- NOT_REPORTED if silent.

group_comparison (tri-state):
- Yes only if explicitly comparing across named groups using topic outputs.
- No only if explicitly stating no group comparison.
- NOT_REPORTED if silent.

strongest_inferential_claim (code from the paper’s language; evidence required):
- Descriptive = descriptive, interpretive, mapping language only.
- Associational = correlational/associational claims (predict, relate, associated with) without causal identification.
- Causal_Without_Design = causal language without an identification strategy/design.
- Causal_With_Design = causal language WITH explicit design/identification strategy (DiD, IV, RDD, experiment, matching+robustness, etc.).
- NOT_APPLICABLE = no downstream.

validation_approach (explicit-only):
- Internal_Metrics_Only = only TM-native diagnostics/metrics are used as “validation” of outputs.
- Discipline_Standards = external validation aligned with disciplinary standards (hand-coding comparison, expert evaluation, predictive validity against external outcomes, benchmark datasets, formal human evaluation).
- Both = both internal metrics and external/discipline validation.
- None_Reported = no evaluation/validation described.

methodological_integration:
- Standalone = TM is the only analytic method described for results.
- Sequential_Pipeline = TM feeds into another method (e.g., regression/ML).
- Mixed_Methods = TM integrated with qualitative methods in analysis (not just labeling).
- Triangulation = TM compared to another method’s results (dictionary, supervised coding, ethnography, etc.).
- NOT_REPORTED if unclear.
"""


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — EXAMPLES (swappable)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_EXAMPLES = """

FEW-SHOT EXAMPLES BY FIELD
═══════════════════════════

[DATA SOURCE + UNIT OF ANALYSIS]
"we collected 500,000 tweets" → data_source=[Social_Media]; unit_of_analysis=Short_Form_Post
"Reddit comments from r/politics" → data_source=[Social_Media]; unit_of_analysis=Short_Form_Post
"each newspaper article was used as a document" → data_source=[News_Media]; unit_of_analysis=Long_Form_Document
"we analyzed article abstracts from Web of Science" → data_source=[Academic_Text]; unit_of_analysis=Abstract_Or_Title
"we split each speech into 1000-word segments" → data_source=[Government_Or_Legal] or [Other] (if campaign speeches); unit_of_analysis=Sub_Document_Segment
"we concatenated all tweets by user into one document" → data_source=[Social_Media]; unit_of_analysis=Long_Form_Document (constructed doc)

[ALGORITHM]
"we used latent Dirichlet allocation (LDA)" → algorithm=LDA; algorithm_detail="latent Dirichlet allocation"
"we employed the biterm topic model (BTM)" → algorithm=BTM; algorithm_detail="biterm topic model"
"we used topic modeling (Blei et al., 2003)" → algorithm=NOT_REPORTED; algorithm_detail=""
"we employed a semi-automated content analysis" → algorithm=NOT_REPORTED

[SOFTWARE]
"we estimated STM using the stm package in R" → software=stm_R; software_detail=""
"we trained an LDA model in MALLET" → software=MALLET; software_detail=""
"we used tomotopy to fit LDA" → software=Other; software_detail="tomotopy"

[K_final]
"we tested K=5,10,15,20 and selected K=20" → K_final=[20]; K_evidence includes "...selected K=20"
"the optimal K for China Daily was 100 and for CNN was 50" → K_final=[100,50]
"we estimated models with K = 5–50 (see Appendix)" → K_final=[] (no final K explicitly stated)

[METADATA_IN_MODEL]
"we used STM with party affiliation and year as prevalence covariates" → metadata_in_model=Structural
"we estimated an LDA model, then regressed topic proportions on outlet type" → metadata_in_model=Post_Hoc
"we used dynamic topic modeling to capture topic evolution over time" → metadata_in_model=Structural
"we used BERTopic with no covariates or posthoc comparisons reported" → metadata_in_model=None

[K SELECTION STRATEGY]
EVIDENCE: "coherence and exclusivity suggested K=15–25; we inspected topics and chose K=20"
→ selection_strategy=Named_Metric_Plus_Human; metrics_used=[Coherence, Exclusivity_FREX]; human_judgment_described=Yes

EVIDENCE: "the coherence score indicated a twenty-topic solution (Table 3)"
→ selection_strategy=Named_Metric_Only; metrics_used=[Coherence]; human_judgment_described=NOT_REPORTED or No (only if explicitly says no human review)

EVIDENCE: "we compared solutions and selected the most interpretable model with 12 topics"
→ selection_strategy=Human_Judgment_Only; metrics_used=[]; human_judgment_described=Yes

EVIDENCE: "we used STM’s built-in diagnostics to select a 40-topic model"
→ selection_strategy=Named_Metric_Only; metrics_used=[Multi_Metric_Package]; human_judgment_described=NOT_REPORTED (unless inspection is stated)

EVIDENCE: "we opted for a model with eight topics"
→ selection_strategy=No_Justification_Or_Not_Reported; metrics_used=[]; human_judgment_described=NOT_REPORTED

[LABELING — labeling_basis / labeling_process / label_validated]
EVIDENCE: "we labeled each topic based on the ten highest-probability terms"
→ labeling_basis=Top_Words_Only; labeling_process=Informal; label_validated=NOT_REPORTED

EVIDENCE: "labels were assigned after examining top words and reading representative documents"
→ labeling_basis=Words_And_Docs; labeling_process=Informal; label_validated=NOT_REPORTED

EVIDENCE: "four coders independently labeled topics; κ = .72"
→ labeling_basis=NOT_REPORTED (unless basis stated); labeling_process=Structured_Multi_Coder; label_validated=Yes

EVIDENCE: "we did not assign names to topics; we report only top words"
→ labeling_process=No_Labeling; labeling_basis=Top_Words_Only; label_validated=NOT_REPORTED

EVIDENCE (citation only): "topics are defined as highly probable words (Blei & Lafferty, 2009)"
→ labeling_evidence should be "" (do not treat as authors’ action)

[DOWNSTREAM — has_downstream_analysis / topics_as_findings_or_inputs]
Paper only lists topics and top words with no interpretation → has_downstream_analysis=No; topics_as_findings_or_inputs=Neither
Paper plots topic prevalence over time and interprets trends → has_downstream_analysis=Yes; topics_as_findings_or_inputs=Findings
Paper uses topic proportions as predictors of an outcome → has_downstream_analysis=Yes; topics_as_findings_or_inputs=Inputs
Paper interprets topics AND uses them as predictors → has_downstream_analysis=Yes; topics_as_findings_or_inputs=Both

[DOWNSTREAM — epistemic_function]
"we use topics to operationalize political polarization" → epistemic_function=Measurement_Tool
"we identify major themes in public discourse" → epistemic_function=Descriptive_Mapping
"topics serve as features in a classifier predicting policy adoption" → epistemic_function=Data_Reduction (or Classification_Proxy only if replacing manual coding)
"we replace manual coding by using topic assignments as categories" → epistemic_function=Classification_Proxy

[DOWNSTREAM — topic_output_form]
"we used topic proportions (θ) in regression models" → topic_output_form includes Full_Proportions
"each document was assigned its dominant topic" → topic_output_form includes Reduced_Or_Assigned
"we computed topic diversity (entropy) per outlet" → topic_output_form includes Derived_Measure
"topic vectors were used as features for prediction" → topic_output_form includes Predictive_Feature

[DOWNSTREAM — analytical_families]
"Figure 3 shows topic prevalence from 2010–2020" (no test) → analytical_families includes Descriptive_Presentation
"topic 7 is more prevalent among Group A (β=0.15, p<.01)" → analytical_families includes Statistical_Modeling
"we clustered documents using topic proportions" → analytical_families includes Computational_Analysis
"we conducted close reading of exemplar documents from Topic 4" → analytical_families includes Qualitative_Engagement

[DOWNSTREAM — temporal_analysis / group_comparison]
"we track topic prevalence over time" → temporal_analysis=Yes
"we compare topic prevalence across parties" → group_comparison=Yes
(no mention) → temporal_analysis=NOT_REPORTED; group_comparison=NOT_REPORTED

[DOWNSTREAM — strongest_inferential_claim]
"we describe shifting issue attention" → Descriptive
"topic prevalence is associated with vote share" → Associational
"topics caused policy change" (no design described) → Causal_Without_Design
"using difference-in-differences, we estimate the causal effect ..." → Causal_With_Design

[DOWNSTREAM — validation_approach]
"coherence scores confirmed model quality" → validation_approach=Internal_Metrics_Only
"we compared topics to hand-coded categories (85% agreement)" → validation_approach=Discipline_Standards
"we report coherence AND expert review of labels" → validation_approach=Both
(no evaluation described) → validation_approach=None_Reported

[DOWNSTREAM — methodological_integration]
TM only + interpret topics → Standalone
TM then regression/prediction → Sequential_Pipeline
TM + qualitative analysis of exemplars → Mixed_Methods
TM compared with dictionary/supervised coding results → Triangulation
"""


# ═══════════════════════════════════════════════════════════════════════
# Prompt Assembly
# ═══════════════════════════════════════════════════════════════════════

def build_system_prompt(include_examples: bool = True) -> str:
    if include_examples:
        return SYSTEM_PROMPT_RULES + SYSTEM_PROMPT_EXAMPLES
    return SYSTEM_PROMPT_RULES


# ═══════════════════════════════════════════════════════════════════════
# Config & Utilities
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_input_chars: int = 28000
    retries: int = 3
    retry_delay: float = 2.0


def normalize_label(label: str) -> str:
    label = (label or "").strip().upper()
    return {
        "METHOD": "METHODS", "METHODOLOGY": "METHODS",
        "RESULT": "RESULTS", "FINDINGS": "RESULTS",
    }.get(label, label)


def truncate_smart(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    h = int(max_chars * 0.65)
    t = max_chars - h - 50
    return text[:h].rstrip() + "\n\n[...TRUNCATED...]\n\n" + text[-t:].lstrip()


# ═══════════════════════════════════════════════════════════════════════
# Bundle Construction
# ═══════════════════════════════════════════════════════════════════════

def get_sections_by_labels(sections, labels, include_footnotes=False,
                           use_multi_tags=True, min_weight=0.3):
    keep = set(labels)
    parts = []
    for sec in sections:
        primary = normalize_label(
            sec.get("primary_label") or sec.get("major_label", "")
        )
        mp = primary in keep
        mt = (
            any(
                normalize_label(t.get("label", "")) in keep
                and t.get("weight", 0) >= min_weight
                for t in sec.get("tags", [])
            )
            if use_multi_tags
            else False
        )
        if not (mp or mt):
            continue
        header = sec.get("header_raw", "").strip()
        text = (
            sec.get("text_with_footnotes", sec.get("text", ""))
            if include_footnotes
            else sec.get("text", "")
        ).strip()
        if not text:
            continue
        ld = primary or "SECTION"
        if header and header != "NO_HEADER":
            parts.append(f"### {ld}: {header} ###\n{text}")
        else:
            parts.append(f"### {ld} ###\n{text}")
    return "\n\n".join(parts)


def build_bundle(parsed: Dict[str, Any]) -> str:
    sections = parsed.get("sections", [])
    return get_sections_by_labels(
        sections, ["METHODS", "RESULTS"], include_footnotes=True
    )


# ═══════════════════════════════════════════════════════════════════════
# LLM Calling
# ═══════════════════════════════════════════════════════════════════════

def call_structured_extraction(client, cfg, system_prompt, user_text, response_model):
    if not user_text.strip():
        return {"_error": "empty_input", "_model": response_model.__name__}
    user_text = truncate_smart(user_text, cfg.max_input_chars)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Extract information from the following academic paper text:\n\n"
                + user_text
            ),
        },
    ]
    last_error = None
    for attempt in range(1, cfg.retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                messages=messages,
                response_format=response_model,
            )
            pr = completion.choices[0].message.parsed
            if pr is None:
                ref = completion.choices[0].message.refusal
                if ref:
                    return {
                        "_error": f"refusal: {ref}",
                        "_model": response_model.__name__,
                    }
                raise ValueError("Parsed response is None")
            return pr.model_dump()
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt}/{cfg.retries} failed: {e}")
            if attempt < cfg.retries:
                time.sleep(cfg.retry_delay * attempt)
    return {"_error": last_error, "_model": response_model.__name__}


# ═══════════════════════════════════════════════════════════════════════
# K-value Regex Fallback
# ═══════════════════════════════════════════════════════════════════════

K_PATTERNS = [
    r"k\s*=\s*(\d+)",
    r"(\d+)\s*topics?\b",
    r"number\s+of\s+topics?\s*(?:is|was|=|:)?\s*(\d+)",
    r"selected?\s+(\d+)\s+topics?",
    r"chose\s+(\d+)\s+topics?",
    r"(\d+)[- ]topic\s+(?:model|solution)",
]


def extract_k_fallback(text):
    for p in K_PATTERNS:
        for m in re.findall(p, text, re.I):
            if isinstance(m, tuple):
                m = next((x for x in m if x and x.isdigit()), None)
            if m and m.isdigit() and 2 <= int(m) <= 500:
                return int(m)
    return None


def check_and_reextract_k(technical: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    if isinstance(technical, dict) and "_error" not in technical:
        k_list = technical.get("model", {}).get("K_final", [])
        if not k_list:
            fb = extract_k_fallback(full_text)
            if fb:
                technical.get("model", {})["K_final"] = [fb]
                return {"k_value": fb, "k_source": "regex_fallback"}
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

def extract_paper(parsed, client, cfg, system_prompt):
    paper_id = parsed.get("paper_id", "unknown")
    bundle_text = build_bundle(parsed)
    bundle_size = len(bundle_text)

    technical = call_structured_extraction(
        client, cfg, system_prompt, bundle_text, TechnicalExtraction
    )
    reextracted = check_and_reextract_k(technical, bundle_text)

    record = {
        "paper_id": paper_id,
        "metadata": parsed.get("metadata", {}),
        "extraction_model": cfg.model,
        "bundle_size": bundle_size,
        "technical": technical,
    }
    if reextracted:
        record["reextracted"] = reextracted
    return record


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

VALID_DOMAINS = [
    "sociology", "political_science", "psychology",
    "social_sciences_interdisciplinary", "communication",
]


def main():
    ap = argparse.ArgumentParser(
        description="Extract topic modeling methodology (v6 Koch-style schema)"
    )
    ap.add_argument("--domain", required=True, help="Domain or 'all'")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--no-examples", action="store_true",
                    help="Disable few-shot examples for A/B testing")
    args = ap.parse_args()

    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    cfg = ExtractionConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    data_root = Path(args.data_root)

    include_examples = not args.no_examples
    system_prompt = build_system_prompt(include_examples=include_examples)
    logger.info(f"Prompt: {'with' if include_examples else 'WITHOUT'} examples "
                f"({len(system_prompt)} chars, ~{len(system_prompt.split())} words)")

    if args.domain.lower() == "all":
        domains = VALID_DOMAINS
    else:
        if args.domain not in VALID_DOMAINS:
            raise SystemExit(f"Invalid domain: {args.domain!r}. Choose from: {VALID_DOMAINS}")
        domains = [args.domain]

    for domain in domains:
        in_dir = data_root / "tei" / "processed" / domain
        out_dir = data_root / "outputs" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{domain}.extractions.jsonl"

        files = sorted(in_dir.glob("*.processed.json"))
        if not files:
            files = sorted(in_dir.glob("*.json"))
        if not files:
            logger.warning(f"[{domain}] No files in: {in_dir}")
            continue
        if args.limit > 0:
            files = files[: args.limit]

        logger.info(f"[{domain}] {len(files)} papers, model={cfg.model}")

        existing_ids: set = set()
        if args.skip_existing and out_path.exists():
            with out_path.open() as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line).get("paper_id"))
                    except json.JSONDecodeError:
                        pass

        wrote = errors = 0
        mode = "a" if args.skip_existing else "w"

        with out_path.open(mode, encoding="utf-8") as fout:
            for fp in files:
                pid = fp.stem.replace(".processed", "")
                if pid in existing_ids:
                    continue
                try:
                    parsed = json.loads(fp.read_text())
                    record = extract_paper(parsed, client, cfg, system_prompt)
                    record["domain"] = domain
                    record["schema_version"] = "v6"
                    record["examples_enabled"] = include_examples
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()

                    tech = record["technical"]
                    if isinstance(tech, dict) and "_error" not in tech:
                        algo = tech.get("model", {}).get("algorithm", "?")
                        kv = tech.get("model", {}).get("K_final", [])
                        meta = tech.get("model", {}).get("metadata_in_model", "?")
                    else:
                        algo = kv = meta = "?"

                    wrote += 1
                    logger.info(f"[{domain}] [OK] {pid}: algo={algo}, K={kv or '?'}, meta={meta}")
                except Exception as e:
                    errors += 1
                    logger.error(f"[{domain}] [ERROR] {pid}: {e}")
                    fout.write(
                        json.dumps(
                            {"paper_id": pid, "domain": domain, "_error": str(e)},
                            ensure_ascii=False,
                        ) + "\n"
                    )

        logger.info(f"[{domain}] Done: {wrote} ok, {errors} err → {out_path}")
    logger.info("[DONE]")


if __name__ == "__main__":
    main()