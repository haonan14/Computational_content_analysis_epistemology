#!/usr/bin/env python3
"""
Multi-Pass Content Extraction — Asymmetric Routing Architecture (vFinal)
Strict Results + Discussion Routing & CSS Topic Modeling Specific Categories.
Powered by Google GenAI (gemini-3-pro) and Pydantic
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field
from dotenv import load_dotenv

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# SCHEMA 1: TECHNICAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class DataExtraction(BaseModel):
    data_source: List[Literal["Social_Media", "News_Media", "Government_Or_Legal", "Organizational_Documents", "Academic_Text", "Historical_Archive", "Literature_Or_Fiction", "Survey_Or_Interview", "Curated_Dataset", "Other", "NOT_REPORTED"]] = Field(default_factory=list)
    data_source_detail: str = Field(default="")
    corpus_size: Optional[int] = Field(default=None)
    corpus_size_raw: str = Field(default="")
    unit_of_analysis: Literal["Long_Form_Document", "Short_Form_Post", "Speech_Or_Interview", "Open_Ended_Response", "Sub_Document_Segment", "Abstract_Or_Title", "Aggregated_User_Level", "Other", "NOT_REPORTED"]
    unit_of_analysis_all: List[Literal["Long_Form_Document", "Short_Form_Post", "Speech_Or_Interview", "Open_Ended_Response", "Sub_Document_Segment", "Abstract_Or_Title", "Aggregated_User_Level", "Other", "NOT_REPORTED"]] = Field(default_factory=list)
    preprocessing_reported: Literal["Yes", "NOT_REPORTED"]
    preprocessing_details: str = Field(default="", description="Details on stop words, lemmatization, stemming, or n-grams.")

class ModelExtraction(BaseModel):
    algorithm: Literal["LDA", "STM", "NMF", "BERTopic", "Top2Vec", "BTM", "LSA", "CTM", "DTM", "GSDMM", "Seed_Guided_TM", "Ensemble", "Other_Topic_Model", "NOT_REPORTED"]
    algorithm_detail: str = Field(default="")
    software: Literal["stm_R", "MALLET", "gensim", "topicmodels_R", "scikit_learn", "BERTopic_pkg", "lda_R", "quanteda", "keyATM_R", "Other", "NOT_REPORTED"]
    software_detail: str = Field(default="")
    K_final: List[int] = Field(default_factory=list, description="The final number of topics chosen.")
    K_evidence: str = Field(default="")
    metadata_in_model: Literal["Structural_Covariates", "Post_Hoc_Analysis", "None", "NOT_REPORTED"]

class KSelectionExtraction(BaseModel):
    selection_strategy: Literal["Named_Metric_Only", "Named_Metric_Plus_Human", "Human_Judgment_Only", "Theory_Or_Design_Prior", "Heuristic_Or_Rule_Of_Thumb", "NOT_REPORTED"]
    metrics_used: List[Literal["Coherence", "Exclusivity_FREX", "Perplexity", "Held_Out_Likelihood", "Residuals", "Semantic_Diversity", "Multi_Metric_Package", "Other_Metric"]] = Field(default_factory=list)
    human_judgment_described: Literal["Yes", "NOT_REPORTED"]
    selection_evidence: str = Field(default="")

class LabelingExtraction(BaseModel):
    labeling_basis: Literal["Top_Words_Only", "Top_Words_Plus_FREX", "Top_Words_Plus_Documents", "Docs_Only", "External_Knowledge_Or_Theory", "NOT_REPORTED"]
    labeling_process: Literal["Informal_Author_Consensus", "Structured_Multi_Coder", "Automated_LLM", "Human_with_LLM_assisted", "No_Labeling", "NOT_REPORTED"]
    label_validated: Literal["Yes", "NOT_REPORTED"]
    labeling_evidence: str = Field(default="")
    label_validation_evidence: str = Field(default="", description="Look for Word Intrusion, Topic Intrusion, or Inter-coder reliability (Kappa).")

class TechnicalExtraction(BaseModel):
    """Output schema for the Technical Bundle (Methods, SI, Results)"""
    data: DataExtraction
    model: ModelExtraction
    K_selection: KSelectionExtraction
    labeling: LabelingExtraction

# ═══════════════════════════════════════════════════════════════════════
# SCHEMA 2: DOWNSTREAM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class DownstreamComponent(BaseModel):
    component_id: int = Field(description="0-based index in extraction order.")
    tm_link_type: Literal["Variables", "Structure", "Filter"]
    topics_as_findings_or_inputs: Literal["Findings", "Inputs", "Both"]
    epistemic_function: Literal["Measurement_Tool", "Descriptive_Mapping", "Exploratory_For_Theory", "Data_Reduction", "Classification_Proxy", "NOT_REPORTED"]
    epistemic_function_evidence: str = Field(default="")
    topic_model_position: Literal["Primary_Method", "One_Of_Several", "Preliminary_Step", "Validation_Or_Robustness", "NOT_REPORTED"]
    topic_output_form: List[Literal["Full_Proportions", "Reduced_Or_Assigned", "Derived_Measure", "Predictive_Feature"]] = Field(default_factory=list)
    analytical_families: List[Literal["Descriptive_Presentation", "Statistical_Modeling", "Computational_Analysis", "Network_Analysis", "Qualitative_Engagement"]] = Field(default_factory=list)
    temporal_analysis: Literal["Yes", "No", "NOT_REPORTED"]
    group_comparison: Literal["Yes", "No", "NOT_REPORTED"]
    strongest_inferential_claim: Literal["Descriptive", "Associational", "Causal_Without_Design", "Causal_With_Design", "NOT_APPLICABLE"]
    validation_approach: Literal["Internal_Metrics_Only", "Human_Or_External_Validation", "Both", "NOT_REPORTED"]
    validation_evidence: str = Field(default="")
    methodological_integration: Literal["Standalone", "Sequential_Pipeline", "Mixed_Methods", "Triangulation", "NOT_REPORTED"]
    downstream_description: str = Field(default="")
    downstream_evidence: str = Field(default="")

class DownstreamExtractionLLM(BaseModel):
    """The schema the LLM actually sees and populates."""
    has_downstream_analysis: Literal["Yes", "No", "NOT_REPORTED"]
    components: List[DownstreamComponent] = Field(default_factory=list)
    downstream_description: str = Field(default="")

class DownstreamExtractionFull(DownstreamExtractionLLM):
    """Python-only class. Inherits LLM outputs and auto-computes rollups."""
    @computed_field
    def topics_as_findings_or_inputs_rollup(self) -> str:
        if not self.components: return "Neither"
        vals = {c.topics_as_findings_or_inputs for c in self.components}
        if "Both" in vals or ("Findings" in vals and "Inputs" in vals): return "Both"
        if "Findings" in vals: return "Findings"
        return "Inputs"

    @computed_field
    def epistemic_function_rollup(self) -> List[str]:
        vals = {c.epistemic_function for c in self.components if c.epistemic_function != "NOT_REPORTED"}
        return sorted(vals) if vals else (["NOT_REPORTED"] if self.components else [])

    @computed_field
    def topic_model_position_rollup(self) -> List[str]:
        vals = {c.topic_model_position for c in self.components if c.topic_model_position != "NOT_REPORTED"}
        return sorted(vals) if vals else (["NOT_REPORTED"] if self.components else [])

    @computed_field
    def topic_output_form_rollup(self) -> List[str]:
        return sorted({form for c in self.components for form in c.topic_output_form})

    @computed_field
    def analytical_families_rollup(self) -> List[str]:
        return sorted({fam for c in self.components for fam in c.analytical_families})

    @computed_field
    def temporal_analysis_rollup(self) -> str:
        if any(c.temporal_analysis == "Yes" for c in self.components): return "Yes"
        if any(c.temporal_analysis == "No" for c in self.components): return "No"
        return "NOT_REPORTED"

    @computed_field
    def group_comparison_rollup(self) -> str:
        if any(c.group_comparison == "Yes" for c in self.components): return "Yes"
        if any(c.group_comparison == "No" for c in self.components): return "No"
        return "NOT_REPORTED"

    @computed_field
    def strongest_inferential_claim_rollup(self) -> str:
        if not self.components: return "NOT_APPLICABLE"
        hierarchy = {"Causal_With_Design": 5, "Causal_Without_Design": 4, "Associational": 3, "Descriptive": 2, "NOT_APPLICABLE": 1}
        return max(self.components, key=lambda c: hierarchy.get(c.strongest_inferential_claim, 0)).strongest_inferential_claim

    @computed_field
    def methodological_integration_rollup(self) -> List[str]:
        vals = {c.methodological_integration for c in self.components if c.methodological_integration != "NOT_REPORTED"}
        return sorted(vals) if vals else (["NOT_REPORTED"] if self.components else [])

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════

TECHNICAL_PROMPT = """
You are an expert Computational Social Science Research Assistant.

TASK
Extract the topic modeling methodology used in the CURRENT STUDY into the provided schema (TechnicalExtraction).
You are given a technical bundle that may include Methods, Results, Appendix/SI.
Output EXACTLY one JSON object of type TechnicalExtraction. No prose.

────────────────────────────────────────────────────────
A. HARD CONSTRAINTS
────────────────────────────────────────────────────────

A1) OWN-STUDY ONLY (NO CITATION BLEED)
Extract ONLY what the authors did in this paper.
- EXCLUDE: “Following Smith (2019)…”, “Prior studies used STM…”, textbook descriptions.
- INCLUDE: “we…”, “our data…”, “we fit/estimated…”, or passive voice clearly describing this study’s procedure.

A2) EXECUTION SIGNAL DETECTION (CRITICAL)
Treat topic modeling as EXECUTED only if the text contains at least one of:
1) Explicit action: “we fit/ran/estimated/applied/utilized a topic model / LDA / STM / NMF / BERTopic …”
2) Concrete outputs: topic-word lists, topic labels tied to topic IDs, topic prevalence/proportions (theta), or topic-word distributions (beta in topic-model sense), or a stated “number of topics” used.
3) K-as-topics language: “K topics”, “number of topics”, “we set K=… topics”, “a 20-topic model”.

BAILOUT: If execution signals are absent (only generic “themes/topics” or lit review mentions), then:
- model.algorithm = "NOT_REPORTED"
- Apply missing-value mapping for ALL other fields.

A3) MISSING-VALUE MAPPING (USE EXACTLY)
When information is absent OR not explicitly supported:
- Any Literal field with "NOT_REPORTED" option → "NOT_REPORTED"
- K_selection.selection_strategy → "NOT_REPORTED"
- data.data_source → ["NOT_REPORTED"] (SPECIAL: never [])
- Other list fields → []
- Optional[int] fields (corpus_size) → null
- All detail/evidence strings → ""

A4) EVIDENCE FIELDS MUST BE VERBATIM OR EMPTY
The following evidence fields must be EXACT verbatim quotes (copy-paste) from the provided text, or "":
- model.K_evidence
- K_selection.selection_evidence
- labeling.labeling_evidence
- labeling.label_validation_evidence
Never put meta-commentary like “not mentioned” in evidence.

A5) EVIDENCE GATING (LOGICAL OVERRIDES)
Before finalizing JSON, enforce:
- If model.K_final is non-empty → model.K_evidence MUST be non-empty verbatim quote.
  Otherwise set model.K_final=[] and model.K_evidence="".
- If K_selection.selection_strategy != "NOT_REPORTED" → K_selection.selection_evidence MUST be non-empty verbatim quote.
  Otherwise set selection_strategy="NOT_REPORTED" and selection_evidence="".
- If labeling.labeling_basis != "NOT_REPORTED" OR labeling.labeling_process != "NOT_REPORTED"
  → labeling.labeling_evidence MUST be non-empty verbatim quote.
  Otherwise set labeling_basis="NOT_REPORTED", labeling_process="NOT_REPORTED", labeling_evidence="".
- If labeling.label_validated == "Yes" → labeling.label_validation_evidence MUST be non-empty verbatim quote.
  Otherwise set label_validated="NOT_REPORTED" and label_validation_evidence="".

────────────────────────────────────────────────────────
B. FIELD CODING RULES (CLARIFICATIONS TO AVOID COMMON ERRORS)
────────────────────────────────────────────────────────

B1) data.data_source (multi-select list)
Code ALL text genres explicitly stated as data for THIS study.
If none stated → ["NOT_REPORTED"].

Guidance:
- Social_Media: tweets, posts, comments, Reddit, Telegram, forums.
- News_Media: newspaper/news articles, broadcasts, online news.
- Government_Or_Legal: legislation, court decisions, policy documents, parliamentary debates, official statements.
- Organizational_Documents: press releases, NGO reports, company filings, movement org texts.
- Academic_Text: journal abstracts/titles/articles analyzed as text.
- Historical_Archive: archives, digitized historical newspapers/letters, museum archives (explicitly archival/historical).
- Literature_Or_Fiction: novels, fiction corpora, literary texts.
- Survey_Or_Interview: open-ended survey responses, interview transcripts.
- Curated_Dataset: named benchmark/canonical datasets (e.g., “20 Newsgroups”, “GDELT” if explicitly used as dataset).
- Other: clearly described but does not fit above.

B2) data.corpus_size and corpus_size_raw
- corpus_size: integer ONLY if the authors state a numeric count of documents/posts/speeches/responses/messages used in modeling.
  If vague (“millions of tweets”) → corpus_size=null.
- corpus_size_raw: store the exact phrase describing size (can be vague). If none → "".

B3) data.unit_of_analysis (single-choice)
Code the unit fed into the FINAL topic model.
- Aggregated_User_Level ONLY if authors explicitly aggregate multiple posts by user/account into one document.
- Sub_Document_Segment ONLY if authors explicitly split/segment longer texts (sentence/paragraph/section) BEFORE modeling.
If not explicitly stated → "NOT_REPORTED".

B4) data.preprocessing_reported and preprocessing_details
Because schema allows only "Yes" or "NOT_REPORTED":
- preprocessing_reported="Yes" if ANY preprocessing step is explicitly stated (stopwords, stemming, lemmatization, lowercasing, n-grams, pruning, etc.)
- If authors explicitly say “no preprocessing” or “raw text”, still set preprocessing_reported="NOT_REPORTED" (schema constraint),
  but put that statement (briefly) into preprocessing_details.

B5) model.algorithm (single-choice; final chosen model)
Code ONLY the FINAL chosen topic modeling algorithm used for the main results.
Ignore exploratory “we tried X and Y” unless the text clearly identifies which produced the reported final topics/K.

- If executed but algorithm not named → "Other_Topic_Model"
- If the method is not topic modeling (e.g., purely dictionary, sentiment only, word embeddings only) → "NOT_REPORTED"

B6) model.software (single-choice; MUST be used for topic modeling, not just preprocessing)
Code software ONLY if the package/tool is explicitly used to fit/estimate the topic model.
Do NOT code a package that is mentioned only for preprocessing (e.g., stopwords from scikit-learn) or plotting.

- If tool named but not in list → software="Other" and software_detail contains the name (≤30 words).

B7) model.metadata_in_model (FIT-STAGE ONLY)
This field concerns ONLY whether metadata/covariates enter the TOPIC MODEL ESTIMATION.
- Structural_Covariates: metadata enters model estimation (STM prevalence/content covariates; “included covariates in the model”; DMR; keyATM covariates).
- Post_Hoc_Analysis: model estimated without covariates (even if metadata used later elsewhere).
- None: explicitly states no metadata/covariates considered at all.
- NOT_REPORTED: unclear.

B8) K_selection.metrics_used (anti-inference)
ONLY include a metric if it is explicitly named in the text.
Examples of acceptable explicit naming:
- Coherence: “coherence”, “semantic coherence”, “NPMI”, “PMI”, “UMass”, “c_v”
- Exclusivity_FREX: “exclusivity”, “FREX”
- Perplexity: “perplexity”
- Held_Out_Likelihood: “held-out likelihood”, “marginal likelihood”, “log-likelihood (held-out)”
- Residuals: “residuals”
- Semantic_Diversity: “semantic diversity”
- Multi_Metric_Package: explicit mention of a package/procedure that computes multiple K diagnostics (e.g., “ldatuning”).
If the text says “standard diagnostics” but names no metrics and no package → metrics_used=[].

B9) K_selection.selection_strategy (final K decision logic)
Use what they DID (from the quoted evidence), not what is common practice.
- Named_Metric_Only: final K chosen strictly using named metrics (no mention of interpretability/human review).
- Named_Metric_Plus_Human: named metrics + explicit human inspection/interpretability to make final choice.
- Human_Judgment_Only: interpretability/reading topics only; no metrics named.
- Theory_Or_Design_Prior: K chosen before modeling due to external framework (codebook, predefined categories).
- Heuristic_Or_Rule_Of_Thumb: arbitrary heuristic (e.g., “we set K=50” with a rule; or threshold rules).
- NOT_REPORTED: no evidence.

B10) labeling.labeling_basis / labeling_process (avoid “interpretation ≠ labeling method”)
Only code labeling fields if the paper explicitly describes HOW topics were NAMED/ASSIGNED.
Describing what a topic “represents” is not enough unless they also describe the labeling procedure.

- labeling_basis:
  Top_Words_Only vs Top_Words_Plus_FREX vs Top_Words_Plus_Documents vs Docs_Only vs External_Knowledge_Or_Theory.
- labeling_process:
  Informal_Author_Consensus vs Structured_Multi_Coder vs Automated_LLM vs Human_with_LLM_assisted vs No_Labeling.

B11) labeling.label_validated
Because schema allows only "Yes" or "NOT_REPORTED":
- label_validated="Yes" ONLY if authors explicitly describe a validation procedure for topics/labels (word intrusion, topic intrusion, IRR for topic labeling, expert validation, manual checks tied to topics/labels).
- Otherwise label_validated="NOT_REPORTED".

IMPORTANT TRAP:
If IRR/Kappa is reported for a DIFFERENT hand-coding task (not topic labeling), do NOT treat it as label validation.

────────────────────────────────────────────────────────
C. CONFUSION-RESOLVING MINI EXAMPLES (SNIPPETS → FIELD VALUES)
Each example is intentionally short and realistic; apply EXACT mapping.

─────────────────
C1) EXECUTED vs MENTIONED (algorithm should NOT be hallucinated)
(1) Lit review mention ONLY:
Snippet: "Topic models have been widely used to study media frames."
→ model.algorithm="NOT_REPORTED" (NOT executed)

(2) Generic “topics/themes” without model:
Snippet: "We identify key themes in interviews and discuss three topics that emerged."
→ model.algorithm="NOT_REPORTED" (no execution signal; qualitative “topics”)

(3) Executed signal:
Snippet: "We fit a Structural Topic Model to the corpus."
→ model.algorithm="STM"

─────────────────
C2) K as TOPICS vs other k (k-fold, kNN) — avoid false K_final
(1) Topic K:
Snippet: "We set K = 20 topics for the final model."
→ model.K_final=[20]; model.K_evidence=<that snippet>

(2) k-fold CV (NOT topics):
Snippet: "We used 10-fold cross-validation (k=10) to evaluate prediction."
→ model.K_final=[] (this k is NOT topics)

(3) kNN (NOT topics):
Snippet: "We tuned k-nearest neighbors with k=5."
→ model.K_final=[]

─────────────────
C3) SOFTWARE used for modeling vs used only for preprocessing (common error)
(1) Software counts (model fitting):
Snippet: "We estimated LDA using MALLET."
→ model.software="MALLET"

(2) Software counts (model fitting):
Snippet: "We fit STM using the stm package in R."
→ model.software="stm_R"

(3) Software does NOT count if only preprocessing:
Snippet: "We removed stopwords using scikit-learn's English stopword list."
→ model.software="NOT_REPORTED" (scikit-learn not used to fit the topic model here)

─────────────────
C4) metadata_in_model (FIT-STAGE ONLY)
(1) In-model covariates:
Snippet: "We estimated an STM with party and year as prevalence covariates."
→ model.metadata_in_model="Structural_Covariates"

(2) No in-model covariates (even if later comparisons exist):
Snippet: "We fit LDA, then compared topic prevalence by party."
→ model.metadata_in_model="Post_Hoc_Analysis" (covariates not in estimation)

(3) Explicit none:
Snippet: "We did not include metadata or covariates in the topic model."
→ model.metadata_in_model="None"

─────────────────
C5) K SELECTION strategy and metrics_used (anti-inference)
(1) Named metrics only:
Snippet: "We selected K=30 based on perplexity."
→ selection_strategy="Named_Metric_Only"
→ metrics_used=["Perplexity"]
→ selection_evidence=<that snippet>

(2) Named metrics + human inspection:
Snippet: "We examined coherence and exclusivity and then inspected topic interpretability to choose K=20."
→ selection_strategy="Named_Metric_Plus_Human"
→ metrics_used=["Coherence","Exclusivity_FREX"]

(3) Human judgment only:
Snippet: "We reran models with more and fewer topics until the topics were most interpretable."
→ selection_strategy="Human_Judgment_Only"
→ metrics_used=[]

(4) Multi-metric package explicitly named:
Snippet: "We used the ldatuning package to select the number of topics."
→ metrics_used=["Multi_Metric_Package"]
→ selection_strategy="Named_Metric_Only" (unless interpretability is also explicitly mentioned)

(5) Vague diagnostics (DO NOT infer metrics):
Snippet: "We used standard diagnostics and chose K=25."
→ metrics_used=[]
→ selection_strategy="NOT_REPORTED" (unless a named metric or interpretability statement exists)

─────────────────
C6) LABELING basis vs mere interpretation (common false positive)
(1) Top words only:
Snippet: "We labeled topics using the top 10 highest-probability words."
→ labeling_basis="Top_Words_Only"

(2) Top words + documents:
Snippet: "We assigned labels by inspecting top words and reading the most representative documents."
→ labeling_basis="Top_Words_Plus_Documents"

(3) External knowledge/theory:
Snippet: "We labeled topics using categories from our predefined codebook."
→ labeling_basis="External_Knowledge_Or_Theory"

(4) Mere interpretation (NOT enough to code labeling method):
Snippet: "Topic 3 represents healthcare and includes words like hospital, insurance, and patient."
→ labeling_basis="NOT_REPORTED" unless they also describe HOW they labeled

─────────────────
C7) LABELING process (who/how) including LLM variants
(1) Informal author consensus:
Snippet: "The authors discussed and agreed on topic labels."
→ labeling_process="Informal_Author_Consensus"

(2) Structured multi-coder (must be about topic labeling):
Snippet: "Two coders independently labeled topics; Cohen’s kappa was 0.82."
→ labeling_process="Structured_Multi_Coder"

(3) Automated LLM:
Snippet: "We used GPT-4 to generate topic labels from top words and exemplar documents."
→ labeling_process="Automated_LLM"

(4) Human with LLM assisted:
Snippet: "GPT-4 suggested labels, which the research team reviewed and edited."
→ labeling_process="Human_with_LLM_assisted"

(5) No labeling:
Snippet: "We refer to topics by number and report only top words."
→ labeling_process="No_Labeling"

─────────────────
C8) label_validated traps (IRR may be for a different task)
(1) Valid topic/label validation:
Snippet: "We conducted a word intrusion task to validate topic coherence."
→ label_validated="Yes"; label_validation_evidence=<that snippet>

(2) Valid validation via manual checks tied to topics:
Snippet: "We manually inspected documents with topic proportion > 0.10 to confirm relevance."
→ label_validated="Yes"

(3) TRAP: IRR for other coding (NOT topic labeling validation):
Snippet: "Two coders achieved kappa=0.84 in coding discursive agreement."
→ label_validated="NOT_REPORTED" (unless explicitly about labeling/validating topic labels)
"""



DOWNSTREAM_PROMPT = """You are an expert Computational Social Science Research Assistant.
You are evaluating ONLY the Results and Discussion excerpt provided (not the full paper).
Output exactly ONE JSON object matching the DownstreamExtractionLLM schema:
  {has_downstream_analysis, components, downstream_description}
No prose outside JSON.

──────────────────────────────────────────────────────────────
0) HARD CONSTRAINTS (MUST FOLLOW)
──────────────────────────────────────────────────────────────
- Components MUST be topic-model-output-related. Do NOT extract analyses that are unrelated to topic-model outputs.
- Evidence fields must be VERBATIM quotes only. Never write meta-commentary like “not mentioned”.
- QUOTE WINDOW RULE (for pronouns/anchors):
  If a sentence uses pronouns (“it/this/they/these”) or the analytic action is separated from the topic anchor,
  you MAY include up to 2 consecutive sentences in downstream_evidence to keep the topic anchor present.

EVIDENCE GATING:
- Each component MUST have downstream_evidence (verbatim quote window).
- If epistemic_function != "NOT_REPORTED", epistemic_function_evidence MUST be a verbatim quote window.
  If you cannot quote it, set epistemic_function="NOT_REPORTED" and epistemic_function_evidence="".
- If validation_approach != "NOT_REPORTED", validation_evidence MUST be a verbatim quote window.
  If you cannot quote it, set validation_approach="NOT_REPORTED" and validation_evidence="".

──────────────────────────────────────────────────────────────
A) SECTION COVERAGE GUARD (CRITICAL)
──────────────────────────────────────────────────────────────
If the excerpt is likely incomplete/truncated AND lacks any topic-output cues, set:
  has_downstream_analysis="NOT_REPORTED"
  components=[]
  downstream_description=""

Operational definition of “likely incomplete/truncated”:
- The excerpt has no Results/Discussion signals (no findings language; no tables/figures/results verbs),
  AND it has no explicit topic-output cues such as:
  "Topic 1/Topic #", "topics", "topic proportions/prevalence/probabilities",
  "document-topic", "topic assignment/dominant topic", "K topics", "we interpret topic".

If topic-output cues exist, do NOT use NOT_REPORTED; proceed.

──────────────────────────────────────────────────────────────
B) TM-LINK GATE (COMPONENT CREATION RULE)
──────────────────────────────────────────────────────────────
Create a DownstreamComponent ONLY if you can quote a TM-LINK:
a verbatim sentence window that ties topic-model outputs (topics/proportions/assignments/topic IDs)
to an analytic action (interpret, compare, plot, regress, predict, filter/select, typology).

If NO TM-LINK exists:
- If the excerpt only lists topics/top words with no topic-tied interpretation/action → has_downstream_analysis="No", components=[]
- If the excerpt is likely incomplete per Section A → has_downstream_analysis="NOT_REPORTED", components=[]

──────────────────────────────────────────────────────────────
C) LIST-ONLY VS ANALYSIS (NEGATIVE CONSTRAINT)
──────────────────────────────────────────────────────────────
Set has_downstream_analysis="No" ONLY if Results/Discussion ONLY lists topics/top words and does NOT:
  (i) interpret topics as findings (meaning-making, exemplars, quotes tied to topics), AND does NOT
  (ii) analyze trends over time, AND does NOT
  (iii) compare groups/conditions, AND does NOT
  (iv) use topic outputs as variables/features in models/tests.

If authors interpret topics as findings using narrative interpretation/exemplars/quotes tied to topics:
  has_downstream_analysis="Yes"
  analytical_families includes "Qualitative_Engagement"

──────────────────────────────────────────────────────────────
D) MULTIPLE DOWNSTREAM TASKS → MULTIPLE COMPONENTS
──────────────────────────────────────────────────────────────
Extract ALL distinct TM-output-related downstream tasks as separate components.
Split into multiple components when ANY of these changes materially:
- tm_link_type (Variables vs Structure vs Filter)
- topic_output_form (e.g., Full_Proportions vs Reduced_Or_Assigned vs Derived_Measure)
- analytical_families (e.g., descriptive trend plot vs regression vs network)
Merge when it is the same TM-LINK pattern repeated with minor variations.

Number components with component_id = 0, 1, 2, ... in extraction order.

──────────────────────────────────────────────────────────────
E) FIELD CODING GUIDELINES (PER COMPONENT)
──────────────────────────────────────────────────────────────
tm_link_type:
- Variables: topic outputs are used as variables/features (IV/DV/index/features).
- Structure: results are organized/argued by topics (interpretation, exemplar docs, trends by topic).
- Filter: topic model used to select docs/partition/build typology before later analysis.

topics_as_findings_or_inputs:
- Findings: topics are substantive results.
- Inputs: topic outputs feed another analysis.
- Both: within the component, topics are interpreted AND used as variables.

topic_model_position:
- Primary_Method: topic modeling drives the main Results/Discussion argument.
- One_Of_Several: topic modeling is one component among multiple methods.
- Preliminary_Step: topic model is used to filter/partition/construct typology before the main analysis.
- Validation_Or_Robustness: topic model used mainly as a check.
- NOT_REPORTED: cannot be supported from excerpt.

topic_output_form (select all that apply within the component):
- Full_Proportions: continuous topic proportions/probabilities used/reported.
- Reduced_Or_Assigned: dominant topic assignment or thresholded membership.
- Derived_Measure: index/score built from topic outputs.
- Predictive_Feature: topics used as features for prediction/classification.

analytical_families (select all that apply within the component):
- Descriptive_Presentation: descriptive comparisons/plots by topic, prevalence summaries, trend plots.
- Statistical_Modeling: regression, hypothesis tests, IV/DiD/experiments (if shown in excerpt).
- Computational_Analysis: algorithmic pipelines beyond standard regression (e.g., clustering on topic vectors).
- Network_Analysis: network measures/graphs derived from topic-related outputs.
- Qualitative_Engagement: exemplar documents/quotes used to interpret topics.

temporal_analysis / group_comparison (boundary is mechanical):
- temporal_analysis="Yes" ONLY if the excerpt explicitly mentions time trend/comparison (years, waves, pre/post, “over time”, time plot).
- temporal_analysis="No" if the excerpt makes NO mention of time trend/comparison.
- temporal_analysis="NOT_REPORTED" ONLY if the excerpt is visibly cut off exactly when introducing time (e.g., ends mid-sentence after “over time…”).

- group_comparison="Yes" ONLY if the excerpt explicitly compares groups/conditions/outlets/parties/etc.
- group_comparison="No" if the excerpt makes NO mention of group comparison.
- group_comparison="NOT_REPORTED" ONLY if the excerpt is visibly cut off exactly when introducing a comparison.

strongest_inferential_claim:
- Descriptive: mapping/trends without correlational modeling.
- Associational: association modeling without causal design.
- Causal_Without_Design: causal language but no clear identification strategy demonstrated here.
- Causal_With_Design: a clear identification strategy is demonstrated (IV, DiD, RDD, experiment) in the excerpt.
- NOT_APPLICABLE: use ONLY when tm_link_type="Filter" AND the component itself makes no claim.

validation_approach:
- Code ONLY if explicitly stated in this excerpt.
- Otherwise set validation_approach="NOT_REPORTED" and validation_evidence="".

methodological_integration:
- Standalone: topics presented/interpreted with minimal additional analysis.
- Sequential_Pipeline: topic model outputs feed later modeling/measurement.
- Mixed_Methods: explicit quant+qual integration around topics.
- Triangulation: independent methods used to corroborate findings.
- NOT_REPORTED if cannot be supported.

──────────────────────────────────────────────────────────────
F) MINIMUM EVIDENCE REQUIREMENT (PER COMPONENT)
──────────────────────────────────────────────────────────────
Each component MUST include downstream_evidence with:
- a topic anchor (topic(s), topic proportions, assignments, dominant topic, etc.), AND
- an analytic action (interpret/compare/plot/regress/predict/select/filter/typology).

If you cannot supply such a quote window, do NOT create the component.

──────────────────────────────────────────────────────────────
G) FEW-SHOT MICRO EXAMPLES
──────────────────────────────────────────────────────────────
1) LIST-ONLY (No downstream):
Quote: "Table 1 lists the top words for each topic."
→ has_downstream_analysis="No", components=[]

2) FINDINGS via qualitative interpretation (Structure):
Quote: "Topic 3 captures narratives of distrust; exemplar posts describe 'government lies'."
→ tm_link_type="Structure"; topics_as_findings_or_inputs="Findings"
→ analytical_families includes "Qualitative_Engagement"

3) INPUTS via regression (Variables) + pronoun window:
Quote sentence 1: "Topic 7 is the immigration topic."
Quote sentence 2: "It is then regressed on party affiliation (β=..., p<...)."
→ downstream_evidence may include BOTH sentences (2-sentence window)
→ tm_link_type="Variables"; analytical_families includes "Statistical_Modeling"
→ topic_output_form includes "Full_Proportions" (if proportions mentioned) or "Reduced_Or_Assigned" (if dominant-topic used)

4) FILTER / typology construction (Filter) with NOT_APPLICABLE:
Quote: "We select documents with Topic 7 > 0.2 for close reading."
→ tm_link_type="Filter"; strongest_inferential_claim="NOT_APPLICABLE"
→ analytical_families may include "Qualitative_Engagement"
"""



# ═══════════════════════════════════════════════════════════════════════
# PARSING & I/O UTILITIES (Restored from your v3 script)
# ═══════════════════════════════════════════════════════════════════════

def normalize_label(label: str) -> str:
    label = (label or "").strip().upper()
    return {
        "METHOD": "METHODS", "METHODOLOGY": "METHODS",
        "RESULT": "RESULTS", "FINDINGS": "RESULTS",
        "DISCUSSION": "DISCUSSION", "CONCLUSION": "DISCUSSION",
        "APPENDIX": "APPENDIX"
    }.get(label, label)

def get_sections_by_labels(sections, labels, include_footnotes=True, use_multi_tags=True, min_weight=0.3):
    """Restored exact parsing logic from v3 to map your JSON schema."""
    keep = set(labels)
    parts = []
    
    for sec in sections:
        primary = normalize_label(sec.get("primary_label") or sec.get("major_label", ""))
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
        parts.append(f"### {ld}: {header} ###\n{text}" if header and header != "NO_HEADER" else f"### {ld} ###\n{text}")
            
    return "\n\n".join(parts)

def truncate_smart(text: str, max_chars: int = 40000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    h = int(max_chars * 0.65)
    t = max_chars - h - 50
    return text[:h].rstrip() + "\n\n[...TRUNCATED TO PRESERVE API LIMITS...]\n\n" + text[-t:].lstrip()

# ==============================================================================
# 5. CORE LLM CALLING
# ==============================================================================

def call_gemini_with_retry(client: genai.Client, model_name: str, prompt: str, schema: Any, bundle: str, retries: int = 3, backoff: int = 25) -> Dict:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=bundle,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0
                ),
            )
            
            # Use Pydantic to strictly parse the returned JSON string (safer than response.parsed)
            if response.text:
                parsed_obj = schema.model_validate_json(response.text)
                return parsed_obj.model_dump()
            return {"_error": "Response text is empty"}

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                return {"_error": str(e)}
    return {"_error": "Max retries exceeded"}

def extract_paper(parsed: Dict, client: genai.Client, model_name: str) -> Dict:
    paper_id = parsed.get("paper_id", "unknown")
    sections = parsed.get("sections", [])
    
    # --- 1. TECHNICAL BUNDLE ---
    tech_labels = ["METHODS", "APPENDIX", "RESULTS"]
    technical_bundle = truncate_smart(get_sections_by_labels(sections, tech_labels))
    technical_result = {}
    
    if technical_bundle:
        technical_result = call_gemini_with_retry(
            client, model_name, TECHNICAL_PROMPT, TechnicalExtraction, technical_bundle
        )
    else:
        technical_result = {"_error": "empty_bundle"}

    # --- 2. DOWNSTREAM BUNDLE ---
    downstream_labels = ["RESULTS", "DISCUSSION"]
    downstream_bundle = truncate_smart(get_sections_by_labels(sections, downstream_labels))
    downstream_result = {}
    
    if downstream_bundle:
        # Step A: Extract using the LLM base schema
        llm_res = call_gemini_with_retry(
            client, model_name, DOWNSTREAM_PROMPT, DownstreamExtractionLLM, downstream_bundle
        )
        
        # Step B: Upgrade to Full schema to inject computed properties (if extraction was successful)
        if "_error" not in llm_res:
            try:
                full_obj = DownstreamExtractionFull(**llm_res)
                downstream_result = full_obj.model_dump()
            except Exception as e:
                logger.error(f"Error computing downstream properties for {paper_id}: {e}")
                downstream_result = {"_error": f"Rollup Error: {str(e)}"}
        else:
            downstream_result = llm_res
    else:
        downstream_result = {"_error": "empty_bundle"}

    # Return final payload
    return {
        "paper_id": paper_id,
        "metadata": parsed.get("metadata", {}),
        "extraction_model": model_name,
        "technical_bundle_size": len(technical_bundle),
        "downstream_bundle_size": len(downstream_bundle),
        "technical": technical_result,
        "downstream": downstream_result
    }

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

VALID_DOMAINS = ["sociology", "political_science", "psychology", "social_sciences_interdisciplinary", "communication"]

def main():
    ap = argparse.ArgumentParser(description="Extract topic modeling methodology via Gemini Asymmetric Routing")
    ap.add_argument("--domain", required=True, help="Domain or 'all'")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment variables")

    client = genai.Client(api_key=api_key)
    data_root = Path(args.data_root)

    domains = VALID_DOMAINS if args.domain.lower() == "all" else [args.domain]
    if args.domain.lower() != "all" and args.domain not in VALID_DOMAINS:
        raise SystemExit(f"Invalid domain: {args.domain!r}. Choose from: {VALID_DOMAINS}")

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

        logger.info(f"[{domain}] {len(files)} papers, model={args.model}")

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
                    parsed = json.loads(fp.read_text(encoding="utf-8"))
                    record = extract_paper(parsed, client, args.model)
                    record["domain"] = domain
                    record["schema_version"] = "v_gemini_asymmetric"
                    
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()

                    tech = record.get("technical", {})
                    if "_error" not in tech:
                        algo = tech.get("model", {}).get("algorithm", "?")
                        kv = tech.get("model", {}).get("K_final", [])
                        tbs = record.get("technical_bundle_size", 0)
                        dbs = record.get("downstream_bundle_size", 0)
                        logger.info(f"[{domain}] [OK] {pid}: algo={algo}, K={kv or '?'}, Bundles=({tbs}c, {dbs}c)")
                        wrote += 1
                    else:
                        logger.warning(f"[{domain}] [WARN] {pid} Extracted with Error: {tech.get('_error')}")
                        errors += 1
                    time.sleep(20) # Waits 20 seconds before processing the next paper
                except Exception as e:
                    errors += 1
                    logger.error(f"[{domain}] [ERROR] {pid}: {e}")
                    fout.write(json.dumps({"paper_id": pid, "domain": domain, "_error": str(e)}, ensure_ascii=False) + "\n")

        logger.info(f"[{domain}] Done: {wrote} ok, {errors} err → {out_path}")
    logger.info("[DONE]")

if __name__ == "__main__":
    main()