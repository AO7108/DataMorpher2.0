# Location: src/parser/command_parser.py

import json
import re
import logging
from typing import Any, Dict, List, Optional
from llama_cpp import Llama
from src.config import config


class CommandParser:
    """
    Parses free-text user input into a normalized command dict for the pipeline.
    Public API: parse(user_input: str) -> Dict[str, Any] | None
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # --- NEW: LOAD THE LLM ON INITIALIZATION ---
        self.llm = None
        try:
            llm_config = config.get("llm", {})
            model_path = llm_config.get("model_path")
            if model_path:
                self.logger.info(f"Loading LLM from path: {model_path}")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=llm_config.get("n_ctx", 4096),
                    n_threads=llm_config.get("n_threads", None), 
                    verbose=False
                )
                self.logger.info("LLM loaded successfully.")
            else:
                self.logger.warning("LLM model_path not found in config. Parser will only use heuristics.")
        except Exception as e:
            self.logger.error(f"Failed to load LLM. Error: {e}. Parser will only use heuristics.")
        # --- END NEW SECTION ---
        # Processors we support
        self.processor_keys = {"augment", "curate", "balance", "split", "score"}

        # Modality synonyms
        self.modality_map = {
            "image": {"image", "images", "picture", "pictures", "photo", "photos", "pics"},
            "audio": {"audio", "audios", "sound", "sounds", "clip", "clips"},
            "text": {"text", "texts", "articles", "documents"},
        }

        # Patterns
        self.re_count = re.compile(r"\b(\d{1,4})\b")
        self.re_processor_words = re.compile(
            r"\b(augment|augmentation|curate|curation|balance|balancing|split|split\s*into|score|scoring)\b",
            re.IGNORECASE,
        )
        self.re_image_words = re.compile(r"\b(image|images|picture|pictures|photo|photos|pics)\b", re.IGNORECASE)
        self.re_make_words = re.compile(r"\b(create|make|get|generate|need|want|fetch|collect)\b", re.IGNORECASE)

        # Class split detection: "A vs B", plus "equal/balanced"
        self.re_vs = re.compile(r"(\b[\w-]+\b(?:\s+\b[\w-]+\b){0,2})\s+(?:vs|versus)\s+(\b[\w-]+\b(?:\s+\b[\w-]+\b){0,2})", re.IGNORECASE)
        self.re_equal = re.compile(r"\b(equal|balanced|50/50|same number)\b", re.IGNORECASE)

    # -------------------- Public API --------------------

    def parse(self, user_input: str) -> Optional[Dict[str, Any]]:
        # --- ADD THIS NEW CHECK AT THE TOP ---
        if not user_input or not user_input.strip():
            self.logger.warning("Received empty or whitespace-only input. Rejecting.")
            return {
                "intent": "clarify",
                "clarification": "The user request was empty. Please provide a valid command.",
                "subject": None # Ensure other fields are None to avoid downstream errors
            }
        # --- END OF NEW CHECK ---
        original = user_input or ""
        text = original.strip()

        hints = self._make_hints(text)
        prompt = self._build_prompt(text, hints)
        raw = self._call_llm(prompt)

        data = self._extract_first_json(raw)
        if data is None or not isinstance(data, dict):
            self.logger.warning("Parser failed to extract JSON. Falling back to heuristics.")
            data = {}

        normalized = self._normalize_fields(data, original, hints)

        if normalized.get("intent") != "create_dataset":
            return {
                "intent": normalized.get("intent", "unsupported"),
                "modality": normalized.get("modality"),
                "subject": normalized.get("subject"),
                "attributes": normalized.get("attributes", []),
                "task": normalized.get("task"),
                "count": normalized.get("count"),
                "processors": normalized.get("processors"),
                "clarification": normalized.get("clarification")
                or "This app builds datasets. Try: 'create 20 images of <subject>'.",
                "original_prompt": original,
                "class_split": normalized.get("class_split"),
                "class_labels": normalized.get("class_labels"),
                "balance_hint": normalized.get("balance_hint"),
            }

        return normalized

    # -------------------- Prompting & LLM --------------------

    def _build_prompt(self, user_text: str, hints: Dict[str, Any]) -> str:
        schema = """
Output ONLY a single JSON object with this exact schema and keys:
{
  "intent": "create_dataset" | "analyze" | "export" | "clarify" | "unsupported",
  "modality": "image" | "audio" | "text" | null,
  "subject": string | null,
  "attributes": string[] | [],
  "task": "retrieval" | "generation" | "classification" | "analysis" | null,
  "count": integer | null,
  "processors": string[] subset of ["augment","curate","balance","split","score"] | null,
  "clarification": string | null,
  "original_prompt": string,
  "class_split": boolean | null,
  "class_labels": [string,string] | null,
  "balance_hint": "equal" | "balanced" | null
}
Normalization rules:
- Use lowercase for intent, modality, task, processors, and attributes tokens.
- Keep subject titlecase (e.g., "Tzuyu").
- If user expresses "A vs B", set class_split=true and class_labels=["A","B"].
- If user asks for equal/balanced classes, set balance_hint accordingly.
- Do NOT include any text outside the JSON object.
"""
        few_shot = [
            {
                "input": "create 60 pictures of tzuyu smiling vs non smiling, equal; curate balance split score",
                "output": {
                    "intent": "create_dataset",
                    "modality": "image",
                    "subject": "Tzuyu",
                    "attributes": ["smiling"],
                    "task": "retrieval",
                    "count": 60,
                    "processors": ["curate", "balance", "split", "score"],
                    "clarification": None,
                    "original_prompt": "create 60 pictures of tzuyu smiling vs non smiling, equal; curate balance split score",
                    "class_split": True,
                    "class_labels": ["smiling", "non smiling"],
                    "balance_hint": "equal"
                },
            },
            {
                "input": "create 20 pictures of tzuyu",
                "output": {
                    "intent": "create_dataset",
                    "modality": "image",
                    "subject": "Tzuyu",
                    "attributes": [],
                    "task": "retrieval",
                    "count": 20,
                    "processors": [],
                    "clarification": None,
                    "original_prompt": "create 20 pictures of tzuyu",
                    "class_split": False,
                    "class_labels": None,
                    "balance_hint": None
                },
            },
        ]
        hints_block = json.dumps(hints, ensure_ascii=False)
        prompt = (
            "You are a strict command parser for a dataset-building pipeline.\n"
            + schema
            + "\nExamples:\n"
            + "\n".join(
                f"Input: {ex['input']}\nOutput:\n{json.dumps(ex['output'], ensure_ascii=False)}"
                for ex in few_shot
            )
            + "\n\nHINTS: "
            + hints_block
            + "\n\nNow parse this input into the JSON schema (no extra text):\n"
            f"{user_text}"
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        # If the LLM failed to load, return empty to trigger heuristics
        if not self.llm:
            return ""

        self.logger.info("Sending prompt to local LLM...")
        try:
            llm_config = config.get("llm", {})
            output = self.llm(
                prompt,
                max_tokens=llm_config.get("max_tokens", 1024),
                temperature=llm_config.get("temperature", 0.1),
                stop=["}"] # Stop generating once the JSON object is complete
            )
            response_text = output["choices"][0]["text"]
            # The model often forgets the closing brace, so we add it back
            return response_text + "}"
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return ""
    # --- END MODIFIED METHOD ---

    # -------------------- Extraction & Normalization --------------------

    def _extract_first_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        # Strip code fences or headings if present
        cleaned = re.sub(r"^``````", "", text, flags=re.DOTALL | re.MULTILINE)
        cleaned = re.sub(r"^#+\s.*$", "", cleaned, flags=re.MULTILINE)
        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = cleaned[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    def _normalize_fields(self, data: Dict[str, Any], original: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        intent = (data.get("intent") or "").strip().lower()
        modality = self._normalize_modality(data.get("modality"))
        subject = self._clean_subject(data.get("subject"))
        attributes = self._normalize_attributes(data.get("attributes") or [])
        task = (data.get("task") or "").strip().lower() or ("retrieval" if intent == "create_dataset" else None)
        count = self._extract_count_field(data.get("count"), original)
        processors = self._extract_processors_field(data.get("processors"), original)
        clarification = data.get("clarification")

        class_split = data.get("class_split")
        class_labels = data.get("class_labels")
        balance_hint = data.get("balance_hint")

        # --- CORRECTED SAFETY RAIL ---
        # Check if a 'vs' keyword is actually present in the user's prompt
        has_vs_keyword = " vs " in original.lower() or " vs. " in original.lower() or " versus " in original.lower()
        
        # If a 'vs' keyword is present, always ignore the LLM's labels and
        # force our robust local heuristics to run. This handles multi-word labels correctly.
        if has_vs_keyword:
            # Setting class_split to None triggers the heuristic block below.
            class_split = None
            class_labels = None
        # Else, if there's no 'vs' keyword but the LLM hallucinated a split, reset it.
        elif class_split is True and not has_vs_keyword:
            self.logger.warning("LLM hallucinated a class_split where no 'vs' keyword was found. Resetting.")
            class_split = None 
            class_labels = None
        # --- END OF NEW LOGIC ---

        # Heuristics if LLM omitted fields or if we reset due to hallucination
        if class_split is None:
            split_keyword = None
            if " vs. " in original.lower(): split_keyword = " vs. "
            elif " vs " in original.lower(): split_keyword = " vs "
            elif " versus " in original.lower(): split_keyword = " versus "

            if split_keyword:
                split_point = original.lower().find(split_keyword)
                left_half = original[:split_point]
                right_half = original[split_point + len(split_keyword):]

                label1_raw = re.sub(r".*?\b(?:of|for|about)\b", "", left_half, flags=re.IGNORECASE).strip()
                if not label1_raw:
                    label1_raw = left_half.strip()

                label2_raw = right_half.split(',')[0].strip()
                
                subj_guess = self._guess_subject(original)
                if subj_guess:
                    subj_low = subj_guess.lower()
                    if label1_raw.lower() != subj_low:
                        label1_raw = re.sub(rf"\b{re.escape(subj_low)}\b", "", label1_raw, flags=re.IGNORECASE).strip()
                    if label2_raw.lower() != subj_low:
                        label2_raw = re.sub(rf"\b{re.escape(subj_low)}\b", "", label2_raw, flags=re.IGNORECASE).strip()

                label1 = self._clean_label(label1_raw)
                label2 = self._clean_label(label2_raw)

                if label1 and label2:
                    class_split = True
                    class_labels = [label1, label2]

            if self.re_equal.search(original):
                balance_hint = "equal"

        if not intent:
            intent = "create_dataset" if hints.get("prefer_create_dataset") else "unsupported"
        if not modality and hints.get("likely_modality"):
            modality = hints["likely_modality"]
        if not subject:
            subject = self._guess_subject(original)
        if not attributes:
            attributes = self._extract_attributes_from_text(original)

        if isinstance(class_labels, list) and len(class_labels) == 2:
            class_labels = [self._clean_label(l) for l in class_labels]
            if not class_labels[0] or not class_labels[1]:
                class_labels = None

        return {
            "intent": intent, "modality": modality, "subject": subject, "attributes": attributes, "task": task,
            "count": count, "processors": processors, "clarification": clarification, "original_prompt": original,
            "class_split": bool(class_split) if class_split is not None else False,
            "class_labels": class_labels if isinstance(class_labels, list) and len(class_labels) == 2 else None,
            "balance_hint": balance_hint if balance_hint in ("equal", "balanced") else None,
        }

    # -------------------- Helpers --------------------

    def _make_hints(self, text: str) -> Dict[str, Any]:
        prefer_create = bool(self.re_make_words.search(text) and self.re_image_words.search(text))
        likely_modality = "image" if self.re_image_words.search(text) else None
        return {"prefer_create_dataset": prefer_create, "likely_modality": likely_modality}

    def _normalize_modality(self, value: Any) -> Optional[str]:
        if not value:
            return None
        v = str(value).strip().lower()
        for key, variants in self.modality_map.items():
            if v == key or v in variants:
                return key
        return v if v in {"image", "audio", "text"} else None

    def _clean_subject(self, value: Any) -> Optional[str]:
        if not value:
            return None
        s = str(value).strip()
        s = re.sub(r"[\[\]{}()#@*^~`_+=|\\/]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) <= 80:
            s = s.title()
        return s or None

    def _clean_label(self, value: str) -> str:
        s = value.strip()
        s = re.sub(r"\s+", " ", s)
        # Remove stray punctuation at ends
        s = re.sub(r"^[,;:.-]+|[,;:.-]+$", "", s).strip()
        return s

    def _extract_count_field(self, value: Any, text: str) -> Optional[int]:
        if isinstance(value, int):
            return value
        m = self.re_count.search(text)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 5000:
                    return n
            except Exception:
                pass
        return None

    def _extract_processors_field(self, value: Any, text: str) -> Optional[List[str]]:
        items: List[str] = []
        if isinstance(value, list):
            items.extend([str(v).lower().strip() for v in value if v])
        found = self.re_processor_words.findall(text)
        for f in found:
            token = f.lower()
            if "augment" in token:
                items.append("augment")
            elif "curate" in token:
                items.append("curate")
            elif "balance" in token:
                items.append("balance")
            elif "split" in token:
                items.append("split")
            elif "score" in token:
                items.append("score")
        seen = set()
        norm = []
        for p in items:
            if p in self.processor_keys and p not in seen:
                seen.add(p)
                norm.append(p)
        return norm or None

    def _extract_attributes_from_text(self, text: str) -> List[str]:
        tokens = []
        low = text.lower()
        if "face" in low or "faces" in low:
            tokens.append("face")
        if "smile" in low or "smiling" in low:
            tokens.append("smiling")
        if "portrait" in low or "portraits" in low:
            tokens.append("portrait")
        # dedupe
        out = []
        seen = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _guess_subject(self, text: str) -> Optional[str]:
        t = text.strip()

        # Regex to find a subject that is followed by specific keywords or the end of the string
        # The (?=...) is a "lookahead" that doesn't consume the text
        stop_keywords = r"(?=\s+vs|\s+versus|,|;|:|$|\b(smiling|happy|sad|angry)\b)"

        
        # Define a non-capturing group for common articles to ignore them
        articles = r"(?:\b(?:a|an|the)\s+)?"

        # 1) “… of <subject> …” - Updated to ignore leading articles
        m = re.search(r"\bof\s+" + articles + r"([A-Za-z][A-Za-z0-9\s\-._]{1,60}?)" + stop_keywords, t, flags=re.IGNORECASE)
        cand = m.group(1) if m else None

        # 2) “… pictures/images/photos of <subject> …” - Updated to ignore leading articles
        if not cand:
            m2 = re.search(
                r"\b(?:create|make|get|generate|need|want|fetch|collect)?\s*\d*\s*(?:images?|pictures?|photos?|pics?)\s+of\s+" + articles + r"([A-Za-z][A-Za-z0-9\s\-._]{1,60}?)" + stop_keywords,
                t,
                flags=re.IGNORECASE,
            )
            cand = m2.group(1) if m2 else None

        # 3) Fallback: first capitalized phrase before a common attribute or "vs"
        if not cand:
            m3 = re.search(
                r"\b([A-Z][A-Za-z0-9\-. ]{1,60}?)" + stop_keywords,
                t,
            )
            cand = m3.group(1) if m3 else None

        if cand:
            # Clean any remaining processor words from the candidate
            cand = re.sub(self.re_processor_words, "", cand).strip()
            cand = re.sub(r"\s+", " ", cand).strip()
            if cand:
                return self._clean_subject(cand)
        return None

    def _normalize_attributes(self, attrs: Any) -> List[str]:
        if not attrs:
            return []
        out = []
        seen = set()
        for a in attrs:
            token = str(a).lower().strip()
            if token and token not in seen:
                seen.add(token)
                out.append(token)
        return out
