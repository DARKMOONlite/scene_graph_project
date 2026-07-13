"""Language standardisation for scene graph labels.

Uses spaCy for lemmatisation and WordNet for synonym / hypernym resolution
so that labels from different scene-graph generators are mapped to a
consistent canonical vocabulary before fusion.
"""

from __future__ import annotations
from uuid import UUID

#! for some reason importing spacy is causing errors in some scripts that dont use it, so we import it lazily in the Standardiser class
# try:
#     import os
#     os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import spacy
# except ImportError:
#     spacy = None
    # print("spaCy is not installed. Please install spaCy to use the Standardiser class.")
from scene_graph_project.scene_graph_fusion.pipeline.wordnet import (
    get_synsets_for_noun,
    build_hierarchy,
    wup_confidence,
)
from scene_graph_project.scene_graph_fusion.pipeline.models import SceneGraph, SceneObject

# ---------------------------------------------------------------------------
# Built-in synonym map – covers the most common cross-detector divergences.
# Keys are variant spellings; values are the canonical form.
# ---------------------------------------------------------------------------
_DEFAULT_SYNONYMS: dict[str, str] = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "guy": "person",
    "lady": "person",
    "people": "person",
    "kid": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "tv": "television",
    "sofa": "couch",
    "cellphone": "cell phone",
    "spectacles": "glasses",
    "auto": "car",
}

_DEFAULT_PREDICATE_SYNONYMS: dict[str, str] = {
    "on top of": "on",
    "above": "on",
    "atop": "on",
    "beneath": "under",
    "below": "under",
    "beside": "next to",
    "alongside": "next to",
    "near": "next to",
    "in front of": "in front of",
    "behind": "behind",
    "inside": "in",
    "within": "in",
    "holding": "holding",
    "carrying": "holding",
    "wearing": "wearing",
    "has": "has",
    "with": "has",
}


class Standardiser:
    """Normalise object labels and relationship predicates across scene graphs.

    Pipeline applied per label:
        1. Lower-case + strip whitespace
        2. Explicit synonym mapping (user-supplied + built-in defaults)
        3. spaCy lemmatisation (``dogs`` → ``dog``)
        4. WordNet-based merging: if two labels share a close enough hypernym
           (above ``wup_threshold``), they are mapped to the same canonical form.

    The canonical vocabulary is built lazily as graphs are processed.
    """

    def __init__(
        self,
        *,
        synonym_map: dict[str, str] | None = None,
        predicate_synonym_map: dict[str, str] | None = None,
        wup_threshold: float = 0.8,
        spacy_model: str = "en_core_web_sm",
        blacklist: set[str] | None = None,
    ):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' not found. Please install it with 'python -m spacy download {spacy_model}'."
            )
        self.wup_threshold = wup_threshold
        self.blacklisted_labels = {label.strip().lower() for label in (blacklist or set())}

        # label → canonical label
        self._label_map: dict[str, str] = {}
        self._predicate_map: dict[str, str] = {}

        # seed with synonym tables
        synonyms = {**_DEFAULT_SYNONYMS, **(synonym_map or {})}
        for variant, canonical in synonyms.items():
            self._label_map[variant.lower()] = canonical.lower()

        pred_synonyms = {**_DEFAULT_PREDICATE_SYNONYMS, **(predicate_synonym_map or {})}
        for variant, canonical in pred_synonyms.items():
            self._predicate_map[variant.lower()] = canonical.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def standardise(self, graph: SceneGraph) -> SceneGraph:
        """Standardise all labels and predicates in *graph* **in-place** and return it."""
        for obj in graph.objects:
            obj.canonical_label = self.canonicalise_label(obj.label)
        for rel in graph.relationships:
            rel.canonical_predicate = self.canonicalise_predicate(rel.predicate)
        return graph
    
    def blacklist(self, graph: SceneGraph) -> SceneGraph:
        """Remove blacklisted objects and their incident relationships in-place."""
        removed_object_ids: set[UUID] = set()
        remaining_objects = []

        for obj in graph.objects:
            if obj.canonical_label in self.blacklisted_labels or obj.label in self.blacklisted_labels:
                removed_object_ids.add(obj.uid)
            else:
                remaining_objects.append(obj)

        graph.objects = remaining_objects
        graph.relationships = [
            rel
            for rel in graph.relationships
            if rel.subject_uid not in removed_object_ids and rel.object_uid not in removed_object_ids
        ]
        return graph
    
    def canonicalise_label(self, label: str) -> str:
        """Return the canonical form of an object *label*."""
        label = label.strip().lower()
        if label in self._label_map:
            return self._label_map[label]

        lemma = self._lemmatise(label)
        if lemma in self._label_map:
            self._label_map[label] = self._label_map[lemma]
            return self._label_map[label]

        # check against every known canonical label via WordNet similarity
        canonical = self._find_wordnet_match(lemma, set(self._label_map.values()))
        if canonical is not None:
            self._label_map[label] = canonical
            self._label_map[lemma] = canonical
            return canonical

        # no match — this lemma becomes its own canonical form
        self._label_map[label] = lemma
        self._label_map[lemma] = lemma
        return lemma

    def canonicalise_predicate(self, predicate: str) -> str:
        """Return the canonical form of a relationship *predicate*."""
        predicate = predicate.strip().lower()
        if predicate in self._predicate_map:
            return self._predicate_map[predicate]

        lemma = self._lemmatise(predicate)
        if lemma in self._predicate_map:
            self._predicate_map[predicate] = self._predicate_map[lemma]
            return self._predicate_map[predicate]

        self._predicate_map[predicate] = lemma
        self._predicate_map[lemma] = lemma
        return lemma

    @property
    def label_vocabulary(self) -> set[str]:
        """The current set of canonical object labels."""
        return set(self._label_map.values())

    @property
    def predicate_vocabulary(self) -> set[str]:
        """The current set of canonical predicates."""
        return set(self._predicate_map.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lemmatise(self, text: str) -> str:
        """Lemmatise a (possibly multi-word) string with spaCy."""
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc).strip()

    def _find_wordnet_match(self, term: str, candidates: set[str]) -> str | None:
        """Return the candidate with highest Wu-Palmer confidence above threshold."""
        best_score = 0.0
        best_candidate = None
        for candidate in candidates:
            score = wup_confidence(term, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_score >= self.wup_threshold:
            return best_candidate
        return None
