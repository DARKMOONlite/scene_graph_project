from __future__ import annotations
import spacy
from src.pipeline.wordnet import build_hierarchy, get_synsets_for_noun
import json
from nltk.corpus.reader import Synset
from pprint import pprint
from fastcoref import CorefResult, LingMessCoref
from src.pipeline.prolog import load_coco_jsonl, load_words_from_csv,ProLogFact, get_id_from_file_name,create_popper_files,ProLogExample

METADATA = ["she is holding a horse's reins.",
            "she is showing a horse.",
            "she is holding the horse",
            "she is competing in a horse show.", 
            "she wants to win.", 
            "to train with him",
            "A female jockey leading a white horse in front of an audience.",
            "A woman is holding a horse on a leash",
            "A white horse at a horse show with it's trainer.",
            "Woman in a competition signalling horse to stand still",
            "A woman training a white horse in a contest."]
TARGET_VERBS = {"showing", "leading", "training", "holding","signalling","competing","playing","watching","sitting","crossing","looking"}
KEY_NOUNS = {"horse", "jockey", "person", "animal", "human", "object", "container", "living thing", "organism", "creature"}


class PronounResolver:
    """Resolves pronouns to their noun referents in a collection of sentences.

    Uses spaCy for POS tagging and fastcoref for coreference resolution.
    Sentences containing no pronouns are used as noun-rich context when
    resolving pronouns in sentences that do contain them.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm",verb_list: list[str]|None=list()):
        self.nlp = spacy.load(spacy_model)
        self.coref = LingMessCoref(enable_progress_bar=False)
        self.verb_list = set(verb_list)
        self.lemmatised_verbs = {doc[0].lemma_ for v in self.verb_list if (doc := self.nlp(v))}

    
    def add_target_verbs(self, verb_list: list[str]):
        """Add verbs to the target verb list, lemmatising them for better matching."""
        self.verb_list.update(verb_list)
        self.lemmatised_verbs.update(doc[0].lemma_ for v in verb_list if (doc := self.nlp(v)))
    # --- token-level helpers ---

    def _get_span_for_token(self, token, doc) -> str:
        """Return the root noun of the chunk containing token (strips determiners/adjectives).

        e.g. 'A white horse' -> 'horse', 'A woman' -> 'woman'.
        Falls back to the token text if it is not inside any noun chunk.
        """
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk.root.text.lower() # get root to remove things like adjectives and article (eg: white, a)
        return token.text.lower()

    def get_pronouns(self, sentence: str) -> list[str]:
        """Return all pronouns in a sentence."""
        return [t.text for t in self.nlp(sentence) if t.pos_ == "PRON"]

    def get_nouns(self, sentence: str) -> list[str]:
        """Return all nouns and proper nouns in a sentence."""
        return [t.text for t in self.nlp(sentence) if t.pos_ in ("NOUN", "PROPN")]

    def _collect_prep_phrase(self, token) -> tuple[str, object | None]:
        """Walk prep / advmod+prep children to find a prepositional object.

        Returns (preposition_text, pobj_token) or ("", None).
        Handles simple preps ("on the snow") and multi-word preps ("next to the skis").
        """
        for child in token.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return child.text, grandchild
            # multi-word preps: advmod → prep → pobj  (e.g. "next to")
            if child.dep_ == "advmod":
                for prep in child.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                return f"{child.text}_{prep.text}", pobj
        return "", None

    def get_triplet(self, sentence: str) -> ProLogExample:
        """Extract (verb, subject, object) for the first target verb found in a sentence.

        Handles direct objects, phrasal verbs (particles like "pick up"),
        and prepositional verb phrases ("playing on the snow", "sitting next to").
        target_verbs may be in any inflected form; they are lemmatised before matching.
        """
        doc = self.nlp(sentence)
        for token in doc:
            if token.pos_ != "VERB" or token.lemma_ not in self.lemmatised_verbs:
                continue
            subjects = [t for t in token.lefts if t.dep_ in ("nsubj", "nsubjpass")]
            # acl verbs (e.g. "jockey leading …") have their head noun as implicit subject
            if not subjects and token.dep_ == "acl" and token.head.pos_ in ("NOUN", "PROPN", "PRON"):
                subjects = [token.head]

            # collect particle if present (e.g. "picked up")
            particle = next((t.text for t in token.children if t.dep_ == "prt"), "")

            objects = [t for t in token.rights if t.dep_ in ("dobj", "obj")]
            verb_text = f"{token.text}_{particle}" if particle else token.text

            # fall back to prepositional object when no direct object found
            if not objects:
                prep_text, pobj_token = self._collect_prep_phrase(token)
                if pobj_token is not None:
                    verb_text = f"{verb_text}_{prep_text}" if prep_text else verb_text
                    objects = [pobj_token]

            subject = self._get_span_for_token(subjects[0], doc) if subjects else ""
            obj = self._get_span_for_token(objects[0], doc) if objects else ""

            return ProLogExample(verb_text, subject, obj, positive=True)

        return ProLogExample("", "", "", positive=False)

    # --- sentence-level partitioning ---

    def _partition_sentences(self, sentences: list[str]) -> tuple[list[str], list[str]]:
        """Split sentences into (those without pronouns, those with pronouns)."""
        without, with_ = [], []
        for s in sentences:
            (with_ if any(t.pos_ == "PRON" for t in self.nlp(s)) else without).append(s)
        return without, with_

    # --- coreference resolution ---

    def resolve(self, text: str | list[str]) -> list[list[str]]:
        """Return coreference clusters for text.

        Accepts a single string or a list of sentences; lists are joined into
        one document so the model can resolve cross-sentence references.
        """
        if isinstance(text, list):
            text = " ".join(text)
        preds: CorefResult = self.coref.predict(text)
        return preds.get_clusters(as_strings=True)

    # --- high-level extraction ---

    def extract_pronoun_dict(self, sentences: list[str]) -> dict[str, list[str]]:
        """Map each pronoun to the noun mentions it co-refers with across all sentences."""
        pronouns = {p for s in sentences for p in self.get_pronouns(s)}
        context_sentences, pronoun_sentences = self._partition_sentences(sentences)

        result: dict[str, list[str]] = {}
        for pronoun in pronouns:
            mentions: set[str] = set()
            for sentence in pronoun_sentences:
                if pronoun not in sentence:
                    continue
                # resolve with noun-rich context prepended for better accuracy
                clusters = self.resolve(context_sentences + [sentence])
                for cluster in clusters:
                    if pronoun in cluster:
                        mentions.update(m for m in cluster if m != pronoun)
            if mentions:
                result[pronoun] = list(mentions)
        return result

    def squash_pronoun_dict(self, pronoun_dict: dict[str, list[str]]) -> dict[str, str]:
        """Reduce each pronoun's mention list to a single representative noun.

        TODO: replace shortest-string heuristic with a smarter selection strategy.
        """
        return {pronoun: min(mentions, key=len) for pronoun, mentions in pronoun_dict.items() if mentions}

    def replace_pronouns(self, sentence: str, pronoun_dict: dict[str, str]|dict[str,list[str]]) -> str:
        """Replace pronouns in a sentence with their resolved nouns from pronoun_dict."""
        
        sentences = sentence.split(" ")
        resolved = [pronoun_dict.get(word, word) for word in sentences]
        return " ".join(resolved)





