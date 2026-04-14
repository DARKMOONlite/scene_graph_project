from __future__ import annotations
import nltk
from nltk.corpus import wordnet2022 as wn

nltk.data.path.append('/mnt/sda1/Datasets/NLTK/wordnet-2022')
from nltk.util import acyclic_depth_first as acyclic_search
import json
from collections import deque
from nltk.corpus.reader import Synset
from pprint import pprint
KEY_NOUNS = {"plate","tableware","object","container","human"}



def compress_search_result(result) -> list[str]:
    """Flatten the nested acyclic_depth_first result into a single ordered list of lemma names."""
    if isinstance(result, Synset):
        return [result.lemmas()[0].name().replace('_', ' ')]
    elif isinstance(result, list):
        flat = []
        for item in result:
            flat.extend(compress_search_result(item))
        return flat
    return []


def build_hierarchy(synset: Synset|list[Synset], max_depth: int = 10) -> set[str]:
    """Build a hierarchy of synsets starting from the given synset, following hypernym relations."""
    assert isinstance(synset, Synset) or (isinstance(synset, list) and all(isinstance(s, Synset) for s in synset)), "Input must be a Synset or a list of Synsets."
    if isinstance(synset, list):
        result = set()
        for s in synset:
            result.update(build_hierarchy(s, max_depth))
        return result
    
    result = acyclic_search(synset, lambda s: s.hypernyms(), depth=max_depth)
    result = compress_search_result(result)
    return set(result) 

def get_relevant_hypernyms(synset: Synset|list[Synset], key_nouns: set[str], max_depth: int = 10) -> set[str]:
    """Get the relevant hypernyms of a synset that are in the key_nouns set."""
    hierarchy = build_hierarchy(synset, max_depth)
    return hierarchy.intersection(key_nouns)

def get_synsets_for_noun(noun: str) -> list[Synset]:
    """Get the noun synsets for a (possibly compound) noun phrase.

    Tries the full phrase first (underscored).  If nothing is found and the
    phrase contains multiple words, collects synsets from each word individually
    so that e.g. "female jockey" yields synsets for both "female" and "jockey".
    """
    synsets = wn.synsets(noun.replace(" ", "_"), pos=wn.NOUN, lang='eng')
    if not synsets and " " in noun:
        for word in noun.split():
            synsets.extend(wn.synsets(word, pos=wn.NOUN, lang='eng'))
    return synsets

def get_noun_hierarchy(noun: str, max_depth: int = 10) -> set[str]:
    """Return the full hypernym hierarchy for a noun string.

    Spaces are converted to underscores for the WordNet lookup.
    Returns an empty set if no synsets are found.
    """
    synsets = get_synsets_for_noun(noun)
    return build_hierarchy(synsets, max_depth) if synsets else set()


def wup_confidence(term_a: str, term_b: str, max_depth: int = 10) -> float:
    """Confidence that *term_a* is a hyponym of *term_b*, combining Wu-Palmer
    similarity with sense coverage.

    For each synset of *term_a* whose hypernym hierarchy contains *term_b*,
    the best Wu-Palmer similarity to any synset of *term_b* is recorded.
    The final score is ``best_wup * sqrt(breadth)`` where *breadth* is the
    fraction of *term_a*'s synsets that support the relationship.

    This penalises matches that rely on a single polysemous sense (e.g.
    "head" as a leader) while still rewarding genuine relationships even
    when only some senses agree (e.g. "horse" → "animal").
    """
    import math
    syns_a = get_synsets_for_noun(term_a)
    syns_b = get_synsets_for_noun(term_b)
    if not syns_a or not syns_b:
        return 0.0
    hits = 0
    best_wup = 0.0
    for sa in syns_a:
        if term_b not in build_hierarchy(sa, max_depth):
            continue
        hits += 1
        for sb in syns_b:
            sim = sa.wup_similarity(sb)
            if sim is not None and sim > best_wup:
                best_wup = sim
    breadth = hits / len(syns_a)
    return best_wup * math.sqrt(breadth)

if __name__ == "__main__":
    synsets = wn.synsets("plate", pos=wn.NOUN,lang='eng')
    chains = set()
    for syn in synsets:
        chains.update(get_relevant_hypernyms(syn, KEY_NOUNS))
    pprint(chains)
    