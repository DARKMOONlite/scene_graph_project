from __future__ import annotations
import os
WORDNET_DIR = "/mnt/sda1/Datasets/NLTK/wordnet-2022"
WORDNET_2022_DIR = "/mnt/sda1/Datasets/NLTK/wordnet-2022/corpora/wordnet2022"
WORDNET_CORRECT_DIR = "/mnt/sda1/Datasets/NLTK/wordnet-2022/corpora/wordnet"
import nltk
nltk.data.path.append(WORDNET_DIR)
from nltk.corpus import wordnet2022 as wn
from nltk.util import acyclic_depth_first as acyclic_search
import json
from collections import deque
from nltk.corpus.reader import Synset
from pprint import pprint
from pathlib import Path
import shutil
from enum import Enum
KEY_NOUNS = {"plate","tableware","object","container","human"}

class WordNetType(Enum):
    ADJ = "a"
    ADJ_SAT = "s"
    ADV = "r"
    NOUN = "n"
    VERB = "v"


def install_wordnet():
    """Utility function to download and install the WordNet 2022 corpus."""
    import nltk
    if WORDNET_DIR not in nltk.data.path:
        nltk.data.path.append(WORDNET_DIR)
    if not Path(WORDNET_DIR).is_dir():
        os.makedirs(WORDNET_DIR, exist_ok=True)
        
    if not Path(WORDNET_CORRECT_DIR).is_dir():
        result = nltk.download('wordnet2022', download_dir=WORDNET_DIR)
        print(f"WordNet 2022 installed in {WORDNET_DIR}: {result}")
        shutil.move(WORDNET_2022_DIR, WORDNET_CORRECT_DIR) # move wordnet2022 to wordnet so that it can be loaded by nltk.corpus.wordnet
    
    
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

def get_synsets(word: str, word_type: WordNetType = WordNetType.NOUN) -> list[Synset]:
    """Get the noun synsets for a (possibly compound) noun phrase.

    Tries the full phrase first (underscored).  If nothing is found and the
    phrase contains multiple words, collects synsets from each word individually
    so that e.g. "female jockey" yields synsets for both "female" and "jockey".
    """
    synsets = wn.synsets(word.replace(" ", "_"), pos=word_type.value, lang='eng')
    if not synsets and " " in word:
        for word in word.split():
            synsets.extend(wn.synsets(word, pos=word_type.value, lang='eng'))
    return synsets

def get_noun_hierarchy(noun: str, max_depth: int = 10) -> set[str]:
    """Return the full hypernym hierarchy for a noun string.

    Spaces are converted to underscores for the WordNet lookup.
    Returns an empty set if no synsets are found.
    """
    synsets = get_synsets(noun)
    return build_hierarchy(synsets, max_depth) if synsets else set()


def wup_confidence(term_a: str, term_b: str, max_depth: int = 10, word_type: WordNetType = WordNetType.NOUN) -> float:
    """Symmetric semantic confidence between two terms.

    The score is the stronger of the two directed hypernym scores, so its
    value is unchanged when the terms are swapped.
    """
    return max(
        directed_wup_confidence(term_a, term_b, max_depth, word_type),
        directed_wup_confidence(term_b, term_a, max_depth, word_type),
    )


def directed_wup_confidence(term_a: str, term_b: str, max_depth: int, word_type: WordNetType = WordNetType.NOUN) -> float:
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
    syns_a = get_synsets(term_a, word_type)
    syns_b = get_synsets(term_b, word_type)
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
    