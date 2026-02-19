"""
filters the conceptnet file.
"""
from networkx.utils.tests.test_heaps import data
import ast
from ntpath import sep
from argparse import ArgumentParser, Namespace
from itertools import islice
from collections import Counter
from tqdm import tqdm
import threading
from itertools import islice


def filter_relationship(line: str|None, relations: set[str], mode: str) -> str | None:
    if line is None:
        return None
    fields: list[str] = line.split(sep="\t")
    relation:str = fields[1] if len(fields) > 1 else ""

    if mode == "exclude":
        return line if relation not in relations else None
    else:  # include mode
        return line if relation in relations else None

def filter_languages(line:str|None, languages:set[str],mode:str,strict:bool=False)->str|None:
    if line is None:
        return None
    fields: list[str] = line.split(sep="\t")
    object_:str = fields[2] if len(fields) > 2 else ""
    object_parts = object_.split(sep="/")
    subject_:str = fields[3] if len(fields) > 3 else ""
    object_parts.extend(subject_.split(sep="/"))
    # print(object_parts)
    if mode == "exclude":
        for language in languages:
            if language in object_parts:
                return None
        return line
    else:  # include mode
        count = 0;
        for language in languages:
            num: int = object_parts.count(language)
            count +=num
        if count >= 1 + strict:
            return line
        return None
    
def filter_dataset(line:str|None,datasets:set[str],mode:str)->str|None:
    if line is None:
        return None
    fields: list[str] = line.split(sep="\t")
    dict_:dict[str,str] = ast.literal_eval(fields[4]) 
    key = dict_["dataset"]
    if mode=="exclude":
        return line if key not in datasets else None
    else:
        return line if key in datasets else None
    
def count_line(line: str, counts: Counter, print_mode: str) -> None:
    """
    Update a Counter based on a line's relation or dataset.

    Args:
        line: A tab-separated line from the conceptnet file
        counts: Counter to update in-place
        print_mode: "relations" to count by relation type,
                    "datasets" to count by subject language/dataset
    """
    fields: list[str] = line.split(sep="\t")
    if print_mode == "relations":
        key = fields[1] if len(fields) > 1 else "<unknown>"
    else:  # datasets — extract language from subject URI e.g. /c/en/dog -> en
        dict_:dict[str,str] = ast.literal_eval(fields[4]) 
        key = dict_["dataset"]
    counts[key] += 1


def print_counts(counts: Counter, print_mode: str) -> None:
    """Print sorted counts to the console."""
    print(f"\n--- {print_mode.capitalize()} counts ---")
    for key, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")



def main():
    parser = ArgumentParser("filters the conceptnet file for a subset based on some passed filters")
    parser.add_argument("input_file")
    parser.add_argument("-o","--output",type=str,default="filtered-conceptnet.csv")
    parser.add_argument("-r","--relations",type=str,default="",nargs="*",help="""list of relations to filter: 
    /r/DistinctFrom
    /r/CausesDesire
    /r/PartOf
    /r/MotivatedByGoal
    /r/HasPrerequisite
    /r/HasA
    /r/LocatedNear
    /r/NotHasProperty
    /r/RelatedTo
    /r/dbpedia/language
    /r/SymbolOf
    /r/Causes
    /r/Entails
    /r/IsA
    /r/HasSubevent
    /r/UsedFor
    /r/FormOf
    /r/ExternalURL
    /r/dbpedia/genre
    /r/Antonym
    /r/HasProperty
    /r/DerivedFrom
    /r/ObstructedBy
    /r/ReceivesAction
    /r/dbpedia/genus
    /r/dbpedia/capital
    /r/DefinedAs
    /r/dbpedia/occupation
    /r/NotUsedFor
    /r/CapableOf
    /r/dbpedia/leader
    """)
    parser.add_argument("-d","--datasets",default="",nargs="*",help="""list of datasets to filter:
/d/wiktionary/en: 17425519
/d/wiktionary/fr: 8117066
/d/dbpedia/en: 2573294
/d/wiktionary/de: 1667668
/d/wordnet/3.1: 1409039
/d/jmdict: 896843
/d/cc_cedict: 415387
/d/emoji: 383522
/d/conceptnet/4/zh: 357479
/d/conceptnet/4/en: 226284
/d/opencyc: 166999
/d/verbosity: 162297
/d/kyoto_yahoo: 133333
/d/conceptnet/4/pt: 77645
/d/conceptnet/4/ja: 58751
/d/conceptnet/4/hu: 1973
/d/conceptnet/4/nl: 1650
/d/conceptnet/4/es: 100
/d/conceptnet/4/it: 48
/d/conceptnet/4/fr: 19
/d/conceptnet/4/ko: 1
                        """)
    parser.add_argument("-m","--mode",type=str,choices=["exclude","include"],default="exclude",help="Filter mode: 'exclude' removes specified relations, 'include' keeps only specified relations")
    parser.add_argument("-p","--print",type=str,choices=["relations","datasets"],help="prints out a list including number of all requested data")
    parser.add_argument("-l","--lang",type=str,nargs="*",help="list of 2 letter language tags to filter")
    parser.add_argument("--strict",action="store_true",help="apply filters 'strictly' meaning for example both object and subject must be of the languages provided")
    args: Namespace = parser.parse_args()
    relations: set[str] = set()
    datasets:set[str] = set()
    languages:set[str] = set()
    if args.relations:
        relations = set(args.relations)
    if args.datasets:
        datasets = set(args.datasets)
    if args.lang:
        languages = set(args.lang)
    strict:bool = True if args.strict else False

    print(f"Mode: {args.mode}")
    print(f"Relations: {relations}")
    print(f"Datasets:{datasets}")
    print(f"Languages: {languages}")
    print("=======================")
    removed_count = 0
    kept_count = 0
    counts: Counter = Counter()
    
    with open(args.input_file,mode="r") as infile, open(args.output, mode='w') as outfile:
        for line in tqdm(infile, desc=f"Filtering relations ({args.mode} mode)"):
            result:str = line
            if(len(relations)>0):
                result: str | None = filter_relationship(result, relations, args.mode)
            if(len(datasets)>0 and result is not None):
                result: str | None = filter_dataset(result,datasets,args.mode)
            if(len(languages)>0 and result is not None):
                result:str|None = filter_languages(result,languages,args.mode,strict=strict)
            if result is not None:
                outfile.write(result)
                kept_count += 1
                if args.print:
                    count_line(result, counts, args.print)
            else:
                removed_count += 1
    print(f"\nFiltering complete ({args.mode} mode):")
    print(f"  Kept: {kept_count} lines")
    print(f"  Removed: {removed_count} lines")
    print(f"  Output saved to: {args.output}")
    if args.print:
        print_counts(counts, args.print)

    pass


if __name__ == "__main__":
    main()


