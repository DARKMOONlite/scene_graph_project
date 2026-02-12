"""
Converts Visual Genome relationships.json file to triplets file
"""

import argparse
from pathlib import Path
from pprint import pp
from rule_mining.datasets.visual_genome.Rule import load_json_files, summarize_structure, Triplet



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Convert relationships.json file to triplets file for anyburl")
    parser.add_argument("-f","--file",type=str,required=True,help="the folder or specific file that should be loaded")
    parser.add_argument("-o","--output",type=str,help="the file to print the output to")
    args:argparse.Namespace = parser.parse_args()
    
    triplets:list[Triplet] = []
    try:
        data: list = load_json_files(path=Path(args.file), multiple_files_allowed=False)
        # pp(summarize_structure(data,depth=6))
        count = 0
        for scene in data[0]:
            for relationship in scene["relationships"]:
                # pass
                try:
                    obj = relationship["object"]["object_id"]
                    sub = relationship["subject"]["object_id"]
                    pred=relationship["predicate"]
                    triplets.append(Triplet(obj,sub,pred))
                except Exception as e:
                    count+=1
                    # print(f"Failed to get triplet: {e}")
                
        
        print(f"collected {len(triplets)} triplets, however missed {count} relationships due to either missing object or subject name")
        if args.output is None:
            print("no file saved")
        else:
            with open(args.output,mode="w") as file:
                for triplet in triplets:
                    file.write(triplet.print()+"\n")
            print(f"file saved at {args.output}")
        
    except Exception as e:
        print(f"Exception: {e}")



if __name__=="__main__":
    main()