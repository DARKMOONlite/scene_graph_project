import random
from pathlib import Path
import argparse
import sys

description=""" Visualises txt result Files, 
    
    """
def main():
    parser = argparse.ArgumentParser(prog="Visualiser",description=description)
    parser.add_argument("filename",help="the file to load")
    parser.add_argument("-o","--output",help="File to output the txt file to",default="result.txt")
    parser.add_argument("-n","--num",help="set a maximum number of lines to read (for big files)",type=int)

    args:argparse.Namespace = parser.parse_args()

    file_path = Path(args.filename)
    if not file_path.absolute().resolve().exists():
        print(f"file '{file_path.absolute()}' not found")
        sys.exit()
    destination_path = Path(args.output)
    if not file_path.absolute().resolve().exists():
        file_path.absolute().resolve().mkdir(parents=True)
        
        
        
if __name__ == "__main__":
    main()