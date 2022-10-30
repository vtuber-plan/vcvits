import argparse
import glob
import os
import tqdm
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./dataset/example", help='Dataset path')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input, "*/*.pt"))
    files = list(files)

    for file in tqdm.tqdm(files):
        os.remove(file)
