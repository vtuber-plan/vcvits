import glob
import os
import tqdm
import shutil

pt_files = glob.glob("dataset/example/*/*.pt")
pt_files = list(pt_files)

for f in tqdm.tqdm(pt_files):
    os.remove(f)

