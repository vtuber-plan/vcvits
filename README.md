# VCVITS: VITS-based Voice conversion Model
This is a repository for a voice conversion model based on [VITS](https://github.com/jaywalnut310/vits).

## Pre-requisites
0. Python >= 3.8
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. Download FairSeq Hubert pretrained models：[Download](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md)
0. Download datasets


## Parameters


## Training

### File List Generation
First generate the file list text file from the dataset folder.
One folder for audio files of each speaker in the dataset folder.
Run `filelist.py` to get `filelist.txt`

Then split the training and validation dataset, run `split.py`.



### Config Settings


## Inference
pass
