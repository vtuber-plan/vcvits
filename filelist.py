import argparse
import glob
import os
import tqdm
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./dataset/example", help='Dataset path')
    parser.add_argument('-o', '--output', type=str, default="./filelists/example_audio_filelist.txt", help='File list output path')
    parser.add_argument('-s', '--speakers_info', type=str, default="./filelists/example_audio_speakers_info.txt", help='Speakers info output path')
    args = parser.parse_args()

    speaker_folders = [p.name for p in os.scandir(args.input)]
    speaker_folders = sorted(speaker_folders)
    print(f"Speaker Number: {len(speaker_folders)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for sid, speaker_name in enumerate(tqdm.tqdm(speaker_folders)):
            speaker_folder = os.path.join(args.input, speaker_name)
            speaker_files = list(glob.glob(os.path.join(speaker_folder, "*.wav")))
            speaker_files = sorted(speaker_files)

            for file in speaker_files:
                audio = sf.SoundFile(file)
                if audio.frames / audio.samplerate < 3:
                    continue
                file.replace("\\", "/")
                f.write(f"{file}|{sid}\n")
    
    with open(args.speakers_info, "w", encoding="utf-8") as f:
        for s in speaker_folders:
            f.write(f"{s}\n")