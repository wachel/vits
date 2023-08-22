import os

def remove_all(paths):
    for path in paths:
        os.remove(path)

def collect_all(root_path:str, end_with):
    result = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(end_with):
                wav_path = os.path.join(root, file)
                result.append(wav_path)
    return result

files = []    
files.extend(collect_all('data/cp2077/wav_22k', '.w2vh.pt'))
files.extend(collect_all('data/cp2077/wav_22k', '.w2vh.pt.npy'))
files.extend(collect_all('data/cp2077/wav_22k', '.w2vh.npy'))
files.extend(collect_all('data/cp2077/wav_22k', '.semantic.npy'))
files.extend(collect_all('data/qs/wav_22k', '.w2vh.pt'))
files.extend(collect_all('data/qs/wav_22k', '.w2vh.pt.npy'))
files.extend(collect_all('data/qs/wav_22k', '.w2vh.npy'))
files.extend(collect_all('data/qs/wav_22k', '.semantic.npy'))

remove_all(files)