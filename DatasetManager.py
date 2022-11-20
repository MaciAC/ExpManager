from os import listdir, popen, rename
from json import load, dump
from os.path import join
import constants
from subprocess import call
from pandas import read_csv
from scipy.io.wavfile import read, write
import numpy as np
from time import sleep
import stat


class DatasetManager:

    def __init__(self):
        self.datasets = {}
        max_id = 0
        for dataset in [x for x in listdir(join(constants.EXP_DIR,'datasets')) if 'dataset' in x]:
            with open(join(constants.EXP_DIR, 'datasets', dataset, 'config.json')) as json_file:
                data = load(json_file)
                id = data['id']
                self.datasets[id] = data
                if int(data['id']) > max_id:
                    max_id = int(data['id'])
        self.last_dataset = max_id


    def create_dataset_synthetic(self, mode):
        folder = "/home/mamoros/exp/datasets/dataset_%s/%s_set" % (str(int(self.last_dataset) + 1),
                                                                   constants.DATASET_MODES[mode])
        with open("/home/mamoros/tmp/output.log", "a") as output:
            call(constants.DOCKER_RUN.format(params="-it --rm",
                                        vol_code="/home/mamoros/build/DNS-Challenge/:/DNS-Challenge",
                                        vol_data="/home/mamoros/exp/datasets/real:/datasets " \
                                                 "-v %s:/out" % folder,
                                        name="DNS-Challenge",
                                        img="mamoros:DNS_challenge",
                                        cmd="python3 /DNS-Challenge/noisyspeech_synthesizer_singleprocess.py"),
                 shell=True,
                 stdout=output,
                 stderr=output)
        return folder

    def save_snippet(self, snippets, min_length, sr_transcode=16000):
        data_array = []
        for j, folder, filename, debug, begin, dur in snippets:
            out_name = constants.SNIPPETS_DIR + folder + '/fileid_' + str(j) + '.wav'
            waiting = True
            f = 'query'
            if folder == 'clean':
                f = 'reference'
            while waiting:
                try:
                    sr, data = read(join(constants.AUDIOFILES_DIR, f, filename))
                    waiting = False
                except ValueError:
                    if filename.endswith('.wav'):
                        continue
                    new_file = join(constants.AUDIOFILES_DIR, f, filename[:-4],'.wav')
                    popen("ffmpeg -i '{}' -acodec pcm_s16le '{}' -y".format(
                    constants.AUDIOFILES_DIR + filename,
                    new_file))
                    sr, data = read(new_file)
                    waiting = False
                except FileNotFoundError:
                    print(filename)
                    sleep(1)
            begin = int(begin * sr)
            dur = int(dur * sr)
            if len(data[begin:begin+dur]) < min_length*sr:
                return False
            # normalize
            print(data)
            print(data.max())
            data = data/np.abs(data).max()
            print(data.max())
            data_array.append((data[begin:begin+dur], out_name))
        if len(data_array) != 2:
            return False
        for data, out_name in data_array:
            if not debug:
                amplitude = np.iinfo(np.int16).max
                write(out_name, sr, (amplitude * data).astype(np.int16))
                if sr_transcode != 8000:
                    popen("ffmpeg -i '{}' -ar {} '{}' -n".format(
                        out_name,
                        sr_transcode,
                        out_name.replace('8khz', '16khz')
                ))
            else:
                print(j, out_name)
        return True


    def cp_nfsdataset_audio2snippet(self, filename, min_snippet_len = 0, copy=False, debug=False, max_samples=0, out_sr=8000):
        df = read_csv(filename)
        already_copied = listdir(constants.AUDIOFILES_DIR,)
        j = 0
        limit = len(df.index)
        if max_samples != 0:
            limit = max_samples
        for i, row in df.head(limit).iterrows():
            # copy from nfs if not already copied
            for k in self.KEYS:
                path = join(self.NFS_BASE_PATH, row[k])
                filename = path.split('/')[-1]
                if copy:
                    if filename in already_copied:
                        print('Already copied %s' % filename)
                    else:
                        print('Copy')
                        popen("cp '{}' '{}'".format(path, join(constants.AUDIOFILES_DIR ,k, filename)))
                        already_copied.append(filename)
                begin_time = row[k + '_begin_time']
                end_time = row[k + '_end_time']
                if k == 'reference':
                    r_dur = end_time - begin_time
                    r_begin = begin_time
                    r_filename = filename
                elif k == 'query':
                    q_dur = end_time - begin_time
                    q_begin = begin_time
                    q_filename = filename

            # define new duration for query and ref snippet
            dur = q_dur if q_dur > r_dur else r_dur
            if dur < min_snippet_len or r_filename.endswith('.mp3') or q_filename.endswith('.mp3'):
                continue
            if debug:
                print(j, 'clean', r_filename, debug, r_begin, dur)
                print(j, 'noisy', q_filename, debug, q_begin, dur)
            snippets = [(j, 'clean', r_filename, debug, r_begin, dur),
                        (j, 'noisy', q_filename, debug, q_begin, dur)]
            if self.save_snippet(snippets, min_snippet_len, sr_transcode=out_sr):
                j += 1


    def create_dataset_real(self, mode):
        self.cp_nfsdataset_audio2snippet('/home/mamoros/dataset/dataset.csv', copy=True, max_samples=0, debug=False, out_sr=8000)


    def rename_files(self, folder):
        dirs = listdir(folder)
        for d in dirs:
            base_path = join(folder, d)
            files = listdir(base_path)
            for file in files:
                new_file = join(base_path, '_'.join(file.rsplit('_')[-2:]))
                rename(join(base_path, file), new_file)


    def create_dataset(self):
        ok = False
        options = [str(x) for x in range(1,3)]
        while not ok:
            option = input( "1. Synthetic\n" \
                            "2. Real\n")
            if option in options:
                ok = True
        ok=False
        modes = [str(x) for x in range(1,4)]
        while not ok:
            mode = input( "1. Train\n" \
                            "2. Test\n" \
                            "3. Valid\n")
            if mode in modes:
                ok = True
        if option == '1':
            folder = self.create_dataset_synthetic(mode)
            type_ = "synth"
        if option == '2':
            folder = self.create_dataset_real(mode)
            type_ = "real"

        self.rename_files(folder)

        self.last_dataset += 1
        data = {
            "id": self.last_dataset,
            "type": type_,
            "mode": constants.DATASET_MODES[mode],
            "description": "",
            "size": 0
        }

        with open(join(constants.EXP_DIR, "datasets/dataset_%s/%s_set" % (str(self.last_dataset), constants.DATASET_MODES[mode]), "config.json"), 'w') as json_file:
            dump(data, json_file)