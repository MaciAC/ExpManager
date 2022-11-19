from json import load
from os import mkdir, listdir
from os.path import join
from subprocess import call
from pandas import read_csv
import os
from scipy.io.wavfile import read, write
import time
import numpy as np

ROOT_DIR = '/home/mamoros'

class ExpManager:

    DOCKER_RUN = "docker run -it --rm -v {vol_code} -v {vol_data} --name {name} {img} {cmd}"
    DOCKER_EXEC = "docker exec {name} {cmd}"

    NFS_BASE_PATH = '/srv/nfs/bmat_core/fingerprinting_qa/collections/'

    AUDIOFILES_DIR = '/home/mamoros/exp/datasets/real/audiofiles/'
    SNIPPETS_DIR = '/home/mamoros/exp/datasets/real/'
    KEYS = ['query', 'reference']

    def __init__(self):
        self.exp_path = ROOT_DIR + '/exp'
        self.datasets = []
        self.experimets = []

        try:
            listdir(self.exp_path)
        except FileNotFoundError:
            mkdir(self.exp_path)
            mkdir(join(self.exp_path,'datasets'))

            self.datasets.append(dataset)
        max_id = 0
        for dataset in [x for x in listdir(join(self.exp_path,'datasets')) if 'dataset' in x]:
            with open(join(self.exp_path, 'datasets', dataset, 'config.json')) as json_file:
                data = load(json_file)
                self.datasets.append(data)
                if int(data['id']) > max_id:
                    max_id = int(data['id'])
        self.last_dataset = max_id

        max_id = 0
        for exp in [x for x in listdir(self.exp_path) if 'exp' in x]:
            with open(join(self.exp_path, exp, 'config.json')) as json_file:
                data = load(json_file)
                self.experimets.append(data)
                if int(data['id']) > max_id:
                    max_id = int(data['id'])
        self.last_exp = max_id


    def create_dataset_synthetic(self):
        with open("/home/mamoros/tmp/output.log", "a") as output:
            call(self.DOCKER_RUN.format(vol_code="/home/mamoros/build/DNS-Challenge/:/DNS-Challenge",
                                        vol_data="/home/mamoros/exp/datasets/real:/datasets -v /home/mamoros/exp/datasets/dataset_0/:/out",
                                        name="DNS-Challenge",
                                        img="mamoros:DNS_challenge",
                                        cmd="python3 /DNS-Challenge/noisyspeech_synthesizer_singleprocess.py"),
                 shell=True,
                 stdout=output,
                 stderr=output)


    def save_snippet(self, snippets, min_length, sr_transcode=16000):
        data_array = []
        for j, folder, filename, debug, begin, dur in snippets:
            out_name = self.SNIPPETS_DIR + folder + '/fileid_' + str(j) + '.wav'
            waiting = True
            f = 'query'
            if folder == 'clean':
                f = 'reference'
            while waiting:
                try:
                    sr, data = read(join(self.AUDIOFILES_DIR, f, filename))
                    waiting = False
                except ValueError:
                    if filename.endswith('.wav'):
                        continue
                    new_file = join(self.AUDIOFILES_DIR, f, filename[:-4],'.wav')
                    os.popen("ffmpeg -i '{}' -acodec pcm_s16le '{}' -y".format(
                    self.AUDIOFILES_DIR + filename,
                    new_file))
                    sr, data = read(new_file)
                    waiting = False
                except FileNotFoundError:
                    print(filename)
                    time.sleep(1)
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
                    os.popen("ffmpeg -i '{}' -ar {} '{}' -n".format(
                        out_name,
                        sr_transcode,
                        out_name.replace('8khz', '16khz')
                ))
            else:
                print(j, out_name)
        return True


    def cp_nfsdataset_audio2snippet(self, filename, min_snippet_len = 0, copy=False, debug=False, max_samples=0, out_sr=8000):
        df = read_csv(filename)
        already_copied = os.listdir(self.AUDIOFILES_DIR,)
        j = 0
        limit = len(df.index)
        if max_samples != 0:
            limit = max_samples
        for i, row in df.head(limit).iterrows():
            # copy from nfs if not already copied
            for k in self.KEYS:
                path = os.path.join(self.NFS_BASE_PATH, row[k])
                filename = path.split('/')[-1]
                if copy:
                    if filename in already_copied:
                        print('Already copied %s' % filename)
                    else:
                        print('Copy')
                        os.popen("cp '{}' '{}'".format(path, join(self.AUDIOFILES_DIR ,k, filename)))
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


    def create_dataset_real(self):
        self.cp_nfsdataset_audio2snippet('/home/mamoros/dataset/dataset.csv', copy=True, max_samples=0, debug=False, out_sr=8000)

    def create_dataset(self):
        ok = False
        options = [str(x) for x in range(1,3)]
        while not ok:
            option = input( "1. Synthetic\n" \
                            "2. Real\n")
            if option in options:
                ok = True
        if option == '1':
            self.create_dataset_synthetic()
        if option == '2':
            self.create_dataset_real()


    def create_exp(self):
        exp_id = self.last_exp + 1
        print("Creating new experiment with ID {}".format(exp_id))
        curr_path = join(self.exp_path, "exp_%s" % str(exp_id))
        mkdir(curr_path)
        mkdir(join(curr_path, 'tensorboard'))
        mkdir(join(curr_path, 'checkpoint'))
        print("Getting config from previous exp")
        with open(join(self.exp_path, "exp_%s" % str(exp_id-1), "config.json")) as json_file:
            data = load(json_file)
        data['id'] = exp_id
        dataset_id = data['dataset']
        if dataset_id in self.datasets:
            print("Using dataset %s" % dataset_id)
        else:
            print("No dataset related to id %s" % dataset_id)
            self.create_dataset()


    def prompt(self):
        print('Datasets:')
        for d in self.datasets:
            print(d)
        print('Experimets:')
        for e in self.experimets:
            print(" Id: {}\n Dataset id: {}\n Description:{}".format(
                                                                e['id'],
                                                                e['dataset'],
                                                                e['description']))
        ok = False
        options = [str(x) for x in range(1,4)]
        while not ok:
            option = input( "1. Create experiment\n" \
                            "2. Create dataset\n" \
                            "3. Run experiment\n")
            if option in options:
                ok = True

        if option == '1':
            self.create_exp()
        elif option == '2':
            self.create_dataset()
        elif option == '3':
            self.run_exp()
        else:
            raise Exception

d = ExpManager()
d.prompt()