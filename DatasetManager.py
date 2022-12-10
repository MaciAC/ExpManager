from os import listdir, popen, rename, mkdir
from json import load, dump
from os.path import join, exists
import constants
from subprocess import call
from pandas import read_csv, concat
from scipy.io.wavfile import read, write
import numpy as np
from time import sleep
import sys


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


    def create_dataset_synthetic(self, modes):
        folders = []
        for mode in modes:
            mode_str = constants.DATASET_MODES[mode]
            folder = "/home/mamoros/exp/datasets/dataset_%s/%s_set" % (str(int(self.last_dataset) + 1),
                                                                    mode_str)
            with open("/home/mamoros/tmp/output.log", "a") as output:
                call(constants.DOCKER_RUN.format(params="-it --rm",
                                            vol_code="/home/mamoros/build/DNS-Challenge/:/DNS-Challenge",
                                            vol_data="/home/mamoros/exp/datasets/real:/datasets " \
                                                    "-v %s:/out" % folder,
                                            name="DNS-Challenge",
                                            img="mamoros:DNS_challenge",
                                            cmd="python3 "\
                                                "/DNS-Challenge/noisyspeech_synthesizer_singleprocess.py " \
                                                "--cfg noisyspeech_synthesizer_%s.cfg" % mode_str),
                    shell=True,
                    stdout=output,
                    stderr=output)
            folders.append(folder)
        return folders

    def save_snippet(self, snippets, min_length, sr_transcode=16000):
        data_array = []
        for j, folder, filename, debug, begin, dur in snippets:
            out_name = constants.SNIPPETS_DIR % str(self.last_dataset + 1) + folder + '/fileid_' + str(j) + '.wav'
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
            for k in constants.KEYS:
                path = join(constants.NFS_BASE_PATH, row[k])
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

    def convert_audio_multiprocess(self, refs_in, queries_in, refs_out, queries_out, sr, codec):
        #"cat cmds.sh | xargs -I {} -n 1 -P 24 sh -c 'echo \"{}\"; {}'"
        cmds_file = '/home/mamoros/tmp/cmds_convert.sh'
        with open(cmds_file, 'w') as f:
            for i, file in enumerate(refs_in):
                f.write("ffmpeg -i '{}' -ac 1 -ar {} -acodec {} '{}' -n\n".format(
                        file,
                        sr,
                        codec,
                        refs_out % i,
                        ))
            for i, file in enumerate(queries_in):
                f.write("ffmpeg -i '{}' -ac 1 -ar {} -acodec {} '{}' -n\n".format(
                        file,
                        sr,
                        codec,
                        queries_out % i,
                        ))
        print("cat {} | xargs -I % -n 1 -P 8 sh -c 'echo %; %'".format(cmds_file))

    def extract_figerprint(self, folder_in, folder_out, fp_type):
        input('WARNING first run previous Commands!, press intro once done')
        try:
            mkdir(folder_out)
        except:
            pass
        audio_files = listdir(folder_in)
        cmds_file = '/home/mamoros/tmp/cmds_extract_fp.sh'
        with open(cmds_file, 'w') as f:
            for file in audio_files:
                f.write("fpextractor {in_file} {out_file} {fp}\n".format(
                        in_file=join(folder_in, file),
                        out_file=join(folder_out,file.replace('wav', fp_type)),
                        fp=fp_type))
        print("cat {} | xargs -I % -n 1 -P 8 sh -c 'echo %; %'".format(cmds_file))


    def create_index(self, folder):
        input('WARNING first run previous Commands!, press intro once done')
        fp_files = ['{}/{}\n'.format(folder,x) for x in listdir(folder)]
        cmd_file = '/home/mamoros/tmp/cmds_create_index.sh'
        lst_file = join(folder, 'fp.lst')
        with open(lst_file, 'w') as f:
            f.writelines(fp_files)
        with open(cmd_file, 'w') as f:
            f.write("fpmatcher create_index {lst} {out_name}\n".format(
                lst=lst_file,
                out_name=join(folder, 'index')
            ))
        print("cat {} | xargs -I % -n 1 -P 8 sh -c 'echo %; %'".format(cmd_file))


    def create_test_set(self):
        mkdir(constants.SNIPPETS_DIR % str(self.last_dataset + 1))
        out_dir = join(constants.SNIPPETS_DIR % str(self.last_dataset + 1), 'testing_set')
        mkdir(out_dir)
        mkdir(join(out_dir, 'clean'))
        mkdir(join(out_dir, 'noisy'))
        data_dir = '/srv/nfs/bmat_core/fingerprinting_qa/collections/siae_venues_microphone_vol1'
        dataset_file = join(data_dir, 'groundtruth.csv')
        df = read_csv(dataset_file)
        df.iloc[:, 0] = join(data_dir, 'queries/') + df.iloc[:, 0].astype(str)
        df.iloc[:, 1] = join(data_dir, 'references/') + df.iloc[:, 1].astype(str)
        queries_in = []
        queries_out = []
        exists_files = [[],[]]
        prev_ref = ''
        df = df.sort_values(by=df.columns[1], ignore_index=True)
        ref_ids = []
        curr_id = -1
        prev_exists = False
        for i, row in df.iterrows():
            iterate = [0,1]
            if prev_ref.split('/')[-1] == row[1].split('/')[-1]:
                iterate = [0]
                exists_files[1].append(exists_files[1][-1])
            elif prev_exists:
                curr_id += 1
            prev_exists = True
            ref_ids.append(curr_id)
            for j in iterate:
                if not exists(row[j]):
                    if not exists(row[j].replace('vol1', 'vol2')):
                        exists_files[j].append('NO')
                        prev_exists = False
                        #curr_id -= 1
                    else:
                        exists_files[j].append('2')
                        df.iloc[i,j] = row[j].replace('vol1', 'vol2')
                else:
                    exists_files[j].append('1')
            prev_ref = df.iloc[i,1]

        df['query found'] = exists_files[0]
        df['reference found'] = exists_files[1]
        df['reference ids'] = ref_ids
        df.drop(df[df['query found'] == 'NO'].index, inplace = True)
        df.drop(df[df['reference found'] == 'NO'].index, inplace = True)
        df.reset_index(inplace=True, drop = True)
        df.to_csv(join(out_dir, 'ground_truth_post.csv'))
        queries_in = df.iloc[:, 0].astype(str)
        queries_out = join(out_dir, 'noisy/fileid_%s.wav')

        df['ref_name'] = df.iloc[:, 1].str.split('/').str[-1]
        df = df.drop_duplicates('ref_name')
        refs_in = df.iloc[:, 1].astype(str)
        refs_out = join(out_dir, 'clean/fileid_%s.wav')

        self.convert_audio_multiprocess(refs_in, queries_in, refs_out, queries_out,
                                        8000,
                                        'pcm_s16le')

        self.extract_figerprint(join(out_dir, 'clean'), join(out_dir, 'clean_fp1'), 'fp1')

        self.create_index(join(out_dir, 'clean_fp1'))



    def create_dataset_real(self, mode):
        if mode == ['1', '3']:
            dataset_path = join(constants.EXP_DIR, "datasets/dataset_%s/" % str(self.last_dataset + 1))
            mkdir(dataset_path)
            mkdir(join(dataset_path, 'clean'))
            mkdir(join(dataset_path, 'noisy'))
            self.cp_nfsdataset_audio2snippet('/home/mamoros/exp/datasets/real/better_alignment.csv',
                                            min_snippet_len = 5,
                                            copy=True,
                                            max_samples=0,
                                            debug=False,
                                            out_sr=8000)

        else:
            self.create_test_set()


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
        modes = [str(x) for x in range(1,3)]
        while not ok:
            mode = input( "1. Train&Valid\n" \
                            "2. Test\n")
            if mode in modes:
                ok = True
        if mode == '1':
            modes = ['1', '3']
        else:
            modes = [mode]

        if option == '1':
            folders = self.create_dataset_synthetic(modes)
            type_ = "synth"
            for folder in folders:
                self.rename_files(folder)
        if option == '2':
            self.create_dataset_real(modes)
            type_ = "real"

        self.last_dataset += 1
        data = {
            "id": self.last_dataset,
            "type": type_,
            "mode": "_".join([constants.DATASET_MODES[x] for x in modes]),
            "description": "",
            "size": 0
        }

        with open(join(constants.EXP_DIR, "datasets/dataset_%s/" % str(self.last_dataset), "config.json"), 'w') as json_file:
            dump(data, json_file)