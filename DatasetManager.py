from os import listdir, popen, rename, mkdir
from json import load, dump
from os.path import join, exists
import constants
from subprocess import call
from pandas import read_csv, concat, Series
from scipy.io.wavfile import read, write
import numpy as np
from time import sleep
import subprocess
import io

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
            folder = "/data/mamoros/exp/datasets/dataset_%s/%s_set" % (str(int(self.last_dataset) + 1),
                                                                    mode_str)
            with open("/home/mamoros/tmp/output.log", "a") as output:
                call(constants.DOCKER_RUN.format(params="-it --rm",
                                            vol_code="/home/mamoros/build/DNS-Challenge/:/DNS-Challenge",
                                            vol_data="/data/mamoros/exp/datasets/real:/datasets " \
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
                    sleep(1)
            begin = int(begin * sr)
            dur = int(dur * sr)
            if len(data[begin:begin+dur]) < min_length*sr:
                return False
            # normalize
            data = data/np.abs(data).max()
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



    def run_commands_multiprocess(self, cmds, silent = False):
        p2 = subprocess.Popen(["cat {} | xargs -I % -n 1 -P 20 sh -c '{} %'".format(cmds, '' if silent else 'echo %;')], shell=True)
        out, err = p2.communicate()
        print(cmds)
        print("Completed!")



    def convert_audio_multiprocess(self, audios_in, audios_out, sr, codec):
        #"cat cmds.sh | xargs -I {} -n 1 -P 24 sh -c 'echo \"{}\"; {}'"
        cmds_file = '/home/mamoros/tmp/cmds_convert.sh'
        with open(cmds_file, 'w') as f:
            for file_in, file_out in zip(audios_in, audios_out):
                f.write("ffmpeg -hide_banner -loglevel error -i '{}' -ac 1 -ar {} -acodec {} '{}' -n\n".format(
                        file_in,
                        sr,
                        codec,
                        file_out,
                        ))

        self.run_commands_multiprocess(cmds_file,silent=True)



    def extract_figerprint(self, folder_in, folder_out, fp_type):
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
        self.run_commands_multiprocess(cmds_file)



    def match(self, folder_fps, index, folder_out):
        try:
            mkdir(folder_out)
        except:
            pass
        cmds_file = '/home/mamoros/tmp/cmds_match.sh'
        fps = listdir(folder_fps)
        with open(cmds_file, 'w') as f:
            for fp in fps:
                f.write("fpmatcher identify  -q {fp} -i {index} -c fp1 > {match}\n".format(
                    fp=join(folder_fps, fp),
                    index=index,
                    match=join(folder_out, fp.replace('fp1','csv'))
                ))
        self.run_commands_multiprocess(cmds_file)
        # join all csv created in one and drop empty rows
        print(listdir(folder_out))
        matches = [join(folder_out, x) for x in listdir(folder_out) if x != 'matches.csv']
        df = concat(map(read_csv, matches), ignore_index=True)
        df.dropna(inplace=True)
        df['Query'] = df.apply(lambda row: row.Query.split('/')[-1], axis=1)
        df['Reference'] = df.apply(lambda row: row.Reference.split('/')[-1], axis=1)
        # adapt column names to baf-dataset/compute_statistics.py
        df.columns = ['query', 'query_start', 'query_end', 'reference', 'ref_start', 'ref_end', 'score', 'max_score']
        df.to_csv(join(folder_out, 'matches.csv'), index = False)



    def compute_statistics(self, matches_dir, dataset_dir):
        """
        docker run --rm -it -v /home/mamoros/build/baf-dataset:/baf-dataset -v /home/mamoros/exp/datasets/dataset_4/testing_set/:/testing_set
        mamoros:baf-dataset python3 baf-dataset/compute_statistics.py testing_set/matches/matches.csv testing_set/annotations.csv
        """
        with open("/home/mamoros/tmp/output.log", "a") as output:
            call(constants.DOCKER_RUN.format(params="-it --rm ",
                                        vol_code="/home/mamoros/build/baf-dataset:/baf-dataset",
                                        vol_data='%s:/testing_set -v %s:/matches' % (dataset_dir, matches_dir),
                                        name="baf_compute_statistics",
                                        img="mamoros:baf-dataset",
                                        cmd="python3 baf-dataset/compute_statistics.py matches/matches.csv testing_set/annotations.csv"),
                 shell=True,
                 stdout=output,
                 stderr=output)


    def create_index(self, folder):
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
        self.run_commands_multiprocess(cmd_file)



    def create_test_set(self, mode):
        mkdir(constants.SNIPPETS_DIR % str(self.last_dataset + 1))
        out_dir = join(constants.SNIPPETS_DIR % str(self.last_dataset + 1), 'testing_set')
        mkdir(out_dir)
        mkdir(join(out_dir, 'clean'))
        mkdir(join(out_dir, 'noisy'))

        # Set the data directories
        data_dirs = ['/srv/nfs/bmat_core/fingerprinting_qa/collections/siae_venues_microphone_vol1', '/srv/nfs/bmat_core/fingerprinting_qa/collections/siae_venues_microphone_vol2']

        # Load the dataset file into a DataFrame
        df = read_csv('/data/mamoros/exp/datasets/real/groundtruth.csv')
        df['query_track'] = df['query_track'].str.split('/').str[-1]
        df['reference_track'] = df['reference_track'].str.split('/').str[-1]
        to_drop=[]
        for _, row in df.iterrows():
            query_track = row['query_track']
            reference_track = row['reference_track']
            # Check if both tracks exist in either data directory
            found_query = False
            found_reference = False
            for data_dir in data_dirs:
                if exists(join(data_dir, 'queries', query_track)) or found_query:
                    found_query = True
                    query_track = join(data_dir, 'queries', query_track)
                if exists(join(data_dir, 'references', reference_track)) or found_reference:
                    found_reference = True
                    reference_track = join(data_dir, 'references', reference_track)

            # Update the DataFrame with the new values
            df.at[_, 'query_track'] = query_track
            df.at[_, 'reference_track'] = reference_track
            if not found_query and not found_reference:
                to_drop.append(_)

        df.drop(to_drop, inplace=True)
        ref_indices = {}
        query_indices = {}
        for i, row in df.iterrows():
            query_track = row['query_track']
            reference_track = row['reference_track']
            if query_track not in query_indices:
                query_indices[query_track] = len(query_indices)
            if reference_track not in ref_indices:
                ref_indices[reference_track] = len(ref_indices)
            # Update the DataFrame with the indices for the query and reference tracks
            df.at[i, 'query_filename'] = "fileid_%s.fp1" % query_indices[query_track]
            df.at[i, 'reference_filename'] = "fileid_%s.fp1" % ref_indices[reference_track]
        print(len(query_indices))
        print(len(ref_indices))
        df_annotations = df[['query_filename', 'reference_filename', 'query_begin_time', 'query_end_time']]
        df_annotations.columns = ['query', 'reference', 'query_start', 'query_end']
        df_annotations['x_tag'] = 'unanimity'
        df_annotations.to_csv(join(out_dir, 'annotations.csv'), index=False)
        df.to_csv(join(out_dir, 'test.csv'), index=False)
        queries_in = df['query_track'].drop_duplicates()
        queries_out = join(out_dir, 'noisy/') +  df['query_filename'].drop_duplicates().apply(lambda x: x.split('.')[0]) + '.wav'
        df['ref_name'] = df.reference_track.str.split('/').str[-1]
        df = df.drop_duplicates('ref_name')
        refs_in = df['reference_track'].drop_duplicates()
        refs_out = join(out_dir, 'clean/') +  df['reference_filename'].drop_duplicates().apply(lambda x: x.split('.')[0]) + '.wav'
        df.to_csv(join(out_dir, 'test.csv'), index=False)

        self.convert_audio_multiprocess(refs_in, refs_out, 8000,'pcm_s16le')
        self.convert_audio_multiprocess(queries_in, queries_out, 8000,'pcm_s16le')

        self.extract_figerprint(join(out_dir, 'clean'), join(out_dir, 'clean_fp1'), 'fp1')

        self.create_index(join(out_dir, 'clean_fp1'))

        self.extract_figerprint(join(out_dir, 'noisy'), join(out_dir, 'noisy_fp1'), 'fp1')

        self.match(
            join(out_dir, 'noisy_fp1'),
            '/data/mamoros/exp/datasets/dataset_%s/testing_set/clean_fp1/index' % str(self.last_dataset + 1),
            join(out_dir, 'matches'))

        self.compute_statistics(join(out_dir, 'matches'), out_dir)



    def create_dataset_real(self, mode):
        if mode == ['1', '3']:
            dataset_path = join(constants.EXP_DIR, "datasets/dataset_%s/" % str(self.last_dataset + 1))
            mkdir(dataset_path)
            mkdir(join(dataset_path, 'clean'))
            mkdir(join(dataset_path, 'noisy'))
            self.cp_nfsdataset_audio2snippet('/data/mamoros/exp/datasets/real/better_alignment.csv',
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