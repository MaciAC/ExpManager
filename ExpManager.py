from json import load, dump
from os import mkdir, listdir
from os.path import join
import constants
from DatasetManager import DatasetManager
from subprocess import call
from pandas import concat, read_csv

class ExpManager:

    def __init__(self):
        self.experiments = {}

        try:
            listdir(constants.EXP_DIR)
        except FileNotFoundError:
            mkdir(constants.EXP_DIR)
            mkdir(join(constants.EXP_DIR,'datasets'))
        self.load_experiments()
        self.datasetManager=DatasetManager()


    def load_experiments(self):
        max_id = 0
        for exp in [x for x in listdir(constants.EXP_DIR) if 'exp' in x]:
            with open(join(constants.EXP_DIR, exp, 'config.json')) as json_file:
                data = load(json_file)
                exp_id = int(data['id'])
                self.experiments[exp_id] = data
                if exp_id > max_id:
                    max_id = exp_id
        self.last_exp = max_id

    def create_exp(self):
        exp_id = self.last_exp + 1
        print("Creating new experiment with ID {}".format(exp_id))
        curr_path = join(constants.EXP_DIR, "exp_%s" % str(exp_id))
        mkdir(curr_path)
        mkdir(join(curr_path, 'tensorboard'))
        mkdir(join(curr_path, 'checkpoint'))
        print("Getting config from previous exp")
        with open(join(constants.EXP_DIR, "exp_%s" % str(exp_id-1), "config.json")) as json_file:
            data = load(json_file)
        data['id'] = exp_id
        dataset_id = data['dataset']
        print(dataset_id)
        if dataset_id in self.datasetManager.datasets.keys():
            print("Using dataset %s" % dataset_id)
        else:
            print("No dataset related to id %s" % dataset_id)
            self.datasetManager.create_dataset()
            data['dataset'] = self.datasetManager.last_dataset
        with open(join(constants.EXP_DIR, "exp_%s" % str(exp_id), "config.json"), 'w') as json_file:
            dump(data, json_file)


    def print_experiments(self):
        print('Experimets:')
        for e in self.experiments.values():
            optimization = e['train_config']['optimization']
            print(" Id: {}\n\t Dataset id: {}\n\t Description:{}\n\t" \
                  " lr: {}\n\t momentum: {}\n\t epochs: {}\n\n".format(
                                                                e['id'],
                                                                e['dataset'],
                                                                e['description'],
                                                                optimization['learning_rate'],
                                                                optimization['momentum'],
                                                                optimization['n_epochs']))


    def print_datasets_and_experiments(self):
        print('Datasets:')
        for d in self.datasetManager.datasets.values():
            print(" Dataset id: {}\n Type: {}\n Description: {}\n Size: {}\n".format(
                                                                d['id'],
                                                                d['type'],
                                                                d['description'],
                                                                d['size']))
        self.print_experiments()


    def choose_experiment(self):
        self.print_experiments()
        ok = False
        options = [str(x) for x in range(len(self.experiments))]
        while not ok:
                option = input( "Choose experiment id" )
                if option in options:
                    ok = True
        return int(option)


    def run_exp(self):
        exp_id = self.choose_experiment()
        exp = self.experiments[exp_id]
        dataset_id = exp["dataset"]
        vol_data = "/home/mamoros/exp/datasets/dataset_%s:/dataset " \
                   "-v /home/mamoros/exp/exp_%s:/exp/" % (dataset_id, exp_id)
        with open("/home/mamoros/tmp/output.log", "a") as output:
            call(constants.DOCKER_RUN.format(params="-it --rm --gpus all",
                                        vol_code="/home/mamoros/build/CleanUNet:/model",
                                        vol_data=vol_data,
                                        name="CleanUNet",
                                        img="mamoros:CleanUNet",
                                        cmd="python3 model/train.py -c exp/config.json"),
                 shell=True,
                 stdout=output,
                 stderr=output)


    def serve_tensorboard(self):
        exp_id = self.choose_experiment()
        with open("/home/mamoros/tmp/output.log", "a") as output:
                call(constants.DOCKER_RUN.format(params="-it --rm -p 8050:8050",
                                            vol_code="/home/mamoros/exp/exp_%s/tensorboard/:/tensorboard" % str(exp_id),
                                            vol_data="/home/mamoros/exp/exp_%s/tensorboard/:/tensorboard" % str(exp_id),
                                            name="tensorboard",
                                            img="tensorflow/tensorflow",
                                            cmd="tensorboard --logdir /tensorboard --port 8050 --bind_all"),
                    shell=True,
                    stdout=output,
                    stderr=output)


    def inference(self):
        exp_id = self.choose_experiment()
        exp = self.experiments[exp_id]
        dataset_id = exp["dataset"]
        vol_data = "/home/mamoros/exp/datasets/dataset_%s:/dataset " \
                   "-v /home/mamoros/exp/exp_%s:/exp/" % (dataset_id, exp_id)
        with open("/home/mamoros/tmp/output.log", "a") as output:
            call(constants.DOCKER_RUN.format(params="-it --rm --gpus all --shm-size=24g",
                                        vol_code="/home/mamoros/build/CleanUNet:/model",
                                        vol_data=vol_data,
                                        name="CleanUNet_denoise",
                                        img="mamoros:CleanUNet",
                                        cmd="python3 model/denoise.py -c exp/config.json"),
                 shell=True,
                 stdout=output,
                 stderr=output)


    def match(self, folder_fps, index, folder_out):
        try:
            mkdir(folder_out)
        except:
            pass
        input('WARNING first run previous Commands!, press intro once done')
        cmds_file = '/home/mamoros/tmp/cmds_match.sh'
        fps = listdir(folder_fps)
        with open(cmds_file, 'w') as f:
            for fp in fps:
                f.write("fpmatcher identify  -q {fp} -i {index} -c fp1 > {match}\n".format(
                    fp=join(folder_fps, fp),
                    index=index,
                    match=join(folder_out, fp.replace('fp1','csv'))
                ))
        print("cat {} | xargs -I % -n 1 -P 8 sh -c 'echo %; %'".format(cmds_file))



    def transcode(self, folder_in, folder_out, sr, codec):
        #"cat cmds.sh | xargs -I {} -n 1 -P 24 sh -c 'echo \"{}\"; {}'"
        try:
            mkdir(folder_out)
        except:
            pass
        cmds_file = '/home/mamoros/tmp/cmds_transcode.sh'
        in_files = listdir(folder_in)
        with open(cmds_file, 'w') as f:
            for file in in_files:
                f.write("ffmpeg -i '{}' -ac 1 -ar {} -acodec {} '{}' -n\n".format(
                        join(folder_in, file),
                        sr,
                        codec,
                        join(folder_out, file)
                        ))
        print("cat {} | xargs -I % -n 1 -P 8 sh -c 'echo %; %'".format(cmds_file))



    def join_matches(self, folder):
        input('WARNING first run previous Commands!, press intro once done')
        matches = [join(folder, x) for x in listdir(folder)]
        df = concat(map(read_csv, matches), ignore_index=True)
        df.dropna(inplace=True)
        df.to_csv(join(folder, 'matches.csv'))


    def evaluation(self):
        """

        evaluate -> https://github.com/guillemcortes/baf-dataset/blob/main/compute_statistics.py
        """
        exp_id = self.choose_experiment()
        exp_dir = '/home/mamoros/exp/exp_%d' % exp_id
        with open(join(exp_dir, "config.json")) as json_file:
            data = load(json_file)
        dataset_id = data['dataset']
        self.transcode(
            join(exp_dir, 'denoised/0k/'),
            join(exp_dir, 'transcoded'),
            8000,
            'pcm_s16le')
        self.datasetManager.extract_figerprint(
            join(exp_dir, 'transcoded'),
            join(exp_dir,'denoised_fp1'),
            'fp1')
        self.match(
            join(exp_dir,'denoised_fp1'),
            join('/home/mamoros/exp/datasets/dataset_%d/testing_set/clean_fp1/index' % dataset_id),
            join(exp_dir, 'matches'))
        self.join_matches(join(exp_dir, 'matches'))




    def prompt(self):
        self.print_datasets_and_experiments()

        run = True
        while run:
            ok = False
            options = [str(x) for x in range(8)]
            while not ok:
                option = input( "0. Close\n" \
                                "1. Create experiment\n" \
                                "2. Create dataset\n" \
                                "3. Run experiment\n" \
                                "4. Show datasets and experiments\n" \
                                "5. Monitor experiment\n" \
                                "6. Inference\n" \
                                "7. Evaluation\n")
                if option in options:
                    ok = True
            if option == '0':
                run = False
            elif option == '1':
                self.create_exp()
            elif option == '2':
                self.datasetManager.create_dataset()
            elif option == '3':
                self.run_exp()
            elif option == '4':
                self.print_datasets_and_experiments()
            elif option == '5':
                self.serve_tensorboard()
            elif option == '6':
                self.inference()
            elif option == '7':
                self.evaluation()
            else:
                raise Exception

        self.print_datasets_and_experiments()


d = ExpManager()
d.prompt()