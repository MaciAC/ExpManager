from json import load, dump
from os import mkdir, listdir
from os.path import join
import constants
from DatasetManager import DatasetManager
from subprocess import call

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
            call(constants.DOCKER_RUN.format(params="-it --rm --gpus all",
                                        vol_code="/home/mamoros/build/CleanUNet:/model",
                                        vol_data=vol_data,
                                        name="CleanUNet_denoise",
                                        img="mamoros:CleanUNet",
                                        cmd="python3 model/denoise.py -c exp/config.json"),
                 shell=True,
                 stdout=output,
                 stderr=output)


    def evaluation(self):
        """
        multi process xargs

        cat cmds.sh | xargs -I {} -n 1 -P 24 sh -c 'echo "{}"; {}'



        all audios transcoded

        ffmpeg -i /srv/nfs/bmat_core/fingerprinting_qa/collections/siae_venues_microphone_vol1/queries/rec207397491__1440_1622.wav 
                -ac 1 -ar 8000 -acodec pcm_s16le test.wav


        extraxt fp

        fpextractor test.wav test.fp1 fp1


        make list of fp filenames

        cat refs.lst
        test2.fp1
        test3.fp1
        test.fp1


        create index

        emolina@machin:~$ fpmatcher create_index refs.lst index
        info - pid:0x00000a6e - 2022-12-07 09:01:06.369480:  populated with 3 valid fingerprints in 3.41208ms (1.13736ms/fingerprint)


        match

        fpmatcher identify  -q  test.fp1 -i index -c fp1 > matches.csv
        info - pid:0x00000a73 - 2022-12-07 09:01:39.935810:  'fpmatcher@index' identified 181360.0ms of 181360.0ms from fingerprint 'test.fp1' in 11.6ms (15608.6xRT)
        emolina@machin:~$ cat matches.csv
        "Query","Query begin time","Query end time","Reference","Reference begin time","Reference end time","Confidence","Max confidence"
        "","","","","","","",""
        "test.fp1","0.048","181.84","test2.fp1","0.048","181.84","203.167","214"

        evaluate -> https://github.com/guillemcortes/baf-dataset/blob/main/compute_statistics.py
        """
        exp_id = self.choose_experiment()

        


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
            elif option == '6':
                self.evaluation()
            else:
                raise Exception

        self.print_datasets_and_experiments()


d = ExpManager()
d.prompt()