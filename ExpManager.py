from json import load
from os import mkdir, listdir
from os.path import join
import constants
from DatasetManager import DatasetManager
from subprocess import call

class ExpManager:

    def __init__(self):
        self.experiments = []

        try:
            listdir(constants.EXP_DIR)
        except FileNotFoundError:
            mkdir(constants.EXP_DIR)
            mkdir(join(constants.EXP_DIR,'datasets'))

        max_id = 0
        for exp in [x for x in listdir(constants.EXP_DIR) if 'exp' in x]:
            with open(join(constants.EXP_DIR, exp, 'config.json')) as json_file:
                data = load(json_file)
                self.experiments.append(data)
                if int(data['id']) > max_id:
                    max_id = int(data['id'])
        self.last_exp = max_id

        self.datasetManager=DatasetManager()



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


    def print_datasets_and_experiments(self):
        print('Datasets:')
        for d in self.datasetManager.datasets.values():
            print(" Dataset id: {}\n Type: {}\n Description: {}\n Size: {}".format(
                                                                d['id'],
                                                                d['type'],
                                                                d['description'],
                                                                d['size']))
        print('Experimets:')
        for e in self.experiments:
            print(" Id: {}\n Dataset id: {}\n Description:{}".format(
                                                                e['id'],
                                                                e['dataset'],
                                                                e['description']))


    def choose_experiment(self):
        print('Experimets:')
        for e in self.experiments:
            print(" Id: {}\n Dataset id: {}\n Description:{}".format(
                                                                e['id'],
                                                                e['dataset'],
                                                                e['description']))
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


    def prompt(self):
        self.print_datasets_and_experiments()

        run = True
        while run:
            ok = False
            options = [str(x) for x in range(6)]
            while not ok:
                option = input( "0. Close\n" \
                                "1. Create experiment\n" \
                                "2. Create dataset\n" \
                                "3. Run experiment\n" \
                                "4. Show datasets and experiments\n" \
                                "5. Monitor experiment\n" )
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
            else:
                raise Exception

        self.print_datasets_and_experiments()


d = ExpManager()
d.prompt()