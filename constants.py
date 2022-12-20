
DOCKER_RUN = "docker run {params} -v {vol_code} -v {vol_data} --name {name} {img} {cmd}"
DOCKER_EXEC = "docker exec {name} {cmd}"

NFS_BASE_PATH = '/srv/nfs/bmat_core/fingerprinting_qa/collections/'

ROOT_DIR = '/data/mamoros'
EXP_DIR = ROOT_DIR + '/exp'
AUDIOFILES_DIR = ROOT_DIR + '/exp/datasets/real/audiofiles/'
SNIPPETS_DIR = ROOT_DIR + '/exp/datasets/dataset_%s/'
KEYS = ['query', 'reference']

DATASET_MODES={
    "1": "training",
    "2": "testing",
    "3": "validating"
}

