
DOCKER_RUN = "docker run -it --rm -v {vol_code} -v {vol_data} --name {name} {img} {cmd}"
DOCKER_EXEC = "docker exec {name} {cmd}"

NFS_BASE_PATH = '/srv/nfs/bmat_core/fingerprinting_qa/collections/'

ROOT_DIR = '/home/mamoros'
EXP_DIR = ROOT_DIR + '/exp'
AUDIOFILES_DIR = ROOT_DIR + '/exp/datasets/real/audiofiles/'
SNIPPETS_DIR = ROOT_DIR + '/exp/datasets/real/'
KEYS = ['query', 'reference']

