ROOT = ".."
DATA_DIR = f"{ROOT}/data"

DEVICE = "cpu" # Set to GPU or CPU

# Model
MODEL_REPO = "facebookresearch/pytorchvideo"
MODEL_NAME = "slowfast_r50"

# Model configs
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

# Dataset
KINETICS_400_DATASET_URL = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
KINETICS_400_DATASET_FILENAME = f"{DATA_DIR}/kinetics_classnames.json"

# sample video
SAMPLE_VIDEO_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
SAMPLE_VIDEO_PATH = f'{DATA_DIR}/archery.mp4'