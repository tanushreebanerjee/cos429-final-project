from model import VideoMAEModel
from clip_sampler import get_frames_from_video_path
from importlib import reload
import av
import json
from tqdm import trange, tqdm
import numpy as np
from experiments import run_experiment
from experiments_runall import run_experiment_all
import pickle
import json

np.random.seed(0)
with open('kinetics-dataset/video_paths.pickle', 'rb') as f:
    video_paths = pickle.load(f)
    
with open('data/kinetics400/validate/validate.json') as f:
    annotations_dict = json.load(f)

model = VideoMAEModel()  

run_experiment_all(model, "obj-detection-all", video_paths, seed = 20, num_frames=16, num_examples = 250, outer_batch_size = 250, batch_size = 25)