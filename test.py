from model import VideoMAEModel
from clip_sampler import get_frames_from_video_path
from importlib import reload
import av
import json
from tqdm import trange, tqdm
import numpy as np
from experiments import run_experiment
import pickle
import json

np.random.seed(0)
with open('kinetics-dataset/video_paths.pickle', 'rb') as f:
    video_paths = pickle.load(f)
    
with open('data/kinetics400/validate/validate.json') as f:
    annotations_dict = json.load(f)

model = VideoMAEModel()  
run_experiment(model, "random-sequential", video_paths, batch_size = 10)