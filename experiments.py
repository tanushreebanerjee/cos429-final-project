import random
from clip_sampler import get_frames_from_video_path
from tqdm import trange, tqdm
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

def run_experiment(model, sampling_strategy, video_paths, batch_size = 10, num_examples=100, num_frames=16, frame_rate = 1, seed=10):
  random.seed(seed)
  num_correct = 0
  df = pd.DataFrame(columns = ["Actual Label", "Predicted Label", "Video Path", "Video ID", "Sampling Strategy", "Video Index", "Correct"])
  indices = random.sample(range(0, len(video_paths)), num_examples)
  model_inputs = model.preprocess_videos(indices, video_paths, sampling_strategy, num_frames, frame_rate)
  dataset = TensorDataset(model_inputs)
  prediction_ids = model.batch_predict(dataset, batch_size)
  predicted_labels = [model.model.config.id2label[pred_id] for pred_id in prediction_ids]
  actual_labels = [video_paths[x].parent.name for x in indices]

  video_paths_arr = []
  video_ids_arr = []
  video_index_arr = []
  sampling_strategies = [sampling_strategy] * len(indices)
  for i in indices:
    video_path = video_paths[i]
    video_paths_arr.append(video_path)
    video_ids_arr.append(str(video_path.stem)[:11])
    video_index_arr.append(i)

  df = pd.DataFrame(list(zip(actual_labels, predicted_labels, video_paths_arr, video_ids_arr, sampling_strategies, video_index_arr)),
               columns =["Actual Label", "Predicted Label", "Video Path", 
                         "Video ID", "Sampling Strategy", "Video Index"])
  
  df["Correct"] = df["Actual Label"] == df["Predicted Label"]

  output_file_path = "outputs/" + sampling_strategy + "_" + str(seed) + ".csv"
  df.to_csv(output_file_path, index=False)


    



