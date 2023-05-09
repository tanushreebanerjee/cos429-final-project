import random
from clip_sampler import get_frames_from_video_path
from tqdm import trange, tqdm
import torch
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import load_pickle_file

def run_experiment(model, sampling_strategy, video_paths, batch_size = 10, outer_batch_size = 100, num_examples=100, num_frames=16, frame_rate = 1, seed=10):
  print("RUN experiment")
  random.seed(seed)
  all_indices = random.sample(range(0, len(video_paths)), num_examples)
  print(all_indices)
  new_dataset = TensorDataset(torch.IntTensor(all_indices))
  dataloader = DataLoader(new_dataset, sampler=SequentialSampler(new_dataset), batch_size=outer_batch_size)
  
  flow_dict = load_pickle_file("flow_dict.pickle")
  flow_id_dict = load_pickle_file("flow_id_dict.pickle")

  num_correct = 0
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  df = pd.DataFrame(columns = ["Actual Label", "Predicted Label", "Video Path", "Video ID", "Sampling Strategy", "Video Index", "Correct"])

  master_predicted_labels = []
  master_actual_labels = []

  j = 0
  for batch in dataloader:
    print("Iteration " + str(j) + ":")
    torch.cuda.empty_cache()
    indices = batch[0].numpy()
    model_inputs = model.preprocess_videos(indices, video_paths, sampling_strategy, flow_dict, flow_id_dict, num_frames, frame_rate)
    dataset = TensorDataset(torch.stack([x for x in model_inputs]).to(device))
    prediction_ids = model.batch_predict(dataset, batch_size).cpu()
    predicted_labels = [model.model.config.id2label[pred_id.item()] for pred_id in prediction_ids]
    actual_labels = [video_paths[x].parent.name for x in indices]
    master_actual_labels = np.concatenate((master_actual_labels,actual_labels),axis=0)
    master_predicted_labels = np.concatenate((master_predicted_labels,predicted_labels),axis=0)
    j = j + 1

  video_paths_arr = []
  video_ids_arr = []
  video_index_arr = []
  sampling_strategies = [sampling_strategy] * len(all_indices)
  for i in all_indices:
    video_path = video_paths[i]
    video_paths_arr.append(video_path)
    video_ids_arr.append(str(video_path.stem)[:11])
    video_index_arr.append(i)

  df = pd.DataFrame(list(zip(master_actual_labels, master_predicted_labels, video_paths_arr, video_ids_arr, sampling_strategies, video_index_arr)),
               columns =["Actual Label", "Predicted Label", "Video Path", 
                         "Video ID", "Sampling Strategy", "Video Index"])
  
  df["Correct"] = df["Actual Label"] == df["Predicted Label"]

  accuracy = sum(df["Correct"]) / len(df)
  print("Accuracy: " + str(accuracy))

  output_file_path = "outputs/" + sampling_strategy + "-" + str(seed) + "-" + str(num_examples) + "-" + str(num_frames) + ".csv"
  df.to_csv(output_file_path, index=False)


    



