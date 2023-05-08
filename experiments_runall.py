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

def run_experiment_all(model, sampling_strategy, video_paths, batch_size = 10, outer_batch_size = 100, num_examples=100, num_frames=16, frame_rate = 1, seed=10):
  print("run experiment")
  random.seed(seed)
  all_indices = random.sample(range(0, len(video_paths)), num_examples)
  new_dataset = TensorDataset(torch.IntTensor(all_indices))
  dataloader = DataLoader(new_dataset, sampler=SequentialSampler(new_dataset), batch_size=outer_batch_size)
  

  num_correct = 0
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if sampling_strategy == "obj-detection-all":
      SAMPLING_STRATS = ["obj-detection-top16","obj-detection-low16","obj-detection-top8","obj-detection-top4","obj-detection-top1","obj-detection-mixed"]
  elif sampling_strategy == "position-all":
      SAMPLING_STRATS = ["position-fourths","position_beginning","position_middle","position_end","position-mixed"]
  else:
      SAMPLING_STRATS = [sampling_strategy]
      
  master_predicted_labels = [[] for i in range(len(SAMPLING_STRATS))]
  master_actual_labels = [[] for i in range(len(SAMPLING_STRATS))]

  j = 0
  for batch in dataloader:
    print("Iteration " + str(j) + ":")
    torch.cuda.empty_cache()
    indices = batch[0].numpy()
    model_inputs = model.preprocess_videos_all(indices, video_paths, sampling_strategy, len(SAMPLING_STRATS), num_frames, frame_rate)
    for i in range(len(model_inputs)):
        sampling_model_inputs = model_inputs[i]
        dataset = TensorDataset(torch.stack([x for x in sampling_model_inputs]).to(device))
        prediction_ids = model.batch_predict(dataset, batch_size).cpu()
        predicted_labels = [model.model.config.id2label[pred_id.item()] for pred_id in prediction_ids]
        actual_labels = [video_paths[x].parent.name for x in indices]
        master_actual_labels[i] = np.concatenate((master_actual_labels[i],actual_labels),axis=0)
        master_predicted_labels[i] = np.concatenate((master_predicted_labels[i],predicted_labels),axis=0)
    j = j + 1

  for num in range(len(SAMPLING_STRATS)):
    sampling_strategy = SAMPLING_STRATS[num]  
    video_paths_arr = []
    video_ids_arr = []
    video_index_arr = []
    sampling_strategies = [sampling_strategy] * len(all_indices)
    for i in all_indices:
        video_path = video_paths[i]
        video_paths_arr.append(video_path)
        video_ids_arr.append(str(video_path.stem)[:11])
        video_index_arr.append(i)

    df = pd.DataFrame(list(zip(master_actual_labels[num], master_predicted_labels[num], video_paths_arr, video_ids_arr, sampling_strategies, video_index_arr)),
                columns =["Actual Label", "Predicted Label", "Video Path", 
                            "Video ID", "Sampling Strategy", "Video Index"])
    
    df["Correct"] = df["Actual Label"] == df["Predicted Label"]

    accuracy = sum(df["Correct"]) / len(df)
    print(f"Accuracy for {sampling_strategy}: " + str(accuracy))

    output_file_path = "outputs/" + sampling_strategy + "-" + str(seed) + "-" + str(num_examples) + "-" + str(num_frames) + ".csv"
    df.to_csv(output_file_path, index=False)
    


def run_theoretical_best(model, sampling_strategy, video_paths, batch_size = 10, outer_batch_size = 100, num_examples=100, num_frames=16, frame_rate = 1, seed=10):
  print("run experiment")
  random.seed(seed)
  all_indices = random.sample(range(0, len(video_paths)), num_examples)
  new_dataset = TensorDataset(torch.IntTensor(all_indices))
  dataloader = DataLoader(new_dataset, sampler=SequentialSampler(new_dataset), batch_size=outer_batch_size)
  

  num_correct = 0
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
      
  master_predicted_labels = []
  master_actual_labels = []
  master_correct_frames = []

  j = 0
  for batch in dataloader:
    print("Iteration " + str(j) + ":")
    torch.cuda.empty_cache()
    indices = batch[0].numpy()
    model_inputs, video_frame_idxs = model.preprocess_all_frames(indices, video_paths)
    actual_labels = [video_paths[x].parent.name for x in indices]
    master_actual_labels = np.concatenate((master_actual_labels,actual_labels),axis=0)
    for i in range(len(model_inputs)):
        video_processed_frames = model_inputs[i]
        correct_frames = []
        dataset = TensorDataset(torch.stack(video_processed_frames).to(device))
        prediction_ids = model.batch_predict(dataset, batch_size).cpu()
        predicted_labels = [model.model.config.id2label[pred_id.item()] for pred_id in prediction_ids]
        for frame_label_index in range(len(predicted_labels)):
            frame_label = predicted_labels[frame_label_index]
            if frame_label == actual_labels[i]:
                correct_frames.append(video_frame_idxs[i][frame_label_index])
                master_predicted_labels = np.append(master_predicted_labels,frame_label)
                break
        if len(correct_frames) == 0:
            # add the most recent predicted label
            master_predicted_labels = np.append(master_predicted_labels,frame_label)
                
        master_correct_frames.append(correct_frames)
            
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
    df["First Correct Frame"] = master_correct_frames

    accuracy = sum(df["Correct"]) / len(df)
    print(f"Accuracy: " + str(accuracy))

    output_file_path = "outputs/" + sampling_strategy + "-" + str(seed) + "-" + str(num_examples) + "-" + str(num_frames) + ".csv"
    df.to_csv(output_file_path, index=False)
    


        



