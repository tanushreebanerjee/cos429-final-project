import torch
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm
from clip_sampler import get_frames_from_video_path, get_frames_from_video_path_all_strats
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import get_all_frames_from_container

class VideoMAEModel():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").cuda()

    def preprocess_videos(self, indices, video_paths, sampling_strategy, flow_dict, flow_id_dict, num_frames=16, frame_rate=1):
      inputs = []
      for i in tqdm(indices, desc = "Preprocessing:"):
        video_path = video_paths[i]
        video_frames = get_frames_from_video_path(sampling_strategy, video_path, num_frames, frame_rate, flow_dict, flow_id_dict)
        inp = self.image_processor(list(video_frames), return_tensors="pt")
        inputs.append(inp["pixel_values"][0])
      return inputs


    def batch_predict(self, dataset, batch_size = 10):
      dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
      predicted_labels = torch.empty(0, dtype = torch.int8).to(self.device)

      for batch in tqdm(dataloader, desc = "Evaluating:"):
        with torch.no_grad():
          outputs = self.model(batch[0])
          logits = outputs.logits
        preds = torch.argmax(logits, dim = 1)
        predicted_labels = torch.cat((predicted_labels, preds))

      return predicted_labels
    
    def batch_predict_no_tqdm(self, dataset, batch_size = 10):
      dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
      predicted_labels = torch.empty(0, dtype = torch.int8).to(self.device)

      for batch in dataloader:
        with torch.no_grad():
          outputs = self.model(batch[0])
          logits = outputs.logits
        preds = torch.argmax(logits, dim = 1)
        predicted_labels = torch.cat((predicted_labels, preds))

      return predicted_labels
    
    # Return a list of preprocessed frames for each different obj-detection sampling strategy
    def preprocess_videos_all(self, indices, video_paths, sampling_strategy, NUM_SAMPLING_STRATS, num_frames=16, frame_rate=1):
      inputs = [[] for i in range(NUM_SAMPLING_STRATS)]
      for i in tqdm(indices, desc = "Preprocessing:"):
        video_path = video_paths[i]
        video_frames, video_frame_indices = get_frames_from_video_path_all_strats(sampling_strategy, video_path, num_frames)
        for j in range(len(video_frames)):
          experiment_frames = video_frames[j]
          inp = self.image_processor(list(experiment_frames), return_tensors="pt")
          inputs[j].append(inp["pixel_values"][0])
      
      output = []
      for input in inputs:
        all_inputs = torch.stack(input).to(self.device)
        output.append(all_inputs)
      return output, video_frame_indices
    
    def preprocess_all_frames(self, indices, video_paths, stride=5):
      inputs = []
      video_frame_idxs = []
      for i in tqdm(indices, desc = "Preprocessing:"):
        video_frame_idxs_temp = []
        video_path = video_paths[i]
        container = av.open(str(video_path.resolve()))
        video_frames = get_all_frames_from_container(container)
        inputs_per_video = []
        for j in range(0, len(video_frames), stride):
          experiment_frames = video_frames[[j] * 16] 
          video_frame_idxs_temp.append(j)
          inp = self.image_processor(list(experiment_frames), return_tensors="pt")
          inputs_per_video.append(inp["pixel_values"][0])
        inputs.append(inputs_per_video)
        video_frame_idxs.append(video_frame_idxs_temp)
      return inputs, video_frame_idxs
