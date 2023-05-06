import torch
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm
from clip_sampler import get_frames_from_video_path
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

class VideoMAEModel():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").cuda()

    def preprocess_videos(self, indices, video_paths, sampling_strategy, num_frames=16, frame_rate=1):
      inputs = []
      for i in tqdm(indices, desc = "Preprocessing:"):
        video_path = video_paths[i]
        video_frames = get_frames_from_video_path(sampling_strategy, video_path, num_frames, frame_rate)
        inp = self.image_processor(list(video_frames), return_tensors="pt")
        inputs.append(inp["pixel_values"][0])
      all_inputs = torch.stack(inputs).to(self.device)
      return all_inputs


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


