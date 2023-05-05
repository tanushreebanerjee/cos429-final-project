import torch
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

class VideoMAEModel():
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    def get_predicted_label(self, video):
      inputs = self.image_processor(list(video), return_tensors="pt")
      with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits
      label_num = logits.argmax(-1).item()
      label_name = self.model.config.id2label[label_num]

      return (label_name, label_num, logits)


