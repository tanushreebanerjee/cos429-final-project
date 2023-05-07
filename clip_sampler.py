import numpy as np
import av
from utils import get_frames_from_container, get_all_frames_from_container
from utils import save_pickle_file, get_flow_from_frames
import random
# random.seed(0)
# np.random.seed(0)
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny").cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_obj_detection(frame_type, images, num_frames):
    num_objects_list = [0] * len(images)
    # Get number of objs per frame
    for i in range(len(images)):
        image = images[i]
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9)[0]
        num_objects = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score.item() > 0.9:
                num_objects += 1
        num_objects_list[i] = num_objects
    frames = []
    if frame_type == "obj-detection-top16" or frame_type == "obj-detection-all":
        indices = np.sort(np.argpartition(num_objects_list, -16)[-16:])
        frames.append(images[indices])
    if frame_type == "obj-detection-low16" or frame_type == "obj-detection-all":
        indices = np.sort(np.argpartition(num_objects_list, 16)[:16])
        frames.append(images[indices])
    if frame_type == "obj-detection-top8" or frame_type == "obj-detection-all":
        indices = np.argpartition(num_objects_list, -8)[-8:]
        indices = np.sort(np.repeat(indices, 2))
        frames.append(images[indices])
    if frame_type == "obj-detection-top4" or frame_type == "obj-detection-all":
        indices = np.argpartition(num_objects_list, -4)[-4:]
        indices = np.sort(np.repeat(indices, 4))
        frames.append(images[indices])
    if frame_type == "obj-detection-top1" or frame_type == "obj-detection-all":
        indices = np.argmax(num_objects_list)
        indices = np.repeat(indices, 16)
        frames.append(images[indices])
    if frame_type == "obj-detection-mixed" or frame_type == "obj-detection-all":
        top = np.argpartition(num_objects_list, -8)[-8:]
        low = np.argpartition(num_objects_list, 8)[:8]
        indices = np.concatenate((top,low)) 
        indices = np.sort(indices)
        frames.append(images[indices])
    return frames
  
  
def sample_random_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def truly_random_indices(num_frames, frame_sample_rate, seg_len):
    random_list = random.sample(range(0, seg_len), num_frames)
    indices = np.repeat(random_list, 16/num_frames)
    indices.sort()
    return indices



def sample_fourths(num_frames, frame_sample_rate, seg_len):
    one = random.randint(0, int(0.25 * seg_len))
    two = random.randint(int(0.25 * seg_len), int(0.5 * seg_len))
    three = random.randint(int(0.5 * seg_len), int(0.75 * seg_len))
    four = random.randint(int(0.75 * seg_len), seg_len - 1)

    final = ([one] * 4) + ([two] * 4) + ([three] * 4) + ([four] * 4)
    return final

def uniform_sample_indices(seg_len, percent_of_frames):
  num_examples = percent_of_frames * seg_len
  indices = np.arange(start = 0, stop = seg_len, step = int(seg_len/num_examples))
  return indices

def sample_optical_flow_indices(frame_type, video_path, seg_len, container, flow_dict, flow_id_dict, force_reload=False):
  #0. Check if frames are cached 
  flow_values = []
  #These are the indices of the frames from which the flow was calculated
  flow_values_frame_ids = []
  video_path = str(video_path)

  if video_path in flow_dict and not force_reload:
    #flow values
    flow_values = flow_dict[video_path]
    flow_values_frame_ids = flow_id_dict[video_path]
  else:
    #1. Uniformly Sample 50% of the Frames
    flow_values_frame_ids = uniform_sample_indices(seg_len, 0.5)
    frames = get_frames_from_container(container, flow_values_frame_ids, format_type = "gray")
    #2. Calculate Flow on Those Frames
    flow_values = get_flow_from_frames(frames)
    flow_values_frame_ids = flow_values_frame_ids[1:]
    flow_dict[str(video_path)] = flow_values
    flow_id_dict[str(video_path)] = flow_values_frame_ids
    save_pickle_file(flow_dict, "flow_dict.pickle")
    save_pickle_file(flow_id_dict, "flow_id_dict.pickle")

  # #3. Sample Frames Based on Flow
  if frame_type == "optical-flow-top16":
    flow_indices = np.argpartition(flow_values,-16)[-16:]
    flow_indices.sort()
  elif frame_type == "optical-flow-low16":
    flow_indices = np.argpartition(flow_values, 16)[:16]
    flow_indices.sort()
  elif frame_type == "optical-flow-mixed":
    highest_flow_indices = np.argpartition(flow_values,-8)[-8:]
    lowest_flow_indices = np.argpartition(flow_values, 8)[:8]
    flow_indices = np.concatenate((lowest_flow_indices,highest_flow_indices),axis=0)
    flow_indices.sort()
  elif frame_type == "optical-flow-one":
    print("Eight")
    highest_flow_indices = np.argpartition(flow_values,-8)[-8:]
    flow_indices = np.repeat(highest_flow_indices, 2)
    flow_indices.sort()

  return [flow_values_frame_ids[x] for x in flow_indices]

def get_frames_from_video_path(frame_type, video_path, num_frames, frame_sample_rate, flow_dict, flow_id_dict):
    container = av.open(str(video_path.resolve()))
    seg_len = container.streams.video[0].frames

    if frame_type == "random-sequential":
      indices = sample_random_indices(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "completely-random":
      indices = truly_random_indices(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "one-random":
      indices = truly_random_indices(1, frame_sample_rate, seg_len)
      indices = indices * num_frames
    elif frame_type == "fourths":
      indices = sample_fourths(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "all":
      return get_all_frames_from_container(container)
    elif "optical-flow" in frame_type:
      indices = sample_optical_flow_indices(frame_type, video_path, seg_len, container, flow_dict, flow_id_dict)
    
    frames = get_frames_from_container(container, indices)
    return frames

def get_frames_from_video_path_all_strats(frame_type, video_path, num_frames):
    container = av.open(str(video_path.resolve()))
    seg_len = container.streams.video[0].frames
    return sample_obj_detection(frame_type, get_all_frames_from_container(container), num_frames)