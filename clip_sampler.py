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
    frame_indices = []
    if frame_type == "obj-detection-top16" or frame_type == "obj-detection-all":
        indices = np.sort(np.argpartition(num_objects_list, -16)[-16:])
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "obj-detection-low16" or frame_type == "obj-detection-all":
        indices = np.sort(np.argpartition(num_objects_list, 16)[:16])
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "obj-detection-top8" or frame_type == "obj-detection-all":
        indices = np.argpartition(num_objects_list, -8)[-8:]
        indices = np.sort(np.repeat(indices, 2))
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "obj-detection-top4" or frame_type == "obj-detection-all":
        indices = np.argpartition(num_objects_list, -4)[-4:]
        indices = np.sort(np.repeat(indices, 4))
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "obj-detection-top1" or frame_type == "obj-detection-all":
        indices = np.argmax(num_objects_list)
        indices = np.repeat(indices, 16)
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "obj-detection-mixed" or frame_type == "obj-detection-all":
        top = np.argpartition(num_objects_list, -8)[-8:]
        low = np.argpartition(num_objects_list, 8)[:8]
        indices = np.concatenate((top,low)) 
        indices = np.sort(indices)
        frames.append(images[indices])
        frame_indices.append(indices)
    return frames, frame_indices
  
  

def sample_fourths(num_frames, frame_sample_rate, seg_len):
    one = random.randint(0, int(0.25 * seg_len))
    two = random.randint(int(0.25 * seg_len), int(0.5 * seg_len))
    three = random.randint(int(0.5 * seg_len), int(0.75 * seg_len))
    four = random.randint(int(0.75 * seg_len), seg_len - 1)

    final = ([one] * 4) + ([two] * 4) + ([three] * 4) + ([four] * 4)
    return final


def sample_positions(frame_type, images, num_frames, seg_len):
    frames = []
    frame_indices = []
    if frame_type == "position-fourths" or frame_type == "position-all":
        indices = sample_fourths(num_frames, 1, seg_len)
        indices.sort()
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "position_beginning" or frame_type == "position-all":
        vals = range(int(seg_len/3))
        if len(vals) < num_frames:
            indices = random.sample(range(num_frames), num_frames)
        else:
            indices = random.sample(vals, num_frames)
        indices = np.repeat(indices, 16 // num_frames)
        indices.sort()
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "position_middle" or frame_type == "position-all":
        vals = range(int(seg_len/3), int(2*seg_len/3))
        if len(vals) < num_frames:
            border = (seg_len - num_frames) // 2
            indices = random.sample(range(border, border + num_frames), num_frames)
        else:
            indices = random.sample(vals, num_frames)
        indices = np.repeat(indices, 16 // num_frames)
        indices.sort()
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "position_end" or frame_type == "position-all":
        vals = range(int(2*seg_len/3), seg_len)
        if len(vals) < num_frames:
            indices = random.sample(range(seg_len-num_frames, seg_len), num_frames)
        else:
            indices = random.sample(vals, num_frames)
        indices = np.repeat(indices, 16 // num_frames)
        indices.sort()
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "position-mixed" or frame_type == "position-all":
        indices = [0]
        indices += random.sample(range(1, int(seg_len/3)), 5) + random.sample(range(int(seg_len/3), int(2*seg_len/3)), 5) + random.sample(range(int(2*seg_len/3), seg_len), 5)
        indices.sort()
        frames.append(images[indices])
        frame_indices.append(indices)
    if frame_type == "position-mixed-all":
        for num, repeat in [(5,1),(2,2),(1,4)]:
            indices = [0]
            indices += random.sample(range(1, int(seg_len/3)), num) + random.sample(range(int(seg_len/3), int(2*seg_len/3)), num) + random.sample(range(int(2*seg_len/3), seg_len), num)
            if num == 2:
                while len(indices) < 8:
                    idx = random.sample(range(seg_len), 1)
                    if idx not in indices:
                        indices += idx
            indices = np.repeat(indices, repeat)
            indices.sort()
            frames.append(images[indices])
            frame_indices.append(indices)
        
    return frames, frame_indices
    

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

from numpy import diff
def get_flow_indices_der(flow_values, flow_values_frame_ids, seg_len, num_frames):
  dx = 0.0001
  dy = (diff(flow_values)/dx)
  dy = np.insert(dy, 0, 0)
  # plt.plot(dy)
  dy = np.absolute(dy)

  sub_indices = np.arange(0,len(dy), 7)
  if (len(sub_indices) < 4):
    sub_indices = np.arange(0,len(dy), 4)

  avg_dy_arr = []
  for i in range(len(sub_indices)):
    if (i != len(sub_indices) - 1):
      index_start = sub_indices[i]
      index_end = sub_indices[i + 1]
    else:
      index_start = sub_indices[i]
      index_end = len(dy)
    
    avg_flow_val = np.average(dy[index_start:index_end])
    avg_dy_arr.append(avg_flow_val)

  #Gives the ranges that have the lowest derivative
  num_values_to_get = 6

  if len(avg_dy_arr) <= num_values_to_get:
    num_values_to_get = 5
  
  min_avg_dy_arr = np.argpartition(avg_dy_arr, num_values_to_get)[:num_values_to_get]
  max_avg_dy_arr = np.argpartition(avg_dy_arr,-2)[-2:]

  if len(avg_dy_arr) <= 6:
    differential = 6 - len(avg_dy_arr) 
    differential = differential + 1
    min_avg_dy_arr = np.append(min_avg_dy_arr, [0] * differential)


  avg_dy_arr_total = np.concatenate((min_avg_dy_arr,max_avg_dy_arr),axis=0)

  
  # print(len(min_avg_dy_arr))


  # plt.plot(sub_indices, avg_dy_arr)
  # plt.scatter([sub_indices[x] for x in min_avg_dy_arr], [avg_dy_arr[x] for x in min_avg_dy_arr])
  return [sub_indices[x] for x in avg_dy_arr_total]

def sample_optical_flow_indices(num_frames, frame_type, video_path, seg_len, container, flow_dict, flow_id_dict, force_reload=False):
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
  elif frame_type == "optical-flow-num":
    highest_flow_indices = np.argpartition(flow_values,-num_frames)[-num_frames:]
    flow_indices = np.repeat(highest_flow_indices, 16/num_frames)
    flow_indices.sort()
  elif frame_type == "optical-flow-constant-der":
    flow_indices = get_flow_indices_der(flow_values, flow_values_frame_ids, seg_len, num_frames)
    flow_indices = np.repeat(flow_indices, 16/num_frames)
    flow_indices.sort()

  return [flow_values_frame_ids[x] for x in flow_indices]


def uniform_sample_index(seg_len, num_frames):
  arr = np.linspace(0,seg_len, num_frames, endpoint=False)
  arr = np.rint(arr).astype(int)
  arr = np.repeat(arr, 16/num_frames)
  arr.sort()

  return arr


def get_frames_from_video_path(frame_type, video_path, num_frames, frame_sample_rate, flow_dict, flow_id_dict):
    container = av.open(str(video_path.resolve()))
    seg_len = container.streams.video[0].frames

    if frame_type == "random-sequential":
      indices = sample_random_indices(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "completely-random":
      indices = truly_random_indices(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "random-num":
      indices = truly_random_indices(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "fourths":
      indices = sample_fourths(num_frames, frame_sample_rate, seg_len)
    elif frame_type == "uniform-sample":
      indices = uniform_sample_index(seg_len, num_frames)
    elif frame_type == "all":
      return get_all_frames_from_container(container)
    elif "optical-flow" in frame_type:
      indices = sample_optical_flow_indices(num_frames, frame_type, video_path, seg_len, container, flow_dict, flow_id_dict, force_reload=False)

    return get_frames_from_container(container, indices)

def get_frames_from_video_path_all_strats(frame_type, video_path, num_frames):
    container = av.open(str(video_path.resolve()))
    seg_len = container.streams.video[0].frames
    
    if "obj-detection" in frame_type:
      return sample_obj_detection(frame_type, get_all_frames_from_container(container), num_frames)
    elif "position" in frame_type:
      return sample_positions(frame_type, get_all_frames_from_container(container), num_frames, seg_len)