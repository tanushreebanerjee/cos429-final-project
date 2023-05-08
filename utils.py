import av
import numpy as np
import cv2 as cv
import time 
import pickle
import pandas as pd

def get_frames_from_container(container, indices, format_type="rgb24"):
  '''
  Decode the video with PyAV decoder.
  Args:
      container (`av.container.input.InputContainer`): PyAV container.
      indices (`List[int]`): List of frame indices to decode.
  Returns:
      result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
  '''
  frames = []
  container.seek(0)
  start_index = indices[0]
  end_index = indices[-1]
  curr_index = 0
  for i, frame in enumerate(container.decode(video=0)):
    if i > end_index:
        break
    if i >= start_index and i in indices:
        frames.append(frame)
        curr_index = curr_index + 1
        while (curr_index < len(indices) and indices[curr_index] == i):
          frames.append(frame)
          curr_index = curr_index + 1

  return np.stack([x.to_ndarray(format=format_type, height = 360, width=480) for x in frames])


def get_all_frames_from_container(container):
  frames = []
  container.seek(0)
  for i, frame in enumerate(container.decode(video=0)):
    frames.append(frame)
  return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def save_pickle_file(obj, file_name):
  with open(file_name, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_file(file_name):
  with open(file_name, 'rb') as handle:
      return pickle.load(handle)

def uniform_sample_indices(seg_len, percent_of_frames):
  num_examples = percent_of_frames * seg_len
  indices = np.arange(start = 0, stop = seg_len, step = int(seg_len/num_examples))
  return indices

def get_flow_from_frames(frames):
  flow_values = []
  for i in range(0, len(frames) - 1):
    flow = np.linalg.norm(cv.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0))
    flow_values.append(flow)
  return flow_values

def get_missclassified(path):
  df_1 = pd.read_csv(path)
  return df_1[df_1["Correct"] == False]["Video Index"].to_list()