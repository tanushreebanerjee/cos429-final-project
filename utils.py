import av
import numpy as np

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


