import numpy as np
import av
from utils import get_frames_from_container, get_all_frames_from_container
import random
# random.seed(0)
# np.random.seed(0)

def sample_random_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def truly_random_indices(num_frames, frame_sample_rate, seg_len):
    random_list = random.sample(range(0, seg_len), num_frames)
    random_list.sort()
    return random_list

def sample_fourths(num_frames, frame_sample_rate, seg_len):
    one = random.randint(0, int(0.25 * seg_len))
    two = random.randint(int(0.25 * seg_len), int(0.5 * seg_len))
    three = random.randint(int(0.5 * seg_len), int(0.75 * seg_len))
    four = random.randint(int(0.75 * seg_len), seg_len - 1)

    final = ([one] * 4) + ([two] * 4) + ([three] * 4) + ([four] * 4)
    return final


def get_frames_from_video_path(frame_type, video_path, num_frames, frame_sample_rate):
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
    
    frames = get_frames_from_container(container, indices)
    return frames
