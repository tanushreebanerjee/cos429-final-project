import random
from clip_sampler import get_frames_from_video_path
from tqdm import trange, tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def run_experiment(model, sampling_strategy, video_paths, num_examples=100, num_frames=16, frame_rate = 1, seed=10):
  random.seed(seed)
  indices = random.sample(range(0, len(video_paths)), num_examples)
  print(indices)
  num_correct = 0

  df = pd.DataFrame(columns = ["Actual Label", "Predicted Label", "Video Path", "Video ID", "Sampling Strategy", "Video Index", "Correct"])

  for i in trange(num_examples):
    random_video_index = indices[i]
    video_path = video_paths[random_video_index]
    video_frames = get_frames_from_video_path(sampling_strategy, video_path, num_frames, frame_rate)
    predicted_label, label_num, logits = model.get_predicted_label(video_frames)
    actual_label = video_path.parent.name
    if predicted_label == actual_label:
        num_correct += 1

    print(f'\nActual Label: {actual_label}, Predicted Label: {predicted_label}, Accuracy: {num_correct/(i + 1)}')

    df = df.append(
      {'Actual Label': actual_label, 
        'Predicted Label': predicted_label, 
        "Video Path" : str(video_path), 
        "Video ID" : str(video_path.stem)[:11],
        "Sampling Strategy": sampling_strategy,
        "Video Index" : random_video_index,
        "Correct" : predicted_label == actual_label
      }, ignore_index = True)

  output_file_path = "outputs/" + sampling_strategy + "_" + str(seed) + ".csv"
  df.to_csv(output_file_path, index=False)


    



