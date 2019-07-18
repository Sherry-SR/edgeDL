from collections import defaultdict
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def is_interesting_tag(tag):
    if 'val' in tag or 'train' in tag or 'test' in tag:
        return True
    else:
        return False


def parse_events_file(path: str) -> pd.DataFrame:
    metrics = defaultdict(list)
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:

            if isinstance(v.simple_value, float) and is_interesting_tag(v.tag):
                metrics[v.tag].append(v.simple_value)
    metrics_df = pd.DataFrame({k: v for k,v in metrics.items() if len(v) > 1})
    return metrics_df


dataset = parse_events_file(path = './checkpoints/pelvis/unet3d/logs/events.out.tfevents.1563368101.SH-IDC1-10-5-38-150')
dataset.plot()
plt.show()