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
        step = e.step
        for v in e.summary.value:
            if isinstance(v.simple_value, float) and is_interesting_tag(v.tag):
                for k in metrics.keys():
                    if len(metrics[k]) < step :
                        metrics[k].extend([None]*(step-len(metrics[k])))
                if (metrics.get(v.tag) is None):
                    metrics[v.tag] = [None] * step
                metrics[v.tag][step - 1] = v.simple_value
    metrics_df = pd.DataFrame({k: v for k,v in metrics.items() if len(v) > 1})
    return metrics_df


dataset = parse_events_file(path = './checkpoints/pelvis/casenet2d/logs/events.out.tfevents.1564327255.SH-IDC1-10-5-38-155')
dataset[['train_eval_score_avg', 'val_eval_score_avg']].interpolate().plot()
dataset[['train_loss_avg', 'val_loss_avg']].interpolate().plot()
plt.show()