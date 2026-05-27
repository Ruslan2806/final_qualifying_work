import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def smooth_data(data, weight=0.8):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def draw_ssd_results(log_path):
    with open(log_path, 'r') as f:
        data = yaml.safe_load(f)
        log_data = data['training_log']

    epochs = [item['epoch'] for item in log_data]
    
    metrics_list = [
        ('train/loss', [item['loss'] for item in log_data]),
        ('metrics/f1', [item['f1'] for item in log_data]),
        ('metrics/precision(B)', [item['precision'] for item in log_data]),
        ('metrics/recall(B)', [item['recall'] for item in log_data]),
        ('metrics/mAP50(B)', [item['mAP50'] for item in log_data])
    ]

    
    fig = plt.figure(figsize=(15, 10))
    
    positions = [
        (0, 0), (0, 1),  
        (1, 0), (1, 1), (1, 2) 
    ]

    for i, (title, values) in enumerate(metrics_list):
        ax = plt.subplot2grid((2, 6), (0, 1+i*2) if i < 2 else (1, (i-2)*2), colspan=2)
        
        ax.plot(epochs, values, color='#1f77b4', marker='o', markersize=5, 
                label='results', linewidth=2, clip_on=False)
        
        smoothed = smooth_data(values, weight=0.6)
        ax.plot(epochs, smoothed, color='#ff7f0e', linestyle='--', 
                label='smooth', linewidth=2)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('epoch', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.legend(fontsize=10)

    plt.tight_layout(pad=3.0)
    output_name = "ssd_metrics_grid.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"График сохранен в {output_name}")
    plt.show()

if __name__ == "__main__":
    log_file = Path(__file__).parent / "training_log.yaml"
    if log_file.exists():
        draw_ssd_results(log_file)
    else:
        print("Файл training_log.yaml не найден!")