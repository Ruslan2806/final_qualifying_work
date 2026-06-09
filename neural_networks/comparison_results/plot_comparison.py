import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
REPORT_JSON = BASE_PATH / "comparison_report.json"
RESULTS_JSON = BASE_PATH / "comparison_results.json"
OUTPUT_DIR = BASE_PATH
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_accuracy_matrix(report_data):
    metrics = ["Precision", "Recall", "mAP50", "mAP50_95"]
    models = [r['model'] for r in report_data]
    
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes = axes.flatten()
    colors = {'day': '#f1c40f', 'night': '#34495e'}

    for i, m in enumerate(metrics):
        day_vals = [r['day'][m] for r in report_data]
        night_vals = [r['night'][m] for r in report_data]
        
        axes[i].bar(x - width/2, day_vals, width, label='День', 
                    color=colors['day'], edgecolor='black', alpha=0.85)
        axes[i].bar(x + width/2, night_vals, width, label='Ночь', 
                    color=colors['night'], edgecolor='black', alpha=0.95)
        
        axes[i].set_title(m, fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, fontsize=11)
        axes[i].set_ylim(0, 1.1) # Метрики от 0 до 1
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        
        for j in range(len(x)):
            axes[i].text(x[j] - width/2, day_vals[j] + 0.02, f'{day_vals[j]:.2f}', ha='center', fontsize=9)
            axes[i].text(x[j] + width/2, night_vals[j] + 0.02, f'{night_vals[j]:.2f}', ha='center', fontsize=9)

        if i == 0: 
            axes[i].legend(loc='lower left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=200)
    print("Сохранен график: accuracy_comparison.png")

def plot_latency(results_data):
    """Строит график времени обработки (Inference Time)."""
    models = [r['model'] for r in results_data]
    lats = [r['all']['Lat_ms'] for r in results_data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, lats, color='#e74c3c', edgecolor='black', width=0.6)
    
    plt.ylabel('Время обработки кадра, мс', fontsize=12)
    plt.xlabel('Архитектура нейронной сети', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f} ms', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_comparison.png", dpi=200)
    print("Сохранен график: latency_comparison.png")

def main():
    try:
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл {e.filename}")
        return

    plot_accuracy_matrix(report_data)
    plot_latency(results_data)
    
    print("\nВизуализация успешно завершена.")

if __name__ == "__main__":
    main()