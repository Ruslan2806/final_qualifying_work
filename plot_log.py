import json
import matplotlib.pyplot as plt
import numpy as np

def plot_telemetry():
    try:
        with open("telemetry_log.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Файл telemetry_log.json не найден!")
        return

    t = [e["timestamp"] for e in data]
    z_raw = [e["z_raw"] for e in data]
    z_filt = [e["z_filt"] for e in data]
    
    v_raw = [0]
    for i in range(1, len(z_raw)):
        dt = t[i] - t[i-1]
        v_raw.append((z_raw[i] - z_raw[i-1]) / dt)
    
    v_filt = [e["v_filt"] for e in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, z_raw, 'r--', alpha=0.5, label='Сырые данные')
    ax1.plot(t, z_filt, 'b-', linewidth=2, label='Фильтр Калмана')
    ax1.set_title('Сравнение дистанции')
    ax1.set_ylabel('Расстояние (м)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, np.array(v_raw) * 3.6, 'r--', alpha=0.5, label='Сырая скорость (дифференцирование)')
    ax2.plot(t, np.array(v_filt) * 3.6, 'g-', linewidth=2, label='Скорость (Калман + EMA)')
    ax2.set_title('Сравнение скорости сближения')
    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Скорость (км/ч)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_graph.png")
    plt.show()

if __name__ == "__main__":
    plot_telemetry()