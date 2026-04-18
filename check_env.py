"""Проверка окружения: Python, TensorFlow, CUDA, GPU.

Запуск:
    python check_env.py

Ожидаемый вывод на Fedora 43 + RTX 3060 + TF 2.15:
    Python:     3.10.x
    TensorFlow: 2.15.x
    CUDA доступна: True
    GPU-устройства: [PhysicalDevice(...)]
    Имя GPU:    NVIDIA GeForce RTX 3060
    VRAM:       12288 МБ
"""
import sys
import platform


def main() -> int:
    print("=" * 60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ ДЛЯ ИАР")
    print("=" * 60)

    # Python
    print(f"Python:     {platform.python_version()}")
    if sys.version_info[:2] not in {(3, 9), (3, 10), (3, 11)}:
        print("  ВНИМАНИЕ: TensorFlow 2.15 поддерживает только Python 3.9-3.11.")

    # TensorFlow
    try:
        import tensorflow as tf
    except ImportError as e:
        print(f"\nОШИБКА: не удалось импортировать TensorFlow ({e})")
        print("Установите: pip install 'tensorflow[and-cuda]==2.15.1'")
        return 1

    print(f"TensorFlow: {tf.__version__}")

    # CUDA и GPU
    cuda_available = tf.test.is_built_with_cuda()
    print(f"TF собран с CUDA: {cuda_available}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU-устройства: {gpus}")

    if not gpus:
        print("\nОШИБКА: GPU не виден TensorFlow.")
        print("Диагностика:")
        print("  1. Проверить драйвер:     nvidia-smi")
        print("  2. Проверить LD_LIBRARY:  echo $LD_LIBRARY_PATH")
        print("  3. Проверить cuDNN:       python -c 'import nvidia.cudnn'")
        return 1

    # Детали GPU
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", "неизвестно")
        print(f"  GPU {i}: {name}")

    # Проверка, что реально можем посчитать на GPU
    print("\nТест вычислений на GPU:")
    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            _ = c.numpy()  # принудительно выполнить
        print("  Умножение матриц 1000×1000 на GPU: OK")
    except Exception as e:
        print(f"  ОШИБКА при вычислениях: {e}")
        return 1

    # Оперативная память GPU
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total, free = result.stdout.strip().split(", ")
            print(f"  VRAM: {total} МБ всего, {free} МБ свободно")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # nvidia-smi не критичен

    print("\n" + "=" * 60)
    print("ОКРУЖЕНИЕ ГОТОВО К ОБУЧЕНИЮ МОДЕЛЕЙ ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
