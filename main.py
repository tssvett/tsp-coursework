import warnings

import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import math
from tabulate import tabulate  # Для красивого вывода таблицы

warnings.filterwarnings('ignore')  # Игнорируем предупреждения


def read_file(filepath: str = '10.txt') -> np.array:
    """Чтение данных из файла"""
    with open(filepath, 'r', encoding='utf-8') as file:
        file.readline()  # Пропускаем заголовок
        data = [float(line[0:8]) for line in file]
    return np.array(data)


def calculate_mean(process: list) -> float:
    sum: float = 0.
    for i in range(0, len(process)):
        sum += process[i]
    sum /= len(process)
    return sum


def calculate_variance(process: list) -> float:
    mean = calculate_mean(process)
    sum: float = 0.
    for i in range(0, len(process)):
        sum += (process[i] - mean) * (process[i] - mean)
    sum /= (len(process) - 1)
    return sum


def calculate_correlations(data: np.array, max_lag: int) -> tuple:
    """
    Вычисление выборочной корреляционной функции (КФ) и нормированной КФ (НКФ)
    с использованием исправленной формулы.
    """
    n = len(data)
    mean_val = calculate_mean(data)
    variance = calculate_variance(data)
    R = []
    for k in range(max_lag + 1):
        numerator = sum((data[j] - mean_val) * (data[j + k] - mean_val) for j in range(n - k))
        denominator = n - k - 1
        R_k = numerator / denominator
        R.append(R_k)
    R = np.array(R)
    r = R / variance
    return R, r


def estimate_correlation_interval(ncf: np.array, threshold: float = 1 / np.e) -> int:
    """Оценка интервала корреляции по порогу"""
    for lag, value in enumerate(ncf):
        if value < threshold:
            return lag
    return len(ncf) - 1


def print_correlation_table(R: np.array, ncf: np.array):
    """Вывод таблицы с корреляционной функцией и нормированной корреляционной функцией"""
    table_data = [[i, f"{R[i]:.4f}", f"{ncf[i]:.4f}"] for i in range(len(R))]
    headers = ["n", "КФ", "НКФ"]
    print("\nТаблица 1 – Первые значения выборочной КФ и НКФ")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center"))


def plot_process_fragment(process: np.array, points_to_plot: int = 150):
    """Визуализация фрагмента случайного процесса с линиями среднего и стандартного отклонения"""
    mean_value = np.mean(process)
    std_dev = np.std(process, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.plot(process[:points_to_plot], label='Случайный процесс', color='blue', alpha=0.7)
    plt.axhline(mean_value, color='orange', label=f'Среднее = {mean_value:.3f}', linewidth=2)
    plt.axhline(mean_value + std_dev, color='green', linestyle='--', label=f'Среднее + σ = {mean_value + std_dev:.3f}')
    plt.axhline(mean_value - std_dev, color='red', linestyle='--', label=f'Среднее - σ = {mean_value - std_dev:.3f}')
    plt.ylabel('Значение процесса')
    plt.xlabel('Время, отсчёты')
    plt.title('Анализ случайного процесса')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(mean_value - 3 * std_dev, mean_value + 3 * std_dev)
    plt.tight_layout()
    plt.show()


def plot_normalized_correlation(lags: np.array, ncf: np.array, corr_interval: int):
    """Построение графика нормированной корреляционной функции с линиями между точками"""
    plt.figure(figsize=(10, 6))

    # Рисуем линии между точками
    plt.plot(lags, ncf, 'k-', linewidth=1, label='Линия между точками')

    # Рисуем точки и вертикальные линии (stem)
    markerline, stemlines, baseline = plt.stem(
        lags,
        ncf,
        linefmt='k--',
        markerfmt='ko',
        basefmt=" "
    )

    # Настраиваем толщину линий и размер маркеров
    plt.setp(stemlines, linewidth=1)
    plt.setp(markerline, markersize=5)

    # Горизонтальные и вертикальные линии порогов
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(1 / np.e, color='red', linestyle='--', label='Порог 1/e')
    plt.axvline(corr_interval, color='green', linestyle='--', label=f'Интервал = {corr_interval}')

    plt.xlabel('Сдвиг (лаг), k')
    plt.ylabel('Нормированная корреляционная функция')
    plt.title('Нормированная корреляционная функция процесса')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_process_fragment(process):
    """Визуализация фрагмента случайного процесса"""
    POINTS_TO_PLOT = 150
    mean_value = np.mean(process)
    std_dev = np.std(process)

    plt.figure(figsize=(10, 6))
    plt.plot(process[:POINTS_TO_PLOT], label='Случайный процесс', color='blue', alpha=0.7)
    plt.axhline(mean_value, color='orange', label=f'Среднее = {mean_value:.3f}', linewidth=2)
    plt.axhline(mean_value + std_dev, color='green', linestyle='--', label=f'Среднее + σ = {mean_value + std_dev:.3f}')
    plt.axhline(mean_value - std_dev, color='red', linestyle='--', label=f'Среднее - σ = {mean_value - std_dev:.3f}')
    plt.ylabel('Значение процесса')
    plt.xlabel('Время, отсчёты')
    plt.title('Анализ случайного процесса')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(mean_value - 3 * std_dev, mean_value + 3 * std_dev)
    plt.tight_layout()
    plt.show()


def control_point_1():
    max_lag = 10
    # Чтение данных
    process = read_file()

    # Вычисление статистик
    mean_val = calculate_mean(process)
    variance = calculate_variance(process)
    _, normilized_correlation = calculate_correlations(process, max_lag)
    print(f"Выборочное среднее: {mean_val:.4f}")
    print(f"Выборочная дисперсия: {variance:.4f}")
    print(f"fНормированная корреляционная функция: {normilized_correlation}")

    # Визуализация процесса
    plot_process_fragment(process)

    # Вычисление корреляционных функций
    R, ncf = calculate_correlations(process, max_lag)
    corr_interval = estimate_correlation_interval(ncf)

    # Вывод таблицы с КФ и НКФ
    print_correlation_table(R, ncf)

    # Визуализация нормированной корреляционной функции
    lags = np.arange(max_lag + 1)
    plot_normalized_correlation(lags, ncf, corr_interval)

    print(f"\nИнтервал корреляции: {corr_interval} отсчёта")

    return process


# --- Функции для задания 2 (модели АР) ---

def findBetasAR(R: list, m: int, a_list: list = None, b_list: list = None, matr_a=None, vec_b=None):
    """
    Рекурсивное решение систем уравнений для поиска коэффициентов α и β моделей АР(M).
    Возвращает списки a_list (α0 для каждого M) и b_list (списки β для каждого M).
    """
    if a_list is None:
        a_list = []
    if b_list is None:
        b_list = []

    if m == 0:
        a_list.reverse()
        b_list.reverse()
        return a_list, b_list

    if m == 4:
        matr_a = np.array([[1, R[1], R[2], R[3]],
                           [0, R[0], R[1], R[2]],
                           [0, R[1], R[0], R[1]],
                           [0, R[2], R[1], R[0]]])
        vec_b = np.array([[R[0]], [R[1]], [R[2]], [R[3]]])

    # Решаем систему Ax = b
    res = np.linalg.solve(matr_a, vec_b).ravel()

    a0 = math.sqrt(res[0])
    a_list.append(a0)
    print(f"M = {m - 1} a = {math.sqrt(res[0]):.4f}")

    # β параметры
    b_temp = []
    for i in range(1, m):
        b_temp.append(res[i])
        print(f"Для M = {m - 1} параметр β_{i} = {res[i]:.4f}")
    b_list.append(b_temp)
    print()

    # Рекурсивный вызов для следующего порядка
    return findBetasAR(R, m - 1, a_list, b_list, matr_a[:m - 1, :m - 1], vec_b[:m - 1])


def theoretical_normalCorrelationAR(b_list, normalCorrelation_list, max_lag):
    """
    Вычисление теоретической НКФ для моделей АР(M) по найденным коэффициентам β.
    Возвращает список списков с теоретическими НКФ для каждого порядка M.
    """
    theoretical_list = []
    for M in range(4):
        buf_list = []
        for k in range(max_lag + 1):
            if k <= M:
                # Для лагов <= M теоретическая НКФ совпадает с выборочной
                buf_list.append(normalCorrelation_list[k])
                print(f"M = {M} и k = {k} theoretical r_n = {normalCorrelation_list[k]:.4f}")
            else:
                # Для лагов > M считаем по формуле
                theoretical = 0
                for m in range(1, M + 1):
                    theoretical += b_list[M][m - 1] * buf_list[k - m]
                print(f"M = {M} и k = {k} theoretical r_n = {theoretical:.4f}")
                buf_list.append(theoretical)
        theoretical_list.append(buf_list)
        print()
    return theoretical_list


def findEps(theoretical_nkf_list, normalCorrelation_list, max_lag, model_type='AR'):
    """
    Вычисление погрешности ε² для каждой модели.
    model_type: 'AR' или 'MA' для корректного вывода.
    """
    eps_list = []
    for order in range(4):
        if theoretical_nkf_list[order][0] is not None:
            eps = 0
            for k in range(1, max_lag + 1):
                eps += (theoretical_nkf_list[order][k] - normalCorrelation_list[k]) ** 2
            if model_type == 'AR':
                print(f"M = {order} eps = {eps:.6f}")
            else:
                print(f"N = {order} eps = {eps:.6f}")
            eps_list.append(eps)
        else:
            print(f"{model_type} модель порядка {order} не существует")
            eps_list.append(None)
            print()
    return eps_list


# --- Основная функция для задания 2 ---

def control_point_2():
    print("\nПункт 2: Анализ моделей авторегрессии (АР)")

    process = read_file()
    max_lag = 10
    R, ncf = calculate_correlations(process, max_lag)

    # Находим коэффициенты α и β для моделей АР(M), M=0..3
    a_list, b_list = findBetasAR(R, 4)

    # Рассчитываем теоретическую НКФ для моделей АР
    theoretical_ncf_list = theoretical_normalCorrelationAR(b_list, ncf, max_lag)

    # Рассчитываем погрешности моделей
    eps_list = findEps(theoretical_ncf_list, ncf, max_lag, model_type='AR')

    # Формируем таблицу для вывода
    table_data = []
    for M in range(4):
        if a_list[M] is not None:
            betas = b_list[M]
            betas_str = ' '.join(f"{b:.4f}" for b in betas) if betas else "-"
            table_data.append([
                f"АР({M})",
                f"{a_list[M]:.4f}",
                betas_str,
                f"{eps_list[M]:.6f}" if eps_list[M] is not None else "-"
            ])
        else:
            table_data.append([f"АР({M})", "-", "-", "-"])

    headers = ["Порядок модели", "α0", "β параметры", "ε²"]
    print("\nТаблица моделей авторегрессии АР(M):")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center"))

    # Выбираем лучшую модель по минимальной погрешности
    filtered_eps = [(i, e) for i, e in enumerate(eps_list) if e is not None]
    if filtered_eps:
        best_order, best_eps = min(filtered_eps, key=lambda x: x[1])
        print(f"\nРекомендуемая модель: АР({best_order}) с погрешностью ε² = {best_eps:.6f}")
    else:
        print("\nНе удалось построить работоспособную модель АР")


# Функция системы нелинейных уравнений для модели СС(N)
def equationsMA(seq, R, n):
    if n == 0:
        a0 = seq[0]
        return a0 ** 2 - R[0]
    elif n == 1:
        a0, a1 = seq
        return (a0 ** 2 + a1 ** 2 - R[0],
                a0 * a1 - R[1])
    elif n == 2:
        a0, a1, a2 = seq
        return (a0 ** 2 + a1 ** 2 + a2 ** 2 - R[0],
                a0 * a1 + a1 * a2 - R[1],
                a0 * a2 - R[2])
    else:
        a0, a1, a2, a3 = seq
        return (a0 ** 2 + a1 ** 2 + a2 ** 2 + a3 ** 2 - R[0],
                a0 * a1 + a1 * a2 + a2 * a3 - R[1],
                a0 * a2 + a1 * a3 - R[2],
                a0 * a3 - R[3])


# Поиск коэффициентов α для моделей СС(N)
def findAlphasMA(R):
    ans_list = []
    for n in range(4):
        # Начальное приближение: a0 = sqrt(R[0]), остальные 0
        zeros = [0.0] * n
        zeros.append(math.sqrt(R[0]))
        zeros.reverse()
        x0 = np.array(zeros)
        res = fsolve(equationsMA, x0, args=(R, n))
        norm = np.linalg.norm(equationsMA(res, R, n))
        if norm < 1e-4:
            ans_list.append(res.tolist())
            print(f"Модель СС({n}) найдена: α = {[f'{v:.4f}' for v in res]}")
        else:
            ans_list.append([None])
            print(f"Модель СС({n}) не существует")
    print()
    return ans_list

# Вычисление теоретической НКФ для моделей СС(N)
def theoretical_normalCorrelationMA(a_list, normalCorrelation_list, max_lag):
    theoretical_list = []
    for N in range(4):
        if a_list[N][0] is not None:
            buf_list = []
            for k in range(max_lag + 1):
                if k <= N:
                    # Для лагов <= N теоретическая НКФ совпадает с выборочной
                    buf_list.append(normalCorrelation_list[k])
                    print(f"N = {N} и k = {k} theoretical r_n = {normalCorrelation_list[k]:.4f}")
                else:
                    # Для лагов > N НКФ равна 0
                    buf_list.append(0)
                    print(f"N = {N} и k = {k} theoretical r_n = 0")
            theoretical_list.append(buf_list)
            print()
        else:
            theoretical_list.append([None])
            print(f"N = {N} модель не существует\n")
    return theoretical_list

# Функция для вычисления погрешности ε² для моделей СС
def findEpsMA(theoretical_nkf_list, normalCorrelation_list, max_lag):
    eps_list = []
    for order in range(4):
        if theoretical_nkf_list[order][0] is not None:
            eps = 0
            for k in range(1, max_lag + 1):
                eps += (theoretical_nkf_list[order][k] - normalCorrelation_list[k]) ** 2
            print(f"N = {order} eps = {eps:.6f}")
            eps_list.append(eps)
        else:
            print(f"Модель СС({order}) не существует")
            eps_list.append(None)
            print()
    return eps_list

# Основная функция для третьего задания
def control_point_3():
    print("\nПункт 3: Анализ моделей скользящего среднего (СС)")

    process = read_file()
    max_lag = 10
    R, ncf = calculate_correlations(process, max_lag)

    # Находим коэффициенты α для моделей СС
    a_list = findAlphasMA(R)

    # Рассчитываем теоретическую НКФ для моделей СС
    theoretical_ncf_list = theoretical_normalCorrelationMA(a_list, ncf, max_lag)

    # Рассчитываем погрешности моделей
    eps_list = findEpsMA(theoretical_ncf_list, ncf, max_lag)

    # Формируем таблицу для вывода
    table_data = []
    for N in range(4):
        if a_list[N][0] is not None:
            alphas = a_list[N]
            alphas_str = ' '.join(f"{a:.4f}" for a in alphas)
            eps_str = f"{eps_list[N]:.6f}" if eps_list[N] is not None else "-"
            table_data.append([
                f"СС({N})",
                alphas_str,
                eps_str
            ])
        else:
            table_data.append([f"СС({N})", "Модель не существует", "-"])

    headers = ["Порядок модели", "Параметры модели (α0 ... αN)", "ε²"]
    print("\nТаблица моделей скользящего среднего СС(N):")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center"))

    # Выбираем лучшую модель по минимальной погрешности
    filtered_eps = [(i, e) for i, e in enumerate(eps_list) if e is not None]
    if filtered_eps:
        best_order, best_eps = min(filtered_eps, key=lambda x: x[1])
        print(f"\nРекомендуемая модель: СС({best_order}) с погрешностью ε² = {best_eps:.6f}")
    else:
        print("\nНе удалось построить работоспособную модель СС")

# --- Запуск ---

if __name__ == '__main__':
    # Запуск первого контрольного пункта
    control_point_1()

    # Запуск второго контрольного пункта (модели АР)
    control_point_2()

    # Третий пункт (модели СС) можно добавить аналогично
    control_point_3()
