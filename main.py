import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
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


from scipy.optimize import fsolve
import numpy as np


# --- Функция для подсчёта ошибки ---
def eps(theor, emp):
    # Сумма квадратов разностей по первым 11 лагам (k=0..10)
    return np.sum((theor[:11] - emp[:11]) ** 2)


# --- Функции для смешанных моделей АРСС(M,N) ---

def apcc11f(p):
    b1, a0, a1 = p
    return korr[2] / korr[1] - b1, b1 * korr[1] + a0 ** 2 + a1 ** 2 + a1 * b1 * a0 - korr[0], b1 * korr[0] + a1 * a0 - \
           korr[1]


def apcc11(korr):
    print("АРСС(1,1)")
    b1, a0, a1 = fsolve(apcc11f, (0, np.sqrt(korr[0]), 0))
    p = b1, a0, a1
    if np.linalg.norm(apcc11f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc11f(p), 1))
        if abs(b1) < 1:
            print("Устойчива")
            ap11 = np.zeros(11)
            ap11[0] = korr[0] / d
            ap11[1] = korr[1] / d
            ap11[2] = korr[2] / d
            for i in range(3, 11):
                ap11[i] = b1 * ap11[i - 1]
            eps11 = eps(ap11, normcorr)
            print("eps=", eps11)
        else:
            print("Не устойчива")


def apcc12f(p):
    b1, a0, a1, a2 = p
    return korr[3] / korr[2] - b1, b1 * korr[
        1] + a0 ** 2 + a1 ** 2 + a1 * b1 * a0 + a2 * b1 * b1 * a0 + a2 * b1 * a1 + a2 ** 2 - korr[0], b1 * korr[
               0] + a1 * a0 + a0 * a2 * b1 + a1 * a2 - korr[1], b1 * korr[1] + a2 * a0 - korr[2]


def apcc12(korr):
    print("АРСС(1,2)")
    b1, a0, a1, a2 = fsolve(apcc12f, (0, np.sqrt(korr[0]), 0, 0))
    p = b1, a0, a1, a2
    if np.linalg.norm(apcc12f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc12f(p), 1))
        if abs(b1) < 1:
            print("Устойчива")
            ap12 = np.zeros(11)
            ap12[0] = korr[0] / d
            ap12[1] = korr[1] / d
            ap12[2] = korr[2] / d
            ap12[3] = korr[3] / d
            for i in range(4, 11):
                ap12[i] = b1 * ap12[i - 1]
            eps12 = eps(ap12, normcorr)
            print("eps=", eps12)
        else:
            print("Не устойчива")


def apcc13f(p):
    b1, a0, a1, a2, a3 = p
    return korr[4] / korr[3] - b1, -korr[0] + b1 * korr[
        1] + a0 ** 2 + a1 * b1 * a0 + a1 ** 2 + a2 * b1 * b1 * a0 + a2 * b1 * a1 + a2 ** 2 + a3 * (
                   b1 ** 3) * a0 + a3 * b1 * b1 * a1 + a3 * b1 * a2 + a3 ** 2, -korr[1] + b1 * korr[
               0] + a1 * a0 + a2 * b1 * a0 + a2 * a1 + a3 * b1 * b1 * a0 + a3 * b1 * a1 + a3 * a2, -korr[2] + b1 * korr[
               1] + a2 * a0 + a3 * b1 * a0 + a3 * a1, -korr[3] + b1 * korr[2] + a3 * a0


def apcc13(korr):
    print("АРСС(1,3)")
    b1, a0, a1, a2, a3 = fsolve(apcc13f, (0, np.sqrt(korr[0]), 0, 0, 0))
    p = b1, a0, a1, a2, a3
    if np.linalg.norm(apcc13f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc13f(p), 1))
        if abs(b1) < 1:
            print("Устойчива")
            ap13 = np.zeros(11)
            ap13[0] = korr[0] / d
            ap13[1] = korr[1] / d
            ap13[2] = korr[2] / d
            ap13[3] = korr[3] / d
            ap13[4] = korr[4] / d
            for i in range(5, 11):
                ap13[i] = b1 * ap13[i - 1]
            eps13 = eps(ap13, normcorr)
            print("eps=", eps13)
        else:
            print("Не устойчива")


def apcc21f(p):
    b1, b2, a0, a1 = p
    return b1 * korr[1] + b2 * korr[0] - korr[2], b1 * korr[2] + b2 * korr[1] - korr[3], -korr[0] + b1 * korr[1] + b2 * \
           korr[2] + a0 ** 2 + a1 * b1 * a0 + a1 ** 2, -korr[1] + b1 * korr[0] + b2 * korr[1] + a1 * a0


def apcc21(korr):
    print("АРСС(2,1)")
    b1, b2, a0, a1 = fsolve(apcc21f, (0, 0, np.sqrt(korr[0]), 0))
    p = b1, b2, a0, a1
    if np.linalg.norm(apcc21f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc21f(p), 1))
        if abs(b2) < 1 and abs(b1) < 1 - b2:
            print("Устойчива")
            ap21 = np.zeros(11)
            ap21[0] = korr[0] / d
            ap21[1] = korr[1] / d
            ap21[2] = korr[2] / d
            ap21[3] = korr[3] / d
            for i in range(4, 11):
                ap21[i] = b1 * ap21[i - 1] + b2 * ap21[i - 2]
            eps21 = eps(ap21, normcorr)
            print("eps=", eps21)
        else:
            print("Не устойчива")


def apcc22f(p):
    b1, b2, a0, a1, a2 = p
    return b1 * korr[2] + b2 * korr[1] - korr[3], b1 * korr[3] + b2 * korr[2] - korr[4], -korr[0] + b1 * korr[1] + b2 * \
           korr[2] + a0 ** 2 + a1 * b1 * a0 + a1 ** 2 + a2 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2), -korr[1] + b1 * korr[
               0] + b2 * korr[1] + a1 * a0 + a2 * (b1 * a0 + a1), -korr[2] + b1 * korr[1] + b2 * korr[0] + a2 * a0


def apcc22(korr):
    print("АРСС(2,2)")
    b1, b2, a0, a1, a2 = fsolve(apcc22f, (0, 0, np.sqrt(korr[0]), 0, 0))
    p = b1, b2, a0, a1, a2
    if np.linalg.norm(apcc22f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc22f(p), 1))
        if abs(b2) < 1 and abs(b1) < 1 - b2:
            print("Устойчива")
            ap22 = np.zeros(11)
            ap22[0] = korr[0] / d
            ap22[1] = korr[1] / d
            ap22[2] = korr[2] / d
            ap22[3] = korr[3] / d
            ap22[4] = korr[4] / d
            for i in range(5, 11):
                ap22[i] = b1 * ap22[i - 1] + b2 * ap22[i - 2]
            eps22 = eps(ap22, normcorr)
            print("eps=", eps22)
        else:
            print("Не устойчива")


def apcc23f(p):
    b1, b2, a0, a1, a2, a3 = p
    return b1 * korr[3] + b2 * korr[2] - korr[4], b1 * korr[4] + b2 * korr[3] - korr[5], -korr[0] + b1 * korr[1] + b2 * \
           korr[2] + a0 ** 2 + a1 * b1 * a0 + a1 ** 2 + a2 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2) + a3 * (
                   b1 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2) + b2 * (b1 * a0 + a1) + a3), -korr[1] + b1 * korr[
               0] + b2 * korr[1] + a1 * a0 + a2 * (b1 * a0 + a1) + a3 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2), -korr[
        2] + b1 * korr[1] + b2 * korr[0] + a2 * a0 + a3 * (b1 * a0 + a1), -korr[3] + b1 * korr[2] + b2 * korr[
               1] + a3 * a0


def apcc23(korr):
    print("АРСС(2,3)")
    b1, b2, a0, a1, a2, a3 = fsolve(apcc23f, (0, 0, np.sqrt(korr[0]), 0, 0, 0))
    p = b1, b2, a0, a1, a2, a3
    if np.linalg.norm(apcc23f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc23f(p), 1))
        if abs(b2) < 1 and abs(b1) < 1 - b2:
            print("Устойчива")
            ap23 = np.zeros(11)
            ap23[0] = korr[0] / d
            ap23[1] = korr[1] / d
            ap23[2] = korr[2] / d
            ap23[3] = korr[3] / d
            ap23[4] = korr[4] / d
            ap23[5] = korr[5] / d
            for i in range(6, 11):
                ap23[i] = b1 * ap23[i - 1] + b2 * ap23[i - 2]
            eps23 = eps(ap23, normcorr)
            print("eps=", eps23)
        else:
            print("Не устойчива")


def apcc31f(p):
    b1, b2, b3, a0, a1 = p
    return -korr[2] + b1 * korr[1] + b2 * korr[0], -korr[3] + b1 * korr[2] + b2 * korr[1] + b3 * korr[0], -korr[
        4] + b1 * korr[3] + b2 * korr[2] + b3 * korr[1], -korr[0] + b1 * korr[1] + b2 * korr[2] + b3 * korr[
               3] + a0 ** 2 + a1 * (b1 * a0 + a1), -korr[1] + b1 * korr[0] + b2 * korr[1] + b3 * korr[2] + a1 * a0


def apcc31(korr):
    print("АРСС(3,1)")
    b1, b2, b3, a0, a1 = fsolve(apcc31f, (0, 0, 0, np.sqrt(korr[0]), 0))
    p = b1, b2, b3, a0, a1
    if np.linalg.norm(apcc31f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc31f(p), 1))
        if abs(b3) < 1 and abs(b1 + b3) < 1 - b2 and abs(b2 - b1 * b3) < abs(1 - b3 ** 2):
            print("Устойчива")
            ap31 = np.zeros(11)
            ap31[0] = korr[0] / d
            ap31[1] = korr[1] / d
            ap31[2] = korr[2] / d
            ap31[3] = korr[3] / d
            ap31[4] = korr[4] / d
            for i in range(5, 11):
                ap31[i] = b1 * ap31[i - 1] + b2 * ap31[i - 2] + b3 * ap31[i - 3]
            eps31 = eps(ap31, normcorr)
            print("eps=", eps31)
        else:
            print("Не устойчива")


def apcc32f(p):
    b1, b2, b3, a0, a1, a2 = p
    return -korr[3] + b1 * korr[2] + b2 * korr[1] + b3 * korr[0], -korr[4] + b1 * korr[3] + b2 * korr[2] + b3 * korr[
        1], -korr[5] + b1 * korr[4] + b2 * korr[3] + b3 * korr[2], -korr[0] + b1 * korr[1] + b2 * korr[2] + b3 * korr[
               3] + a0 ** 2 + a1 * (b1 * a0 + a1) + a2 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2), -korr[1] + b1 * korr[
               0] + b2 * korr[1] + b3 * korr[2] + a1 * a0 + a2 * (b1 * a0 + a1), -korr[2] + b1 * korr[1] + b2 * korr[
               0] + b3 * korr[1] + a2 * a0


def apcc32(korr):
    print("АРСС(3,2)")
    b1, b2, b3, a0, a1, a2 = fsolve(apcc32f, (0, 0, 0, np.sqrt(korr[0]), 0, 0))
    p = b1, b2, b3, a0, a1, a2
    if np.linalg.norm(apcc32f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc32f(p), 1))
        if abs(b3) < 1 and abs(b1 + b3) < 1 - b2 and abs(b2 - b1 * b3) < abs(1 - b3 ** 2):
            print("Устойчива")
            ap32 = np.zeros(11)
            ap32[0] = korr[0] / d
            ap32[1] = korr[1] / d
            ap32[2] = korr[2] / d
            ap32[3] = korr[3] / d
            ap32[4] = korr[4] / d
            ap32[5] = korr[5] / d
            for i in range(6, 11):
                ap32[i] = b1 * ap32[i - 1] + b2 * ap32[i - 2] + b3 * ap32[i - 3]
            eps32 = eps(ap32, normcorr)
            print("eps=", eps32)
        else:
            print("Не устойчива")


def apcc33f(p):
    b1, b2, b3, a0, a1, a2, a3 = p
    return -korr[4] + b1 * korr[3] + b2 * korr[2] + b3 * korr[1], -korr[5] + b1 * korr[4] + b2 * korr[3] + b3 * korr[
        2], -korr[6] + b1 * korr[5] + b2 * korr[4] + b3 * korr[3], -korr[0] + b1 * korr[1] + b2 * korr[2] + b3 * korr[
               3] + a0 ** 2 + a1 * (b1 * a0 + a1) + a2 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2) + a3 * (
                   b1 * (b1 * (b1 * a0 + a1) + b2 * a0 + a2) + b2 * (b1 * a0 + a1) + b3 * a0 + a3), -korr[1] + b1 * \
           korr[0] + b2 * korr[1] + b3 * korr[2] + a1 * a0 + a2 * (b1 * a0 + a1) + a3 * (
                   b1 * (b1 * a0 + a1) + b2 * a0 + a2), -korr[2] + b1 * korr[1] + b2 * korr[0] + b3 * korr[
               1] + a2 * a0 + a3 * (b1 * a0 + a1), -korr[3] + b1 * korr[2] + b2 * korr[1] + b3 * korr[0] + a3 * a0


def apcc33(korr):
    print("АРСС(3,3)")
    b1, b2, b3, a0, a1, a2, a3 = fsolve(apcc33f, (0, 0, 0, np.sqrt(korr[0]), 0, 0, 0))
    p = b1, b2, b3, a0, a1, a2, a3
    if np.linalg.norm(apcc33f(p), 1) > 0.0001:
        print("None")
    else:
        print(p)
        print(np.linalg.norm(apcc33f(p), 1))
        if abs(b3) < 1 and abs(b1 + b3) < 1 - b2 and abs(b2 - b1 * b3) < abs(1 - b3 ** 2):
            print("Устойчива")
            ap33 = np.zeros(11)
            ap33[0] = korr[0] / d
            ap33[1] = korr[1] / d
            ap33[2] = korr[2] / d
            ap33[3] = korr[3] / d
            ap33[4] = korr[4] / d
            ap33[5] = korr[5] / d
            ap33[6] = korr[6] / d
            for i in range(7, 11):
                ap33[i] = b1 * ap33[i - 1] + b2 * ap33[i - 2] + b3 * ap33[i - 3]
            eps33 = eps(ap33, normcorr)
            print("eps=", eps33)
        else:
            print("Не устойчива")


def control_point_4():
    print("\nПункт 4: Анализ смешанных моделей АРСС(M,N)")
    process = read_file()
    max_lag = 10
    global korr, normcorr, d
    korr, normcorr = calculate_correlations(process, max_lag)
    d = korr[0]

    apcc11(korr)
    print()
    apcc12(korr)
    print()
    apcc13(korr)
    print()
    apcc21(korr)
    print()
    apcc22(korr)
    print()
    apcc23(korr)
    print()
    apcc31(korr)
    print()
    apcc32(korr)
    print()
    apcc33(korr)
    print()


def model(mod):
    mod1 = mod[1000:6000]
    mo = np.mean(mod1)
    ds = np.var(mod1, ddof=1)
    korre = [np.mean((mod1[:len(mod1) - k] - mo) * (mod1[k:] - mo)) for k in range(11)]
    normkore = korre / korre[0]
    return mod1, mo, ds, korre, normkore


def plot_ncf_compare(title, normcorr, normkorrt, normkorrm, filename):
    x = np.arange(0, 11, 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, normcorr[:11], 'ko-', label="Исходный процесс")
    plt.plot(x, normkorrt[:11], 'bs--', label="Теоретическая НКФ")
    plt.plot(x, normkorrm[:11], 'r^-.', label="Моделированный процесс")
    plt.xlabel("Лаг k")
    plt.ylabel("Нормированная КФ")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def control_point_5():
    process = read_file()
    max_lag = 10
    korr, normcorr = calculate_correlations(process, max_lag)
    m = np.mean(process)
    d = korr[0]
    s = np.sqrt(d)
    ksi = np.random.normal(0, 1, size=6000)

    # --- Модель АР(3) ---
    b1, b2, b3 = 0.4818, -0.3958, -0.2517
    a0 = 16.1297
    mdap3 = m * (1 - b1 - b2 - b3)
    modap3 = np.zeros(6000)
    modap3[0] = a0 * ksi[0] + mdap3
    modap3[1] = b1 * modap3[0] + a0 * ksi[1] + mdap3
    modap3[2] = b1 * modap3[1] + b2 * modap3[0] + a0 * ksi[2] + mdap3
    for i in range(3, 6000):
        modap3[i] = b1 * modap3[i - 1] + b2 * modap3[i - 2] + b3 * modap3[i - 3] + a0 * ksi[i] + mdap3
    modap3, map3, dap3, korrap3, normkorrap3 = model(modap3)
    normkorrtap3 = np.zeros(11)
    normkorrtap3[0] = 1
    normkorrtap3[1] = b1
    normkorrtap3[2] = b1 * normkorrtap3[1] + b2
    for i in range(3, 11):
        normkorrtap3[i] = b1 * normkorrtap3[i - 1] + b2 * normkorrtap3[i - 2] + b3 * normkorrtap3[i - 3]

    # --- Модель СС(1) ---
    a0_cc1, a1_cc1, a2_cc1 = 19.5061, 9.7507, 0.0
    modcc1 = np.zeros(6000)
    modcc1[0] = a0_cc1 * ksi[0] + m
    modcc1[1] = a0_cc1 * ksi[1] + a1_cc1 * ksi[0] + m
    for i in range(2, 6000):
        modcc1[i] = a0_cc1 * ksi[i] + a1_cc1 * ksi[i - 1] + a2_cc1 * ksi[i - 2] + m
    modcc1, mcc1, dcc1, korrcc1, normkorrcc1 = model(modcc1)
    normkorrtcc1 = np.zeros(11)
    normkorrtcc1[0] = (a0_cc1 ** 2 + a1_cc1 ** 2) / (a0_cc1 ** 2 + a1_cc1 ** 2)
    normkorrtcc1[1] = (a0_cc1 * a1_cc1) / (a0_cc1 ** 2 + a1_cc1 ** 2)
    for i in range(2, 11):
        normkorrtcc1[i] = 0

    # --- Модель АРСС(2,3) ---
    b1, b2, b3 = 0.9675, -0.7111, 0.0
    a0, a1, a2, a3 = 15.9646, -8.4029, 0.3202, 0.6552
    md23 = m * (1 - b1 - b2 - b3)
    mod23 = np.zeros(6000)
    mod23[0] = a0 * ksi[0] + md23
    mod23[1] = b1 * mod23[0] + a0 * ksi[1] + a1 * ksi[0] + md23
    mod23[2] = b1 * mod23[1] + b2 * mod23[0] + a0 * ksi[2] + a1 * ksi[1] + a2 * ksi[0] + md23
    mod23[3] = b1 * mod23[2] + b2 * mod23[1] + b3 * mod23[0] + a0 * ksi[3] + a1 * ksi[2] + a2 * ksi[1] + a3 * ksi[
        0] + md23
    for i in range(4, 6000):
        mod23[i] = b1 * mod23[i - 1] + b2 * mod23[i - 2] + b3 * mod23[i - 3] + a0 * ksi[i] + a1 * ksi[i - 1] + a2 * ksi[
            i - 2] + a3 * ksi[i - 3] + md23
    mod23, m23, d23, korr23, normkorr23 = model(mod23)
    normkorrt23 = np.zeros(11)
    normkorrt23[0] = 1
    normkorrt23[1] = b1
    normkorrt23[2] = b1 * normkorrt23[1] + b2
    for i in range(3, 11):
        normkorrt23[i] = b1 * normkorrt23[i - 1] + b2 * normkorrt23[i - 2] + b3 * normkorrt23[i - 3]

    # --- Таблица параметров ---
    table = [
        ["Параметры", "Исходный процесс", "АР(3) Теор.", "АР(3) Модел.", "СС(1) Теор.", "СС(1) Модел.",
         "АРСС(2,3) Теор.", "АРСС(2,3) Модел."],
        ["M(ξ)", f"{m:.4f}", f"{m:.4f}", f"{map3:.4f}", f"{m:.4f}", f"{mcc1:.4f}", f"{m:.4f}", f"{m23:.4f}"],
        ["D(ξ)", f"{d:.4f}", f"{d:.4f}", f"{dap3:.4f}", f"{d:.4f}", f"{dcc1:.4f}", f"{d:.4f}", f"{d23:.4f}"],
        ["√D(ξ)", f"{np.sqrt(d):.4f}", f"{np.sqrt(d):.4f}", f"{np.sqrt(dap3):.4f}", f"{np.sqrt(d):.4f}",
         f"{np.sqrt(dcc1):.4f}", f"{np.sqrt(d):.4f}", f"{np.sqrt(d23):.4f}"],
    ]
    # Добавим строки НКФ
    for i in range(11):
        table.append([
            f"r({i})",
            f"{normcorr[i]:.4f}",
            f"{normkorrtap3[i]:.4f}",
            f"{normkorrap3[i]:.4f}",
            f"{normkorrtcc1[i]:.4f}",
            f"{normkorrcc1[i]:.4f}",
            f"{normkorrt23[i]:.4f}",
            f"{normkorr23[i]:.4f}"
        ])
    # Добавим строки с погрешностями
    table.append([
        "Погрешность",
        "",
        f"{eps(normkorrtap3, normcorr):.4f}",
        f"{eps(normkorrap3, normcorr):.4f}",
        f"{eps(normkorrtcc1, normcorr):.4f}",
        f"{eps(normkorrcc1, normcorr):.4f}",
        f"{eps(normkorrt23, normcorr):.4f}",
        f"{eps(normkorr23, normcorr):.4f}"
    ])
    print("\nСравнительная таблица параметров и НКФ:")
    print(tabulate(table, headers="firstrow", tablefmt="grid", numalign="center", stralign="center"))

    # --- Графики ---
    plot_ncf_compare("Сравнение НКФ для АР(3)", normcorr, normkorrtap3, normkorrap3, "ar3_ncf.png")
    plot_ncf_compare("Сравнение НКФ для СС(1)", normcorr, normkorrtcc1, normkorrcc1, "cc1_ncf.png")
    plot_ncf_compare("Сравнение НКФ для АРСС(2,3)", normcorr, normkorrt23, normkorr23, "arcc23_ncf.png")
    control_point_5_simulation_fragment()


def simulate_arcc23(m, b1, b2, a0, a1, a2, a3, length=6000, burn_in=1000):
    """Смоделировать процесс АРСС(2,3)"""
    ksi = np.random.normal(0, 1, size=length)
    md = m * (1 - b1 - b2)
    eta = np.zeros(length)
    eta[0] = a0 * ksi[0] + md
    eta[1] = b1 * eta[0] + a0 * ksi[1] + a1 * ksi[0] + md
    eta[2] = b1 * eta[1] + b2 * eta[0] + a0 * ksi[2] + a1 * ksi[1] + a2 * ksi[0] + md
    eta[3] = b1 * eta[2] + b2 * eta[1] + a0 * ksi[3] + a1 * ksi[2] + a2 * ksi[1] + a3 * ksi[0] + md
    for i in range(4, length):
        eta[i] = (b1 * eta[i - 1] + b2 * eta[i - 2] +
                  a0 * ksi[i] + a1 * ksi[i - 1] + a2 * ksi[i - 2] + a3 * ksi[i - 3] + md)
    # Отбрасываем первые burn_in отсчётов
    return eta[burn_in:]


def plot_process_fragment(process, points_to_plot=150):
    """Визуализация фрагмента случайного процесса"""
    mean_value = np.mean(process)
    std_dev = np.std(process, ddof=1)
    plt.figure(figsize=(10, 6))
    plt.plot(process[:points_to_plot], label='Смоделированный процесс', color='blue', alpha=0.7)
    plt.axhline(mean_value, color='orange', label=f'Среднее = {mean_value:.3f}', linewidth=2)
    plt.axhline(mean_value + std_dev, color='green', linestyle='--', label=f'Среднее + σ = {mean_value + std_dev:.3f}')
    plt.axhline(mean_value - std_dev, color='red', linestyle='--', label=f'Среднее - σ = {mean_value - std_dev:.3f}')
    plt.ylabel('Значение процесса')
    plt.xlabel('Время, отсчёты')
    plt.title('Фрагмент смоделированного процесса по АРСС(2,3)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def control_point_5_simulation_fragment():
    m = 40.7955  # среднее исходного процесса
    b1, b2 = 0.9675, -0.7111
    a0, a1, a2, a3 = 15.9646, -8.4029, 0.3202, 0.6552

    # Смоделировать процесс
    process_sim = simulate_arcc23(m, b1, b2, a0, a1, a2, a3)

    # Построить фрагмент
    plot_process_fragment(process_sim, points_to_plot=150)


if __name__ == '__main__':
    # control_point_1()
    # control_point_2()
    # control_point_3()
    # control_point_4()
    control_point_5()
