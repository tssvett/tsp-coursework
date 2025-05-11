import warnings

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_acovf
from tabulate import tabulate  # Для красивого вывода таблицы

warnings.filterwarnings('ignore')  # Игнорируем предупреждения

import numpy as np
import matplotlib.pyplot as plt


def read_file(filepath: str = '10.txt') -> np.array:
    """Чтение данных из файла"""
    with open(filepath, 'r', encoding='utf-8') as file:
        file.readline()  # Пропускаем заголовок
        data = [float(line[0:8]) for line in file]
    return np.array(data)


def calculate_statistics(data: np.array) -> tuple:
    """Вычисление выборочного среднего, исправленной дисперсии и стандартного отклонения"""
    mean_val = np.mean(data)
    variance = np.var(data, ddof=1)  # Несмещённая оценка
    std_dev = np.std(data, ddof=1)
    return mean_val, variance, std_dev


def calculate_sample_correlation(data: np.array, max_lag: int) -> tuple:
    """
    Вычисление выборочной корреляционной функции (КФ) и нормированной КФ (НКФ)
    с использованием исправленной формулы.
    """
    n = len(data)
    mean_val = np.mean(data)
    variance = np.var(data, ddof=1)
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
    """Построение графика нормированной корреляционной функции с порогом и интервалом корреляции"""
    plt.figure(figsize=(10, 6))
    plt.stem(lags, ncf, basefmt=" ", use_line_collection=True)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(1 / np.e, color='red', linestyle='--', label='Порог 1/e')
    plt.axvline(corr_interval, color='green', linestyle='--', label=f'Интервал корреляции = {corr_interval}')
    plt.xlabel('Сдвиг (лаг), k')
    plt.ylabel('Нормированная корреляционная функция')
    plt.title('Нормированная корреляционная функция процесса')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def control_point_1():
    max_lag = 10
    # Чтение данных
    process = read_file()

    # Вычисление статистик
    mean_val, variance, std_dev = calculate_statistics(process)
    print(f"Выборочное среднее: {mean_val:.4f}")
    print(f"Выборочная дисперсия (исправленная): {variance:.4f}")
    print(f"Стандартное отклонение: {std_dev:.4f}")

    # Визуализация процесса
    plot_process_fragment(process)

    # Вычисление корреляционных функций
    R, ncf = calculate_sample_correlation(process, max_lag)
    corr_interval = estimate_correlation_interval(ncf)

    # Вывод таблицы с КФ и НКФ
    print_correlation_table(R, ncf)

    # Визуализация нормированной корреляционной функции
    lags = np.arange(max_lag + 1)
    plot_normalized_correlation(lags, ncf, corr_interval)

    print(f"\nИнтервал корреляции: {corr_interval} отсчёта")

    return process


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


def plot_normalized_correlation(lags: np.array, ncf: np.array, corr_interval: int = None):
    """Визуализация нормированной корреляционной функции"""
    plt.figure(figsize=(12, 6))
    plt.plot(lags, ncf, 'b-', label='НКФ')
    plt.axhline(1 / np.e, color='r', linestyle='--', label='1/e')
    if corr_interval is not None:
        plt.axvline(corr_interval, color='g', linestyle=':', label=f'Интервал корреляции = {corr_interval}')
        plt.scatter(corr_interval, ncf[corr_interval], color='red', s=100, zorder=5)
    plt.xlabel('Лаг')
    plt.ylabel('Корреляция')
    plt.title('Нормированная корреляционная функция')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


def fit_ar_model(data: np.array, order: int) -> tuple:
    """
    Построение модели авторегрессии АР(M)
    Возвращает коэффициенты модели и дисперсию шума
    """
    model = AutoReg(data, lags=order, old_names=False)
    results = model.fit()
    coefficients = results.params  # [const, ar.L1, ar.L2, ...]
    noise_variance = results.sigma2
    return coefficients, noise_variance


def theoretical_ar_ncf(coefficients: np.array, max_lag: int) -> np.array:
    """
    Расчет теоретической НКФ для модели АР(M)
    """
    ar_coeffs = coefficients[1:]  # Игнорируем константу
    order = len(ar_coeffs)

    # Уравнения Юла-Уокера для автокорреляций
    r = np.zeros(max_lag + 1)
    r[0] = 1.0  # Нормировка

    if order == 0:  # АР(0) - белый шум
        return r[:max_lag + 1]

    # Заполняем r[1..order] с помощью уравнений Юла-Уокера
    for k in range(1, order + 1):
        r[k] = -np.sum(ar_coeffs[:k] * r[k - 1::-1])

    # Рекурсивно заполняем остальные значения
    for k in range(order + 1, max_lag + 1):
        r[k] = -np.sum(ar_coeffs * r[k - 1:k - order - 1:-1])

    return r


def plot_ar_models(process: np.array, max_order: int = 3, max_lag: int = 10):
    """
    Строит график сравнения выборочной НКФ с теоретическими НКФ моделей АР
    """
    _, empirical_ncf = calculate_sample_correlation(process, max_lag)
    lags = np.arange(0, max_lag + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(lags, empirical_ncf, 'ko-', label='Выборочная НКФ')

    for order in range(max_order + 1):
        coeffs, noise_var = fit_ar_model(process, order)
        theoretical_ncf = theoretical_ar_ncf(coeffs, max_lag)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)
        plt.plot(lags, theoretical_ncf, '--', label=f'АР({order}), ошибка={error:.4f}')

    plt.xlabel('Лаг')
    plt.ylabel('НКФ')
    plt.title('Сравнение выборочной и теоретических НКФ для моделей АР')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_ar_models_table(process: np.array, max_order: int = 3):
    """
    Выводит таблицу с коэффициентами моделей АР и ошибками
    """
    columns = ['B1', 'B2', 'B3', 'A0', 'error']
    df = pd.DataFrame(index=range(max_order + 1), columns=columns)
    df.index.name = 'M'

    _, empirical_ncf = calculate_sample_correlation(process, 10)  # max_lag=10 для ошибки

    best_order = 0
    best_error = np.inf

    for order in range(max_order + 1):
        coeffs, noise_var = fit_ar_model(process, order)
        theoretical_ncf = theoretical_ar_ncf(coeffs, 10)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)

        row_data = []
        for i in range(1, 4):
            row_data.append(f"{coeffs[i]:.4f}" if i <= order else "NaN")
        row_data.extend([f"{coeffs[0]:.4f}", error])
        df.loc[order] = row_data

        if error < best_error:
            best_error = error
            best_order = order

    print("\nТаблица коэффициентов моделей АР и ошибок:")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".4f"))
    print(f"\nЛучшая модель: АР({best_order}) с ошибкой {best_error:.4f}")


def control_point_2():
    """Пункт 3.2: Построение и сравнение моделей АР"""
    process = read_file()
    print("\nАнализ моделей авторегрессии:")
    plot_ar_models(process)
    print_ar_models_table(process)


def solve_ma_system(R, order):
    """Решает систему уравнений для параметров СС(N) модели"""
    from scipy.optimize import fsolve
    import numpy as np

    def equations(params):
        eqs = []
        alpha = params
        # Уравнения для корреляционной функции
        for k in range(order + 1):
            sum_ = 0.0
            for j in range(order + 1 - k):
                sum_ += alpha[j] * alpha[j + k]
            eqs.append(sum_ - R[k])
        return eqs

    # Начальное приближение
    initial_guess = np.sqrt(np.abs(R[0])) * np.random.rand(order + 1)

    try:
        solution = fsolve(equations, initial_guess, xtol=1e-6, maxfev=1000)
        # Проверка на вещественность решения
        if np.all(np.isreal(solution)):
            return solution
        return None
    except:
        return None


def theoretical_ma_ncf(alpha, max_lag):
    """Вычисляет теоретическую НКФ для модели СС"""
    r = np.zeros(max_lag + 1)
    q = len(alpha) - 1  # Порядок модели

    # Расчет корреляционной функции
    for k in range(max_lag + 1):
        if k > q: break
        r[k] = np.sum(alpha[:q + 1 - k] * alpha[k:])

    # Нормировка
    r = r / r[0]
    return r[:max_lag + 1]


def calculate_model_error(empirical_ncf, theoretical_ncf):
    """Вычисляет среднеквадратичную ошибку"""
    return np.mean((empirical_ncf - theoretical_ncf) ** 2)


def control_point_3():
    """Пункт 3.3: Построение моделей скользящего среднего"""
    process = read_file()
    max_lag = 10
    R, empirical_ncf = calculate_sample_correlation(process, max_lag)

    # Таблица для результатов
    results = {
        'N': [],
        'alpha0': [],
        'alpha1': [],
        'alpha2': [],
        'alpha3': [],
        'error': []
    }

    for n in range(4):
        # Решаем систему уравнений
        solution = solve_ma_system(R, n)

        if solution is None or np.any(np.iscomplex(solution)):
            # Модель не существует
            results['N'].append(n)
            results['alpha0'].append('Не существует')
            for i in range(1, 4): results[f'alpha{i}'].append('-')
            results['error'].append('-')
            continue

        # Расчет теоретической НКФ
        theoretical_ncf = theoretical_ma_ncf(solution, max_lag)
        error = calculate_model_error(empirical_ncf, theoretical_ncf)

        # Заполнение таблицы
        results['N'].append(n)
        for i in range(4):
            key = f'alpha{i}'
            if i <= n:
                results[key].append(f"{solution[i]:.4f}")
            else:
                results[key].append('-')
        results['error'].append(f"{error:.4f}")

    # Вывод таблицы
    df = pd.DataFrame(results)
    print("\nТаблица 3 – Модели скользящего среднего СС(N)")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


if __name__ == '__main__':
    # control_point_1()
    #control_point_2()
     control_point_3()
