import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_acovf
from tabulate import tabulate  # Для красивого вывода таблицы

warnings.filterwarnings('ignore')  # Игнорируем предупреждения


def read_file() -> np.array:
    """Чтение данных из файла"""
    with open('10.txt', 'r', encoding='utf-8') as file:
        file.readline()  # Пропускаем заголовок
        data = [float(line[0:8]) for line in file]
    return np.array(data)


def calculate_statistics(data: np.array) -> tuple:
    """Вычисление статистических характеристик"""
    mean_val = np.mean(data)
    variance = np.var(data, ddof=0)
    std_dev = np.std(data, ddof=0)
    return mean_val, variance, std_dev


def calculate_normalized_correlation(data: np.array, max_lag: int = 100) -> np.array:
    """Вычисление нормированной корреляционной функции"""
    centered = data - np.mean(data)
    correlation = np.correlate(centered, centered, mode='full')
    correlation = correlation[correlation.size // 2:correlation.size // 2 + max_lag + 1]
    return correlation / correlation[0]


def estimate_correlation_interval(ncf: np.array, threshold: float = 1 / np.e) -> int:
    """Оценка интервала корреляции"""
    for lag, value in enumerate(ncf):
        if value < threshold:
            return lag
    return len(ncf) - 1


def control_point_1():
    """Пункт 3.1: Анализ исходного процесса"""
    process = read_file()
    mean_val, variance, std_dev = calculate_statistics(process)
    print(f"Выборочное среднее: {mean_val:.4f}")
    print(f"Выборочная дисперсия: {variance:.4f}")
    print(f"Стандартное отклонение: {std_dev:.4f}")

    # plot_process_fragment(process)

    max_lag = 10
    ncf = calculate_normalized_correlation(process, max_lag)
    corr_interval = estimate_correlation_interval(ncf)

    # plot_normalized_correlation(lags, ncf, corr_interval)
    print(f"Интервал корреляции: {corr_interval} отсчёта")

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
    empirical_ncf = calculate_normalized_correlation(process, max_lag)
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

    empirical_ncf = calculate_normalized_correlation(process, 10)  # max_lag=10 для ошибки

    best_order = 0
    best_error = np.inf

    for order in range(max_order + 1):
        coeffs, noise_var = fit_ar_model(process, order)
        theoretical_ncf = theoretical_ar_ncf(coeffs, 10)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)

        if order == 0:
            df.loc[order] = ['Unstable', 'Unstable', 'Unstable', 'Unstable', error]
        else:
            row_data = []
            for i in range(1, 4):
                row_data.append(f"{coeffs[i]:.6f}" if i <= order else "NaN")
            row_data.extend([f"{coeffs[0]:.6f}", error])
            df.loc[order] = row_data

        if error < best_error:
            best_error = error
            best_order = order

    print("\nТаблица коэффициентов моделей АР и ошибок:")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".6f"))
    print(f"\nЛучшая модель: АР({best_order}) с ошибкой {best_error:.6f}")


def control_point_2():
    """Пункт 3.2: Построение и сравнение моделей АР"""
    process = read_file()
    print("\nАнализ моделей авторегрессии:")
    plot_ar_models(process)
    print_ar_models_table(process)


def fit_ma_model(process: np.array, order: int) -> tuple:
    """
    Подгонка модели СС(N) методом максимума правдоподобия
    Возвращает коэффициенты и дисперсию шума
    """
    model = sm.tsa.SARIMAX(process, order=(0, 0, order), trend='c')
    results = model.fit(disp=False)
    coefficients = results.params[:-1]  # Игнорируем дисперсию шума
    noise_variance = results.params[-1]
    return coefficients, noise_variance


def theoretical_ma_ncf(coefficients: np.array, max_lag: int) -> np.array:
    """
    Расчет теоретической НКФ для модели СС(N)
    """
    order = len(coefficients)
    r = np.zeros(max_lag + 1)
    r[0] = 1.0  # Нормировка

    if order == 0:  # СС(0) - белый шум
        return r

    # Вычисляем автокорреляции для лагов 1..order
    denom = 1 + np.sum(coefficients ** 2)
    for k in range(1, order + 1):
        if k <= order:
            r[k] = coefficients[k - 1] / denom
            if k < order:
                r[k] += np.sum([coefficients[i] * coefficients[i + k] for i in range(order - k)]) / denom
        else:
            r[k] = 0.0

    return r[:max_lag + 1]


def plot_ma_models(process: np.array, max_order: int = 3, max_lag: int = 10):
    """
    Строит график сравнения выборочной НКФ с теоретическими НКФ моделей СС
    """
    empirical_ncf = calculate_normalized_correlation(process, max_lag)
    lags = np.arange(0, max_lag + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(lags, empirical_ncf, 'ko-', label='Выборочная НКФ')

    for order in range(max_order + 1):
        coeffs, noise_var = fit_ma_model(process, order)
        theoretical_ncf = theoretical_ma_ncf(coeffs, max_lag)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)
        plt.plot(lags, theoretical_ncf, '--', label=f'СС({order}), ошибка={error:.4f}')

    plt.xlabel('Лаг')
    plt.ylabel('НКФ')
    plt.title('Сравнение выборочной и теоретических НКФ для моделей СС')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_ma_models_table(process: np.array, max_order: int = 3):
    """
    Выводит таблицу с коэффициентами моделей СС и ошибками
    """
    columns = ['θ1', 'θ2', 'θ3', 'A0', 'error']
    df = pd.DataFrame(index=range(max_order + 1), columns=columns)
    df.index.name = 'N'

    empirical_ncf = calculate_normalized_correlation(process, 10)

    best_order = 0
    best_error = np.inf

    for order in range(max_order + 1):
        coeffs, noise_var = fit_ma_model(process, order)
        theoretical_ncf = theoretical_ma_ncf(coeffs, 10)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)

        if order == 0:
            df.loc[order] = ['Unstable', 'Unstable', 'Unstable', 'Unstable', error]
        else:
            row_data = []
            for i in range(3):
                row_data.append(f"{coeffs[i]:.6f}" if i < order else "NaN")
            row_data.extend([f"{noise_var:.6f}", error])
            df.loc[order] = row_data

        if error < best_error:
            best_error = error
            best_order = order

    print("\nТаблица коэффициентов моделей СС и ошибок:")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".6f"))
    print(f"\nЛучшая модель: СС({best_order}) с ошибкой {best_error:.6f}")


def control_point_3():
    """Пункт 3.3: Анализ моделей скользящего среднего"""
    process = read_file()
    print("\nАнализ моделей скользящего среднего:")

    # Сначала график
    plot_ma_models(process)

    # Затем таблица
    print_ma_models_table(process)


def fit_arma_model(process: np.array, ar_order: int, ma_order: int) -> tuple:
    """
    Подгонка модели ARMA(ar_order, ma_order)
    Возвращает (ar_coeffs, ma_coeffs, intercept, error)
    """
    try:
        model = ARIMA(process, order=(ar_order, 0, ma_order))
        results = model.fit()

        ar_coeffs = results.arparams if ar_order > 0 else np.array([])
        ma_coeffs = results.maparams if ma_order > 0 else np.array([])
        intercept = results.params[-1]  # Константа

        # Расчет ошибки через сравнение НКФ
        theoretical_ncf = theoretical_arma_ncf(ar_coeffs, ma_coeffs, 10)
        empirical_ncf = calculate_normalized_correlation(process, 10)
        error = np.mean((empirical_ncf - theoretical_ncf) ** 2)

        return ar_coeffs, ma_coeffs, intercept, error

    except:
        # В случае ошибки подгонки модели
        return np.nan, np.nan, np.nan, np.inf


def theoretical_arma_ncf(ar_coeffs: np.array, ma_coeffs: np.array, max_lag: int) -> np.array:
    """
    Расчет теоретической НКФ для модели ARMA
    """
    # Для простоты используем встроенную функцию из statsmodels

    ar = np.r_[1, -ar_coeffs] if len(ar_coeffs) > 0 else np.array([1])
    ma = np.r_[1, ma_coeffs] if len(ma_coeffs) > 0 else np.array([1])

    acovf = arma_acovf(ar, ma, nobs=max_lag + 1)
    acf = acovf / acovf[0]  # Нормировка

    return acf[:max_lag + 1]


def print_arma_models_table(process: np.array, max_order: int = 3):
    """
    Выводит таблицу с результатами для моделей ARMA
    """
    columns = ['AR1', 'AR2', 'AR3', 'MA1', 'MA2', 'MA3', 'Const', 'error']
    index = pd.MultiIndex.from_product(
        [range(1, max_order + 1), range(1, max_order + 1)],
        names=['M', 'N']
    )
    df = pd.DataFrame(index=index, columns=columns)

    best_error = np.inf
    best_model = (0, 0)

    for m in range(1, max_order + 1):
        for n in range(1, max_order + 1):
            ar, ma, const, error = fit_arma_model(process, m, n)

            # Заполняем строку таблицы
            row = []
            for i in range(1, 4):
                row.append(f"{ar[i - 1]:.6f}" if i <= m and not np.isnan(ar).any() else "NaN")
            for i in range(1, 4):
                row.append(f"{ma[i - 1]:.6f}" if i <= n and not np.isnan(ma).any() else "NaN")
            row.append(f"{const:.6f}" if not np.isnan(const) else "NaN")
            row.append(error)

            df.loc[(m, n)] = row

            if error < best_error:
                best_error = error
                best_model = (m, n)

    print("\nТаблица моделей ARMA(M,N):")
    print(tabulate(df.reset_index(), headers='keys', tablefmt='grid', floatfmt=".6f"))
    print(f"\nЛучшая модель: ARMA({best_model[0]},{best_model[1]}) с ошибкой {best_error:.6f}")

    return best_model


def check_stability(ar_coeffs: np.array):
    """
    Проверка устойчивости AR части модели
    """
    if len(ar_coeffs) == 0:
        return True

    # Характеристический полином: 1 - ar1*z - ar2*z^2 - ... = 0
    poly = np.r_[1, -ar_coeffs]
    roots = np.roots(poly)

    # Все корни должны быть по модулю > 1
    return all(np.abs(roots) > 1)


def check_invertibility(ma_coeffs: np.array):
    """
    Проверка обратимости MA части модели
    """
    if len(ma_coeffs) == 0:
        return True

    # Характеристический полином: 1 + ma1*z + ma2*z^2 + ... = 0
    poly = np.r_[1, ma_coeffs]
    roots = np.roots(poly)

    # Все корни должны быть по модулю > 1
    return all(np.abs(roots) > 1)


def plot_best_nkf(process, ar_coeffs, ma_coeffs, best_m, best_n):
    """
    Визуализация сравнения НКФ для лучшей модели
    """
    empirical_ncf = calculate_normalized_correlation(process, 10)
    theoretical_ncf = theoretical_arma_ncf(ar_coeffs, ma_coeffs, 10)
    lags = np.arange(0, 11)

    plt.figure(figsize=(12, 6))
    plt.plot(lags, empirical_ncf, 'ko-', label='Выборочная НКФ')
    plt.plot(lags, theoretical_ncf, 'r--', label=f'ARMA({best_m},{best_n}) Теоретическая НКФ')
    plt.xlabel('Лаг')
    plt.ylabel('НКФ')
    plt.title(f'Сравнение НКФ для лучшей модели ARMA({best_m},{best_n})')
    plt.legend()
    plt.grid(True)
    plt.show()


def control_point_4():
    """Пункт 3.4: Анализ смешанных ARMA моделей"""
    process = read_file()
    print("\nАнализ смешанных ARMA моделей:")

    # Построение и сравнение моделей
    best_m, best_n = print_arma_models_table(process)

    # Проверка устойчивости лучшей модели
    ar_coeffs, ma_coeffs, _, _ = fit_arma_model(process, best_m, best_n)

    print("\nПроверка устойчивости лучшей модели:")
    print(f"AR часть (M={best_m}): {'Устойчива' if check_stability(ar_coeffs) else 'Неустойчива'}")
    print(f"MA часть (N={best_n}): {'Обратима' if check_invertibility(ma_coeffs) else 'Необратима'}")

    plot_best_nkf(process, ar_coeffs, ma_coeffs, best_m, best_n)


if __name__ == '__main__':
    control_point_1()
    control_point_2()
    control_point_3()
    control_point_4()
