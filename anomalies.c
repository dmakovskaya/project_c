#include "anomalies.h"
#include <math.h>
#include <float.h>

// Вспомогательная функция для расчета расстояния
static double get_dist(const double *a, const double *b, int f) { // a, b - векторы, f - количество признаков
    double s = 0.0; // Инициализация суммы
    for (int i = 0; i < f; i++) { // Цикл по всем признакам
        double d = a[i] - b[i]; // Вычисление разности координат
        s += d * d; // Накопление суммы квадратов разностей
    }
    return sqrt(s); // Возврат корня из суммы (Евклидово расстояние)
}

// Оценка точности модели (сравнение с истинными метками)
double calculate_accuracy(const int *labels, const int *true_labels, int n) { // labels - прогноз, true_labels - эталон, n - число точек
    int matches = 0; // Счётчик совпадений
    for (int i = 0; i < n; i++) { // Цикл по всем точкам
        if (labels[i] == true_labels[i]) matches++; // Если метки совпали, увеличиваем счётчик
    }
    return (double)matches / n * 100.0; // Возврат точности в процентах
}

// Детекция аномалий на основе расстояния до ближайшего центроида
int detect_anomalies(const double *data, int n, int f, const double *centroids, int k, double threshold, int *out_is_anomaly) { // data - данные, threshold - порог расстояния
    int count = 0; // Счётчик найденных аномалий
    for (int i = 0; i < n; i++) { // Проход по всем клиентам
        double min_d = DBL_MAX; // Инициализация минимального расстояния максимумом
        for (int c = 0; c < k; c++) { // Цикл по всем центроидам
            double d = get_dist(&data[i * f], &centroids[c * f], f); // Расстояние от точки до центроида c
            if (d < min_d) min_d = d; // Поиск ближайшего центроида
        }
        if (min_d > threshold) { // Если точка дальше порога от любого центроида
            out_is_anomaly[i] = 1; // Помечаем как аномалию
            count++; // Увеличиваем общий счетчик аномалий
        } else {
            out_is_anomaly[i] = 0; // Точка нормальная
        }
    }
    return count; // Возврат количества найденных аномалий
}