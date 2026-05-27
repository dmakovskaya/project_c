#ifndef ANOMALIES_H
#define ANOMALIES_H

// Функция для оценки точности кластеризации
double calculate_accuracy(const int *labels, const int *true_labels, int n);

// Функция для поиска аномалий: возвращает количество найденных аномалий
// out_is_anomaly заполняется единицами для аномалий и нулями для нормальных точек
int detect_anomalies(const double *data, int n, int f, const double *centroids, int k, double threshold, int *out_is_anomaly);

#endif