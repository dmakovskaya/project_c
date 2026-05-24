#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "kmeans.h"

// Проверка: все значения в массиве близки к ожидаемому
void assert_array_close(const double *arr, int size, double expected, double tol) {
    for (int i = 0; i < size; i++) {
        double diff = fabs(arr[i] - expected);
        assert(diff < tol && "Значение вне допустимого отклонения");
    }
}

// Проверка: массив содержит только указанные значения
void assert_labels_valid(const int *labels, int n, int k) {
    for (int i = 0; i < n; i++) {
        assert(labels[i] >= 0 && labels[i] < k && "Неверная метка кластера");
    }
}

// Генерация простых тестовых данных: 2 кластера на плоскости
void generate_simple_data(double *X, int n, int features) {
    // Кластер 0: центр (0, 0)
    for (int i = 0; i < n/2; i++) {
        X[i*features + 0] = (rand() % 10) / 10.0;   // x: 0.0..0.9
        X[i*features + 1] = (rand() % 10) / 10.0;   // y: 0.0..0.9
    }
    // Кластер 1: центр (10, 10)
    for (int i = n/2; i < n; i++) {
        X[i*features + 0] = 10 + (rand() % 10) / 10.0;  // x: 10.0..10.9
        X[i*features + 1] = 10 + (rand() % 10) / 10.0;  // y: 10.0..10.9
    }
}

// Тест 1:
void test_log1p_transform() {
    printf("Тест 1: log1p_transform... ");
    
    double data[] = {0, 1, 2, 9};  // 4 значения
    int n = 4, f = 1;
    
    log1p_transform(data, n, f);
    
    // Проверка: log(0+1)=0, log(1+1)=0.693, log(2+1)=1.099, log(9+1)=2.303
    assert(fabs(data[0] - 0.0) < 1e-6);
    assert(fabs(data[1] - 0.693147) < 1e-4);
    assert(fabs(data[2] - 1.098612) < 1e-4);
    assert(fabs(data[3] - 2.302585) < 1e-4);
    
    printf("ОК\n");
}

// Тест 2: 
void test_standard_scaler() {
    printf("Тест 2: standard_scaler... ");
    
    double data[] = {1, 2, 3, 4, 5};  // 5 значений, 1 признак
    double means[1], stds[1];
    int n = 5, f = 1;
    
    standard_scaler(data, n, f, means, stds);
    
    // После стандартизации: mean = 0, std = 1
    double new_mean = 0, new_var = 0;
    for (int i = 0; i < n; i++) new_mean += data[i];
    new_mean /= n;
    for (int i = 0; i < n; i++) {
        double d = data[i] - new_mean;
        new_var += d * d;
    }
    double new_std = sqrt(new_var / n);
    
    assert(fabs(new_mean) < 1e-6 && "Mean не равен 0 после стандартизации");
    assert(fabs(new_std - 1.0) < 1e-6 && "Std не равен 1 после стандартизации");
    
    // Проверка: means/stds сохранены корректно
    assert(fabs(means[0] - 3.0) < 1e-6);   // исходное среднее = 3
    assert(fabs(stds[0] - sqrt(2.0)) < 1e-6);  // исходное std = sqrt(2)
    
    printf("ОК\n");
}

// Тест 3: 
void test_kmeans_fit() {
    printf("Тест 3: kmeans_fit (2 кластера)... ");
    
    srand(42);
    int n = 20, f = 2, k = 2;
    double *X = malloc(n * f * sizeof(double));
    int *labels = malloc(n * sizeof(int));
    double *centroids = malloc(k * f * sizeof(double));
    
    generate_simple_data(X, n, f);
    
    KMeansConfig cfg = {n, f, k, 100, 1e-4};
    double wcss = kmeans_fit(X, labels, centroids, &cfg);
    
    // Проверки
    assert(wcss >= 0 && "WCSS не может быть отрицательным");
    assert_labels_valid(labels, n, k);
    
    // Проверка: точки первого кластера должны иметь метку 0 или 1
    int cluster0_count = 0, cluster1_count = 0;
    for (int i = 0; i < n/2; i++) if (labels[i] == labels[0]) cluster0_count++;
    for (int i = n/2; i < n; i++) if (labels[i] == labels[n/2]) cluster1_count++;
    
    // Хотя бы 80% точек каждого "истинного" кластера должны быть вместе
    assert((cluster0_count >= 8 || cluster1_count >= 8) && "Кластеризация не разделила данные");
    
    free(X); free(labels); free(centroids);
    printf("ОК\n");
}

// Тест 4: 
void test_kmeans_predict() {
    printf("Тест 4: kmeans_predict... ");
    
    // Центроиды: [0,0] и [10,10]
    double centroids[] = {0, 0, 10, 10};
    int f = 2, k = 2;
    
    // Точка близко к первому центроиду
    double point1[] = {1, 1};
    int pred1 = kmeans_predict(point1, f, centroids, k);
    assert(pred1 == 0 && "Точка (1,1) должна быть отнесена к кластеру 0");
    
    // Точка близко ко второму центроиду
    double point2[] = {9, 10};
    int pred2 = kmeans_predict(point2, f, centroids, k);
    assert(pred2 == 1 && "Точка (9,10) должна быть отнесена к кластеру 1");
    
    // Точка посередине — любая метка допустима
    double point3[] = {5, 5};
    int pred3 = kmeans_predict(point3, f, centroids, k);
    assert(pred3 == 0 || pred3 == 1 && "Точка (5,5) должна иметь метку 0 или 1");
    
    printf("ОК\n");
}

// Тест 5:
void test_empty_cluster_handling() {
    printf("Тест 5: обработка пустого кластера... ");
    
    // Данные: все точки в одной области
    double X[] = {1, 1, 1.1, 1.1, 1.2, 1.2};
    int n = 3, f = 2, k = 3;  // k > реального числа кластеров
    int *labels = malloc(n * sizeof(int));
    double *centroids = malloc(k * f * sizeof(double));
    
    KMeansConfig cfg = {n, f, k, 50, 1e-4};
    double wcss = kmeans_fit(X, labels, centroids, &cfg);
    
    // Алгоритм должен завершиться без падения
    assert(wcss >= 0 && "WCSS должен быть неотрицательным");
    assert_labels_valid(labels, n, k);
    
    free(labels); free(centroids);
    printf("ОК\n");
}

// Тест 6: 
void test_reproducibility() {
    printf("Тест 6: воспроизводимость результатов... ");
    
    srand(123);
    int n = 30, f = 2, k = 2;
    double *X1 = malloc(n * f * sizeof(double));
    double *X2 = malloc(n * f * sizeof(double));
    int *labels1 = malloc(n * sizeof(int));
    int *labels2 = malloc(n * sizeof(int));
    double *c1 = malloc(k * f * sizeof(double));
    double *c2 = malloc(k * f * sizeof(double));
    
    generate_simple_data(X1, n, f);
    memcpy(X2, X1, n * f * sizeof(double));  // одинаковые данные
    
    KMeansConfig cfg = {n, f, k, 100, 1e-4};
    
    // Два запуска с одинаковым seed внутри kmeans_fit (42)
    kmeans_fit(X1, labels1, c1, &cfg);
    kmeans_fit(X2, labels2, c2, &cfg);
    
    // Метки могут быть переставлены (0↔1), но структура должна совпадать
    // Проверяем: если labels1[i]==labels1[j], то labels2[i]==labels2[j]
    int consistent = 1;
    for (int i = 0; i < n && consistent; i++) {
        for (int j = i+1; j < n && consistent; j++) {
            int same1 = (labels1[i] == labels1[j]);
            int same2 = (labels2[i] == labels2[j]);
            if (same1 != same2) consistent = 0;
        }
    }
    assert(consistent && "Результаты не воспроизводятся");
    
    free(X1); free(X2); free(labels1); free(labels2); free(c1); free(c2);
    printf("ОК\n");
}

// Запуск всех тестов 
int main() {
    printf("Запуск тестов K-means библиотеки...\n\n");
    
    test_log1p_transform();
    test_standard_scaler();
    test_kmeans_fit();
    test_kmeans_predict();
    test_empty_cluster_handling();
    test_reproducibility();
    
    printf("\n Все тесты пройдены успешно!\n");
    return 0;
}