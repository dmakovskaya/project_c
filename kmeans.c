#include "kmeans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Евклидово расстояние (формула) 
static double dist(const double *a, const double *b, int f) {
    double s = 0.0; //Инициализация
    for (int i = 0; i < f; i++) { // Цикл по всем признакам точки
        double d = a[i] - b[i];// Разность
        s += d * d; // Возводим в квадрат
    }
    return sqrt(s);
}

// Инициализация центроидов методом K-means++ 
static void init_centroids(const double *X, int n, int f, int k, double *centroids, unsigned seed) {
    srand(seed);                                                   // Установка seed для генератора
    int first_idx = rand() % n;                                    // Выбор первого центроида случайно
    memcpy(&centroids[0], &X[first_idx * f], f * sizeof(double));  // Копирование первой точки в центроиды

    double *min_dists = malloc(n * sizeof(double));                // Массив для хранения расстояний до ближайшего центроида
    for (int i = 0; i < n; i++) min_dists[i] = DBL_MAX;            // Инициализация расстояний бесконечностью

    for (int c = 1; c < k; c++) {                                  // Цикл для выбора остальных k-1 центроидов
        double total_sum = 0.0;                                    // Сумма квадратов расстояний
        for (int i = 0; i < n; i++) {                              // Проход по всем точкам
            double d = dist(&X[i * f], &centroids[(c - 1) * f], f);// Расстояние до последнего добавленного центроида
            if (d * d < min_dists[i]) min_dists[i] = d * d;        // Обновление минимального расстояния для точки
            total_sum += min_dists[i];                             // Накопление общей суммы весов
        }

        double r = ((double)rand() / RAND_MAX) * total_sum;        // Выбор случайного значения в диапазоне суммы
        double current_sum = 0.0;                                  // Текущая сумма для поиска точки
        for (int i = 0; i < n; i++) {                              // Поиск точки, соответствующей выбранному весу
            current_sum += min_dists[i];                           // Прибавление квадрата расстояния
            if (current_sum >= r) {                                // Если достигли порога
                memcpy(&centroids[c * f], &X[i * f], f * sizeof(double)); // Копируем эту точку как новый центроид
                break; // Выходим из цикла поиска
            }
        }
    }
    free(min_dists); // Освобождение временной памяти
}

// Присваиваем ближайшему центроиду точку
static int assign(const double *p, int f, const double *c, int k) {// Возвращает индекс ближайшего центроида для точки 
    int best = 0; // Инициализация лучшего индекса первым центроидом
    double min_d = dist(p, c, f);// Вычисление расстояния от точки до первого центроида 
    for (int i = 1; i < k; i++) {// Цикл по остальным центроидам (начиная со второго)
        double d = dist(p, &c[i * f], f);// Расстояние от точки до центроида
        if (d < min_d) { // Если текущее расстояние меньше минимального
            min_d = d; // Обновить минимальное расстояние
            best = i; }// Запомнить индекс этого центроида
    }
    return best;// Вернуть индекс ближайшего центроида
}

// Обновление центроидов как среднее точек кластера
static void update(const double *X, int n, int f, const int *lbl, int k, double *centroids) { //центроиды как средние арифметические точек в каждом кластере
    int *cnt = calloc(k, sizeof(int)); // Выделение и обнуление массива счётчиков точек в каждом кластере
    double *sum = calloc(k * f, sizeof(double));// Выделение и обнуление массива для сумм координат по кластерам
    
     for (int i = 0; i < n; i++) {  // Проход по всем точкам данных
        int c = lbl[i];  // Кластер, которому принадлежит i-я точка
        cnt[c]++;  // Увеличение счётчика точек в кластере 
        for (int j = 0; j < f; j++) // Цикл по признакам 
            sum[c * f + j] += X[i * f + j]; // Добавление j-й координаты i-й точки к сумме по кластеру 
    }
    
    for (int c = 0; c < k; c++) { // Для каждого кластера
        if (cnt[c] > 0) {               // Если кластер не пуст 
            for (int j = 0; j < f; j++)  // Цикл по координатам
                centroids[c * f + j] = sum[c * f + j] / cnt[c]; // Новый центроид = сумма / количество точек
        }
        // Если кластер пуст, центроид не изменяется 
    }
    free(cnt); free(sum);// Освобождение 
}
// Вычисление WCSS
static double calc_wcss(const double *X, int n, int f, const int *lbl, const double *c, int k) { // Сумма квадратов расстояний от точек до своих центроидов
    (void)k;
    double wcss = 0.0; // Инициализация суммы нулём
    for (int i = 0; i < n; i++) { // Проход по всем точкам
        int cl = lbl[i]; // Индекс кластера точки i
        double d = dist(&X[i * f], &c[cl * f], f); // Евклидово расстояние от точки до её центроида
        wcss += d * d; // Добавление квадрата расстояния к общей сумме
    }
    return wcss;// Возврат 
}

//Публичные функции 
void log1p_transform(double *X, int n, int f) { // Применяет преобразование log(1 + x) ко всем элементам матрицы X
    for (int i = 0; i < n * f; i++) // Цикл по всем элементам (всего n*f чисел)
        X[i] = log(X[i] + 1.0); // Замена каждого значения на натуральный логарифм от (значение + 1)
}

void standard_scaler(double *X, int n, int f, double *out_means, double *out_stds) { // Стандартизация: (x - mean)/std для каждого признака
    for (int j = 0; j < f; j++) {// Цикл по признакам 
        double mean = 0; // Начальное значение среднего
        for (int i = 0; i < n; i++) mean += X[i * f + j]; // Суммирование всех значений j-го признака
        mean /= n;// Вычисление среднего арифметического
        out_means[j] = mean; // Сохранение среднего в выходной массив 
        
        double var = 0;// Начальное значение дисперсии
        for (int i = 0; i < n; i++) { // Цикл для вычисления суммы квадратов отклонений
            double d = X[i * f + j] - mean; // Отклонение от среднего
            var += d * d;// Накопление квадрата отклонения
        }
        double std = sqrt(var / n); // Стандартное отклонение (корень из дисперсии)
        out_stds[j] = (std < 1e-9) ? 1.0 : std; // Если std почти ноль, заменяем на 1 (чтобы не делить на ноль)
        
        for (int i = 0; i < n; i++) // Цикл по строкам
            X[i * f + j] = (X[i * f + j] - mean) / out_stds[j]; // Стандартизация значения
    }
}

int kmeans_predict(const double *point, int f, const double *centroids, int k) { // Предсказание кластера для одной точки 
    return assign(point, f, centroids, k); // Вызывает внутреннюю функцию
}

double kmeans_fit(const double *X, int *labels, double *centroids, const KMeansConfig *cfg) { // Основная функция обучения K-Means
    if (!X || !labels || !centroids || !cfg) return -1.0; // Проверка корректности входных указателей
    
    double *old_c = malloc(cfg->k * cfg->n_features * sizeof(double)); // Выделение памяти под старые центроиды для проверки сходимости
    int *tmp_lbl = malloc(cfg->n_samples * sizeof(int));// Временный массив меток для итераций
    if (!old_c || !tmp_lbl) { free(old_c); free(tmp_lbl); return -1.0; } // Если выделение не удалось, освободить частично выделенное и вернуть ошибку
    
    init_centroids(X, cfg->n_samples, cfg->n_features, cfg->k, centroids, 42); // Инициализация центроидов случайными точками 
    
    for (int it = 0; it < cfg->max_iters; it++) { // Основной цикл алгоритма до max_iters итераций
        for (int i = 0; i < cfg->n_samples; i++) // Для каждой точки
            tmp_lbl[i] = assign(&X[i * cfg->n_features], cfg->n_features, centroids, cfg->k); // Определить ближайший центроид (временные метки)
        
        memcpy(old_c, centroids, cfg->k * cfg->n_features * sizeof(double)); // Сохранить текущие центроиды для сравнения
        
        update(X, cfg->n_samples, cfg->n_features, tmp_lbl, cfg->k, centroids); // Пересчитать центроиды на основе временных меток
        
        double max_shift = 0;// Наибольшее смещение среди всех центроидов
        for (int c = 0; c < cfg->k; c++) { // Перебор всех центроидов
            double sh = dist(&old_c[c * cfg->n_features], &centroids[c * cfg->n_features], cfg->n_features); // Евклидово расстояние между старым и новым центроидом
            if (sh > max_shift) max_shift = sh; // Обновить максимальное смещение
        }
        if (max_shift < cfg->tol) break;  // Если максимальное смещение меньше допуска, алгоритм сошёлся — выходим
    }
    
    for (int i = 0; i < cfg->n_samples; i++) // После завершения итераций (или сходимости) финально присваиваем метки
        labels[i] = assign(&X[i * cfg->n_features], cfg->n_features, centroids, cfg->k); // Запись итоговых меток в выходной массив
    
    double wcss = calc_wcss(X, cfg->n_samples, cfg->n_features, labels, centroids, cfg->k); // Вычисление WCSS для полученного разбиения
    free(old_c); free(tmp_lbl); // Освобождение 
    return wcss;// Возврат значения WCSS 
}