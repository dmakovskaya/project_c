#ifndef KMEANS_H
#define KMEANS_H

//Перечисление признаков для RFM
typedef enum {
    RECENCY = 0,// Давность (Recency): сколько дней прошло с последней покупки
    FREQUENCY,  // Частота (Frequency): как часто клиент совершает покупки
    MONETARY,   // Выручка (Monetary): общая сумма, потраченная клиентом
    VARIETY,    // Разнообразие (Variety): количество уникальных категорий или товаров
    AVG_ORDER,  // Средний чек (Average Order Value): средняя сумма одного заказа
    N_FEATURES  // Вспомогательная константа: всего 5 признаков 
} Features;

//параметры K-means
typedef struct {
    int n_samples;// количество точек данных
    int n_features;// количество признаков
    int k;// число кластеров
    int max_iters;// максимум итераций
    double tol;// порог сходимости
} KMeansConfig;

//Препроцессинг
void log1p_transform(double *X, int n_samples, int n_features);
void standard_scaler(double *X, int n_samples, int n_features, double *out_means, double *out_stds);

//K-means алгоритм 
double kmeans_fit(const double *X, int *labels, double *centroids, const KMeansConfig *cfg); //обучение
int kmeans_predict(const double *point, int n_features, const double *centroids, int k); //рассчет на основе данных клиента

#endif 