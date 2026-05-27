#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmeans.h"
#include "anomalies.h"
//Генерация данных с пропорциями 
void generate_rfm_data(double *data, int n_customers) {
    for (int i = 0; i < n_customers; i++) {
        int group;
        if (i < 15) group = 0;  // VIP 
        else if (i < 65) group = 1;  // Постоянные 
        else group = 2;  // Спящие 
        
        double r, f, m, v;            // Переменные для Recency, Frequency, Monetary, Variety
        if (group == 0) {             // Если VIP
            r = 5 + (rand() % 15);    // Recency: (5-19 дней)
            f = 40 + (rand() % 20);   // Frequency: (40-59 заказов)
            m = 2000 + (rand() % 500);// Monetary: (2000-2499)
            v = 30 + (rand() % 10);   // Variety: (30-39 товаров)
        } else if (group == 1) {      // Если постоянные
            r = 30 + (rand() % 30);   // Recency: (30-59 дней)
            f = 15 + (rand() % 15);   // Frequency: (15-29 заказов)
            m = 300 + (rand() % 150); // Monetary: (300-449)
            v = 10 + (rand() % 8);    // Variety: (10-17 товаров)
        } else {                      // Если спящие
            r = 150 + (rand() % 150); // Recency: (150-299 дней)
            f = 1 + (rand() % 2);     // Frequency: (1-2 заказа)
            m = 15 + (rand() % 30);   // Monetary: (15-44)
            v = 1 + (rand() % 3);     // Variety: (1-3 товара)
        }
        data[i*N_FEATURES + RECENCY]   = r;// Запись Recency в соответствующую позицию
        data[i*N_FEATURES + FREQUENCY] = f;// Запись Frequency
        data[i*N_FEATURES + MONETARY]  = m;// Запись Monetary
        data[i*N_FEATURES + VARIETY]   = v;// Запись Variety
        data[i*N_FEATURES + AVG_ORDER] = (f > 0.0) ? m / f : 0.0; // Средний чек = Monetary / Frequency (если частота > 0)
    }
}


// Метод локтя 
void print_elbow(const double *wcss, int k_min, int k_max) {  // Печатает график WCSS для диапазона k от k_min до k_max
    printf("\n Метод локтя (WCSS):\n"); // Заголовок
    printf("   K\tWCSS\t\tГрафик\n"); // Шапка таблицы
    printf("   --------------------------------\n"); // Разделитель
    double max_w = wcss[0], min_w = wcss[k_max - k_min];// Инициализация max и min первым и последним значением WCSS
    for (int k = k_min; k <= k_max; k++) { // Цикл по всем k
        double val = wcss[k - k_min];  // Текущее значение WCSS
        int bar_len = (int)((val - min_w) / (max_w - min_w + 1e-9) * 30); // Длина полоски графика 
        printf("   %d\t%.1f\t|", k, val); // Вывод k и значения
        for (int i = 0; i < bar_len; i++) printf("-"); // Рисование полоски из дефисов
        printf("\n");// Перевод строки
    }
    
}

//Таблица статистики 
void print_segment_stats(const double *raw_data, const int *labels, int n) {//Вывод средних показателей по сегментам
    printf("\n СРЕДНИЕ ПОКАЗАТЕЛИ ПО СЕГМЕНТАМ:\n"); //Заголовок 
    printf("   ┌──────────┬──────────┬──────────┬─────────┬─────────┐\n");//Верхняя граница
    printf("   │ Сегмент  │ Клиентов │ Давность │ Частота │ Выручка │\n");//Шапка
    printf("   ├──────────┼──────────┼──────────┼─────────┼─────────┤\n");//Границы
    
    for (int c = 0; c < 3; c++) { // Цикл по трём сегментам кластерам
        int cnt = 0; // Счётчик клиентов в сегменте c
        double sum_r = 0, sum_f = 0, sum_m = 0;// Суммы Recency, Frequency, Monetary
        for (int i = 0; i < n; i++) {// Цикл по всем клиентам
            if (labels[i] == c) {// Если метка клиента совпадает с текущим сегментом
                cnt++; // Увеличить счётчик
                sum_r += raw_data[i*N_FEATURES + RECENCY];// Добавить Recency
                sum_f += raw_data[i*N_FEATURES + FREQUENCY];// Добавить Frequency
                sum_m += raw_data[i*N_FEATURES + MONETARY];// Добавить Monetary
            }
        }
        if (cnt > 0) {// Если сегмент не пуст
            //Выравнивание
            printf("   │   #%d     │   %4d   │%6.0f дн │  %5.1f  │ £%6.0f │\n", c + 1, cnt, sum_r/cnt, sum_f/cnt, sum_m/cnt);// Нумерация с 1
        }
    }
    printf("   └──────────┴──────────┴──────────┴─────────┴─────────┘\n");// Нижняя граница
    printf("   Сегменты: #1=Постоянные, #2=VIP, #3=Спящие клиенты\n\n");// Легенда
}

//Интерактивный режим
void interactive_mode(const double *means, const double *stds, const double *centroids, int k) { // Функция интерактивного предсказания для нового клиента
    printf("РЕЖИМ ПРОВЕРКИ КЛИЕНТА (для выхода введите букву)\n");// Приветствие 
    double input[N_FEATURES];// Массив для ввода признаков клиента
    
    while (1) { // Цикл бесконечный 
        printf("\n   [Новый клиент]\n"); // Приглашение
        printf("   1. Давность (дни): ");      if (scanf("%lf", &input[RECENCY]) != 1) break;   // Ввод Recency, при ошибке — выход
        printf("   2. Частота заказов: ");     if (scanf("%lf", &input[FREQUENCY]) != 1) break; // Ввод Frequency
        printf("   3. Выручка (GBP): ");       if (scanf("%lf", &input[MONETARY]) != 1) break;  // Ввод Monetary
        printf("   4. Разных товаров: ");      if (scanf("%lf", &input[VARIETY]) != 1) break;   // Ввод Variety
        
         input[AVG_ORDER] = (input[FREQUENCY] > 0.0) ? input[MONETARY] / input[FREQUENCY] : 0.0;// Вычисление среднего чека
        printf("   → Средний чек: £%.2f\n", input[AVG_ORDER]);// Вывод среднего чека
        
        // Предобработка
        double proc[N_FEATURES];// Массив для преобразованного вектора
        for (int f = 0; f < N_FEATURES; f++) { // Цикл по всем признакам
            double log_val = log(input[f] + 1.0);// Логарифмическое преобразование 
            proc[f] = (log_val - means[f]) / stds[f];// Вычисление среднего
        }
        
        // Предсказание
        int seg = kmeans_predict(proc, N_FEATURES, centroids, k); // Определение близжайшего центроида
        int seg_display = seg + 1;  // Конвертация
        
        // Вывод результата
        printf("\n   Клиент отнесён к СЕГМЕНТУ #%d\n", seg_display);// номер сегмента
        if (seg == 1) {  // VIP
            printf("   Профиль: VIP-клиент (высокая лояльность, макс. прибыль)\n");
        } else if (seg == 0) {  // Постоянные
            printf("   Профиль: Постоянный эконом-класс (частые покупки, средний чек)\n");
        } else {  // Спящие
            printf("   Профиль: Спящий клиент (давно не покупал, нужна реактивация)\n");
        }
        printf("   ─────────────────────────────────────\n");
    }
    printf("\n   Завершение работы.\n"); // Сообщение о выходе
}

int main() {
    #ifdef _WIN32
    system("chcp 65001 > nul");
    #endif
    
    srand(42);// Инициализация генератора случайных чисел фиксированным seed
    int n_customers = 150, k = 3; // Количество клиентов и число кластеров
    
    printf("K-means кластеризация клиентов (RFM-анализ)\n");// Заголовок программы
    printf("   Клиентов: %d | Признаков: %d | Сегментов: %d (нумерация с #1)\n\n", 
           n_customers, N_FEATURES, k);// Вывод параметров
    
    // Выделение памяти
    double *raw_data = malloc(n_customers * N_FEATURES * sizeof(double)); // Необработанные данные
    double *proc_data = malloc(n_customers * N_FEATURES * sizeof(double)); // Данные после предобработки
    int *labels = malloc(n_customers * sizeof(int));// Массив меток кластеров
    double *centroids = malloc(k * N_FEATURES * sizeof(double)); // Массив центроидов
    double *means = calloc(N_FEATURES, sizeof(double)); // Средние для каждого признака (инициализированы 0)
    double *stds = calloc(N_FEATURES, sizeof(double));// Стандартные отклонения для каждого признака (инициализированы 0)
    
    // Генерация и предобработка
    generate_rfm_data(raw_data, n_customers); // Заполнение данными
    memcpy(proc_data, raw_data, n_customers * N_FEATURES * sizeof(double));  // Копирование 
    log1p_transform(proc_data, n_customers, N_FEATURES); // Применение преобразования 
    standard_scaler(proc_data, n_customers, N_FEATURES, means, stds); // Стандартизация 

    // Метод локтя
    double wcss[5];  // Массив для хранения WCSS для k=2-6
    printf("Вычисление метода локтя...\n"); 
    for (int test_k = 2; test_k <= 6; test_k++) { // Цикл по k от 2 до 6
        double *tmp_c = malloc(test_k * N_FEATURES * sizeof(double));// Временные центроиды
        int *tmp_l = malloc(n_customers * sizeof(int));// Временные метки
        KMeansConfig cfg = {n_customers, N_FEATURES, test_k, 100, 1e-4};// Конфигурация для текущего k
        wcss[test_k - 2] = kmeans_fit(proc_data, tmp_l, tmp_c, &cfg);// Обучение и сохранение WCSS
        free(tmp_c); free(tmp_l);// Освобождение временной памяти
    }
    print_elbow(wcss, 2, 6);// Вывод графика метода локтя
    
    // Финальное обучение
    printf("Обучение модели с K=%d...\n", k);
    KMeansConfig cfg = {n_customers, N_FEATURES, k, 100, 1e-4}; // Конфигурация для финальной модели
    double final_wcss = kmeans_fit(proc_data, labels, centroids, &cfg); // Обучение модели, сохранение меток и центроидов
    printf("Готово! Инерция (WCSS): %.2f\n", final_wcss);// Вывод финальной инерции
     
    int *anom_flags = calloc(n_customers, sizeof(int));
    double threshold = 4.5;  // Порог для стандартизированных данных

    int anom_count = detect_anomalies(proc_data, n_customers, N_FEATURES,centroids, k, threshold, anom_flags);
    printf("Найдено аномалий: %d\n", anom_count);

    free(anom_flags);
    
    // Вывод статистики 
    print_segment_stats(raw_data, labels, n_customers); // Печать средних показателей по сегментам
    
    // Интерактивный режим
    interactive_mode(means, stds, centroids, k); // Запуск интерактивного опроса
    
    // Очистка
    free(raw_data); free(proc_data); free(labels);// Освобождение памяти
    free(centroids); free(means); free(stds);// Освобождение памяти
    
    return 0; // Успешное завершение программы
}

