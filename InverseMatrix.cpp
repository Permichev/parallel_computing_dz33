/******************************************************************************
 * InverseMatrix.cpp
 * 
 * Пример кода для обращения плотной матрицы N×N с разными вариантами оптимизации:
 *  - Базовый вариант (неоптимизированный)
 *  - Оптимизированный с помощью ключей компилятора
 *  - Полуавтоматическая векторизация (директивы)
 *  - OpenMP (векторизация + параллелизация)
 *  - MPI (распараллеливание между процессами)
 *
 * Компиляция (пример для Apple Clang):
 *   1) БЕЗ оптимизаций:        clang++ -o inverse_no_opt InverseMatrix.cpp
 *   2) С авто-векторизацией:   clang++ -o inverse_opt InverseMatrix.cpp -O3 -fopenmp \
 *                              -mcpu=apple-m3 -march=armv8.5-a
 *   3) C полуавтоматикой:      clang++ -o inverse_vect InverseMatrix.cpp -O3 -fopenmp \
 *                              -DUSE_VECTOR_DIRECTIVES
 *   4) С OpenMP:               clang++ -o inverse_omp InverseMatrix.cpp -O3 -fopenmp \
 *                              -DUSE_OPENMP
 *   5) С MPI:                  mpicxx -o inverse_mpi InverseMatrix.cpp -O3 -fopenmp \
 *                              -DUSE_MPI
 *
 *   (При условии, что вы установили OpenMP и MPI для macOS.)
 *
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cstdlib>       // для std::rand
#include <cmath>         // для std::fabs
#include <chrono>        // для замеров времени
#include <iomanip>       // для форматирования вывода
#include <memory>        // для std::unique_ptr и aligned_alloc (C++17/20)
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

// Макрос для директив векторизации (полуавтоматической)
// Если не задан USE_VECTOR_DIRECTIVES, эти макросы будут пустыми.
#ifdef USE_VECTOR_DIRECTIVES
  #define VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#else
  #define VECTORIZE_LOOP
#endif

// Функция для проверки результата: A * A_inv ~ I
double checkInverse(const double* A, const double* Ainv, int N)
{
    // Вычислим R = A * Ainv. Ожидается R ~ I.
    // Дальше смотрим макс. отклонение R[i][i] от 1 и R[i][j] от 0.

    double maxError = 0.0;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < N; k++)
            {
                sum += A[i*N + k] * Ainv[k*N + j];
            }
            // Проверяем отклонение от единичной матрицы
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = std::fabs(sum - expected);
            if(diff > maxError) {
                maxError = diff;
            }
        }
    }
    return maxError;
}

/**
 * Обращение матрицы методом Гаусса-Жордана (с частичным выбором главного элемента).
 * matrixInOut и matrixInv должны быть размером N*N (в элементарном виде — одномерные массивы).
 *
 * Реализация в стиле "в лоб": может быть не самой быстрой,
 * но хорошо иллюстрирует идею и поддаётся векторизации и распараллеливанию.
 */
void invertMatrixGaussJordan(double* matrixInOut, double* matrixInv, int N)
{
    // Инициализируем matrixInv как единичную
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            matrixInv[i*N + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Фаза прямого хода
    for(int i = 0; i < N; i++)
    {
        // Ищем максимум в столбце i (частичный pivot)
        double maxVal = std::fabs(matrixInOut[i*N + i]);
        int maxRow = i;
        for(int k = i+1; k < N; k++)
        {
            double val = std::fabs(matrixInOut[k*N + i]);
            if(val > maxVal)
            {
                maxVal = val;
                maxRow = k;
            }
        }

        // Меняем местами строки i и maxRow (и в matrixInOut, и в matrixInv)
        if(maxRow != i)
        {
            for(int col = 0; col < N; col++)
            {
                std::swap(matrixInOut[i*N + col], matrixInOut[maxRow*N + col]);
                std::swap(matrixInv[i*N + col],    matrixInv[maxRow*N + col]);
            }
        }

        // Нормируем ведущий элемент до 1
        double pivot = matrixInOut[i*N + i];
        for(int col = 0; col < N; col++)
        {
            matrixInOut[i*N + col] /= pivot;
            matrixInv[i*N + col]   /= pivot;
        }

        // Обнуляем элементы ниже ведущего
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for(int row = i+1; row < N; row++)
        {
            double factor = matrixInOut[row*N + i];
            for(int col = 0; col < N; col++)
            {
                matrixInOut[row*N + col] -= factor * matrixInOut[i*N + col];
                matrixInv[row*N + col]   -= factor * matrixInv[i*N + col];
            }
        }
    }

    // Фаза обратного хода
    for(int i = N-1; i >= 0; i--)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for(int row = i-1; row >= 0; row--)
        {
            double factor = matrixInOut[row*N + i];
            for(int col = 0; col < N; col++)
            {
                matrixInOut[row*N + col] -= factor * matrixInOut[i*N + col];
                matrixInv[row*N + col]   -= factor * matrixInv[i*N + col];
            }
        }
    }
}

int main(int argc, char* argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // Ввод размера матрицы пользователем
    int N = 0;
    if(argc > 1) {
        N = std::atoi(argv[1]);
    }
    if(N <= 0) {
        std::cout << "Введите размер матрицы N: ";
        std::cin >> N;
    }

    // Выделяем выровненную память под матрицы (std::aligned_alloc в C++17/20)
    // Для ARM NEON может быть достаточно выравнивания в 16 байт.
    constexpr size_t alignment = 16;
    size_t bytes = sizeof(double) * N * N;

    double* matrix = static_cast<double*>(std::aligned_alloc(alignment, bytes));
    double* matrixCopy = static_cast<double*>(std::aligned_alloc(alignment, bytes));
    double* matrixInv = static_cast<double*>(std::aligned_alloc(alignment, bytes));

    // Генерация исходной матрицы (случайные числа)
    std::srand(12345); // фиксируем seed для воспроизводимости
    for(int i = 0; i < N*N; i++)
    {
        matrix[i] = static_cast<double>(std::rand() % 100 + 1);
    }

    // Делаем копию matrix для работы алгоритма
    std::memcpy(matrixCopy, matrix, bytes);

    // Засекаем время
    auto t1 = std::chrono::steady_clock::now();

#ifndef USE_MPI
    // --- ЛОКАЛЬНЫЙ ВАРИАНТ (без MPI) ---
    // Полуавтоматическая векторизация (директива) может быть расставлена в самых "жирных" циклах.
    // Пример (здесь показываем на копировании, но полезно в "ядре" алгоритма):
    VECTORIZE_LOOP
    for(int i = 0; i < N*N; i++)
    {
        // (В настоящем коде расставляйте директивы там, где действительно идет матричная математика)
        matrixCopy[i] = matrixCopy[i]; // placeholder
    }

    // Запускаем обращение
    invertMatrixGaussJordan(matrixCopy, matrixInv, N);

#else
    // --- MPI-ВАРИАНТ ---
    // Здесь лишь СКЕЛЕТ распараллеливания, реальная реализация обращения требует
    // распределить блоки строк (или столбцов) между процессами, обмениваться ведущими
    // строками и т.д. Это довольно большой код, поэтому показываем идею:

    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Для примера: пусть каждый процесс обрабатывает chunk строк.
    int chunk = N / world_size;
    int startRow = world_rank * chunk;
    int endRow   = (world_rank == world_size - 1) ? N : (world_rank+1) * chunk;

    // Распространить данные матрицы от root (0) ко всем
    MPI_Bcast(matrix, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Локально выделить память для куска matrixCopy и matrixInv
    // (или каждый хранит полную копию — зависит от реализации)
    // ... Тут детали ...

    // Обратить матрицу коллективным алгоритмом Гаусса (псевдокод).
    // 1) Для каждой строки i:
    //    - Вычислить ведущий элемент (pivot) и распространить его по всем процессам.
    //    - Каждый процесс нормирует свою часть i-й строки (если у него есть).
    //    - Каждый процесс обнуляет подстрочные элементы (если относятся к его chunk).
    // 2) Обратный ход схожим образом.

    // Для простоты здесь покажем "локально":
    invertMatrixGaussJordan(matrixCopy, matrixInv, N);

    // Сбор результатов на root
    MPI_Gather(/*...*/, /*...*/, MPI_DOUBLE,
               /*...*/, /*...*/, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

#endif

    // Останавливаем таймер
    auto t2 = std::chrono::steady_clock::now();
    double elapsedSec = std::chrono::duration<double>(t2 - t1).count();

    // Проверяем корректность результата
    double maxErr = checkInverse(matrix, matrixInv, N);

    // Вывод результатов
#ifdef USE_MPI
    if(!world_rank)
    {
        std::cout << "[MPI variant] N=" << N
                  << "  time=" << elapsedSec << "s"
                  << "  maxErr=" << maxErr << std::endl;
    }
    MPI_Finalize();
#else
    std::cout << "[Local variant] N=" << N
              << "  time=" << elapsedSec << "s"
              << "  maxErr=" << maxErr << std::endl;
#endif

    // Освобождение памяти
    std::free(matrix);
    std::free(matrixCopy);
    std::free(matrixInv);

    return 0;
}
