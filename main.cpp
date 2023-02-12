#include <iostream>
#include <stdio.h>
#include <utility>
#include <fstream>
#include <string>
#include <random>
#include <iomanip>
#include <omp.h>
const int S_IN_MS = 1000; // Seconds in milliseconds
const unsigned long long MAX_SQUARED = (unsigned long long)(std::mt19937::max() >> 1) * (std::mt19937::max() >> 1);

std::pair<double,double> notParallel(double r, int n) {
    double timeStart = omp_get_wtime();
    int inCircle = 0;
    std::random_device randomDevice;
    std::mt19937 gen(randomDevice());

    for (int i = 0; i < n; ++i) {
        unsigned long long x = gen() >> 1;
        unsigned long long y = gen() >> 1;
        if ((x * x + y * y) <= MAX_SQUARED) {
            ++inCircle;
        }
    }
    double timeEnd = omp_get_wtime();
    printf("Time (%i thread(s)): %g ms\n", 1, (timeEnd - timeStart) * S_IN_MS);
    return {(double)inCircle / n * 4 * r * r, (timeEnd - timeStart) * S_IN_MS};
}

std::pair<double,double> parallel(double r, int n, int threadsNum) {
    double timeStart = omp_get_wtime();
    int inCircle = 0;
    std::random_device randomDevice;

#pragma omp parallel default(none) num_threads(threadsNum) shared(inCircle, n, MAX_SQUARED, randomDevice)
    {
        std::mt19937 genTemp(randomDevice());
        int tempCount = 0;
#pragma omp for
        for (int i = 0; i < n; ++i) {
            unsigned long long x = genTemp() >> 1;
            unsigned long long y = genTemp() >> 1;
            if ((x * x + y * y) <= MAX_SQUARED) {
                ++tempCount;
            }
        }
#pragma omp atomic
        inCircle += tempCount;
    }
    double timeEnd = omp_get_wtime();
    printf("Time (%i thread(s)): %g ms\n", threadsNum, (timeEnd - timeStart) * S_IN_MS);
    return {(double)inCircle / n * 4 * r * r, (timeEnd - timeStart) * S_IN_MS};
}

std::pair<double,double> parallel(double r, int n) {
    double timeStart = omp_get_wtime();
    int inCircle = 0;
    std::random_device randomDevice;
    int threadsNum = 1;

#pragma omp parallel default(none) shared(inCircle, n, MAX_SQUARED, randomDevice, threadsNum)
    {
        threadsNum = omp_get_num_threads();
        std::mt19937 genTemp(randomDevice());
        int tempCount = 0;
#pragma omp for
        for (int i = 0; i < n; ++i) {
            unsigned long long x = genTemp() >> 1;
            unsigned long long y = genTemp() >> 1;
            if ((x * x + y * y) <= MAX_SQUARED) {
                ++tempCount;
            }
        }
#pragma omp atomic
        inCircle += tempCount;
    }
    double timeEnd = omp_get_wtime();
    printf("Time (%i thread(s)): %g ms\n", threadsNum, (timeEnd - timeStart) * S_IN_MS);
    return {(double)inCircle / n * 4 * r * r, (timeEnd - timeStart) * S_IN_MS};
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Number of arguments should be 4\n");
        return 0;
    }
    int threadsNum;
    try {
        threadsNum = std::stoi(argv[1]);
    } catch (std::invalid_argument &e) {
        printf("Number of threads should be integer\n");
        return 0;
    } catch (std::out_of_range &e) {
        printf("Too big number of threads\n");
        return 0;
    }
    if (threadsNum < -1) {
        printf("Number of threads should be not less than -1\n");
        return 0;
    }
    double r;
    int n;

    try {
        std::ifstream in;
        in.open(argv[2], std::ios::in);
        in >> r >> n;
        in.close();
    } catch (std::ifstream::failure &e) {
        printf("Failure when reading a file\n");
        return 0;
    }

    std::pair<double, double> answer;
    if (threadsNum == -1) {
        answer = notParallel(r, n);
    } else if (threadsNum == 0) {
        answer = parallel(r, n);
    } else {
        answer = parallel(r, n, threadsNum);
    }
    try {
        std::ofstream out;
        out.open(argv[3], std::ios::out);
        out << std::fixed << std::setprecision(12) << answer.first;
        out.close();
    } catch (std::ofstream::failure &e) {
        printf("Failure when writing in a file");
        return 0;
    }
    return 0;
}
