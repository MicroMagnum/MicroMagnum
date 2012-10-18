%{
#include "Benchmark.h"

static void getBenchmarkRecord(const std::string &id, std::string *desc, int *num_benchmarks, double *total_time, double *avg_time)
{
        Benchmark::Record &rec = Benchmark::inst().getRecord(id);
        *num_benchmarks = rec.num_benchmarks;
        *total_time = rec.total_time;
        *avg_time = rec.total_time / rec.num_benchmarks;
        *desc = rec.desc;
}

static void resetBenchmark()
{
        Benchmark::inst().reset();
}

%}

void getBenchmarkRecord(const std::string &id, std::string *OUTPUT, int *OUTPUT, double *OUTPUT, double *OUTPUT);
void resetBenchmark();
