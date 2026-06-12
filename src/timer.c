#include <stdio.h>
#include <time.h>

static struct timespec start_time, end_time;

void start_timer() {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
}

void stop_timer(int section_id) {
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("[Profiler] Section %d Execution Time: %.6f seconds\n", section_id, elapsed);
}
