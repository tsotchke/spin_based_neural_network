/*
 * benchmarks/bench_common.h
 *
 * Minimal timing helpers + JSON emitter shared across all benchmarks.
 * Dumps results under benchmarks/results/<suite>/<name>.json for trend tracking.
 */
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/utsname.h>

static inline double bench_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline void bench_mkdir_p(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t n = strlen(tmp);
    if (n == 0) return;
    if (tmp[n - 1] == '/') tmp[n - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* Emit one result record. Call bench_emit_begin/bench_emit_metric/bench_emit_end
 * for multi-metric runs, or bench_emit_single for the common one-metric case. */
typedef struct {
    FILE *fp;
    int metric_count;
} bench_emitter_t;

static inline void bench_emit_begin(bench_emitter_t *em, const char *suite, const char *name) {
    char dir[512], path[768];
    snprintf(dir, sizeof(dir), "benchmarks/results/%s", suite);
    bench_mkdir_p(dir);
    snprintf(path, sizeof(path), "%s/%s.json", dir, name);
    em->fp = fopen(path, "w");
    em->metric_count = 0;
    if (!em->fp) { fprintf(stderr, "bench: cannot write %s\n", path); return; }

    struct utsname u;
    uname(&u);
    time_t now = time(NULL);
    fprintf(em->fp,
            "{\n"
            "  \"suite\": \"%s\",\n"
            "  \"name\": \"%s\",\n"
            "  \"utc_epoch\": %ld,\n"
            "  \"os\": \"%s\",\n"
            "  \"arch\": \"%s\",\n"
            "  \"hostname\": \"%s\",\n"
            "  \"metrics\": {\n",
            suite, name, (long)now, u.sysname, u.machine, u.nodename);
}

static inline void bench_emit_metric(bench_emitter_t *em, const char *key, double value) {
    if (!em->fp) return;
    if (em->metric_count > 0) fputs(",\n", em->fp);
    fprintf(em->fp, "    \"%s\": %.9g", key, value);
    em->metric_count++;
}

static inline void bench_emit_int(bench_emitter_t *em, const char *key, long value) {
    if (!em->fp) return;
    if (em->metric_count > 0) fputs(",\n", em->fp);
    fprintf(em->fp, "    \"%s\": %ld", key, value);
    em->metric_count++;
}

static inline void bench_emit_end(bench_emitter_t *em) {
    if (!em->fp) return;
    fputs("\n  }\n}\n", em->fp);
    fclose(em->fp);
    em->fp = NULL;
}

#endif /* BENCH_COMMON_H */
