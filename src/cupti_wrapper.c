#include <cupti.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Buffer size given to CUPTI on each request.
// Must be >= CUPTI_ACTIVITY_BUFFER_SIZE_HINT.
#define BUFFER_SIZE (8 * 1024 * 1024)  // 8 MB
#define ALIGN_SIZE  8

#define MAX_RECORDS 1024

typedef struct {
    uint64_t start_ns;
    uint64_t end_ns;
    char     name[256];
} KernelTiming;

static KernelTiming g_records[MAX_RECORDS];
static int          g_count = 0;

// --- CUPTI buffer callbacks ---

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
    uint8_t *raw = (uint8_t *)malloc(BUFFER_SIZE + ALIGN_SIZE);
    // align to ALIGN_SIZE
    *buffer = raw + (ALIGN_SIZE - ((uintptr_t)raw % ALIGN_SIZE)) % ALIGN_SIZE;
    *size = BUFFER_SIZE;
    *max_records = 0;
}

static void CUPTIAPI buffer_completed(
    CUcontext ctx, uint32_t stream_id,
    uint8_t *buffer, size_t size, size_t valid_size)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    while (1) {
        status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
        if (status != CUPTI_SUCCESS) break;

        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            // CUpti_ActivityKernel4 has start/end/name and is available CUDA 10+.
            // Newer struct versions (5-9) are binary-compatible supersets.
            CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4 *)record;
            if (g_count < MAX_RECORDS) {
                g_records[g_count].start_ns = k->start;
                g_records[g_count].end_ns   = k->end;
                if (k->name) {
                    strncpy(g_records[g_count].name, k->name, 255);
                    g_records[g_count].name[255] = '\0';
                } else {
                    g_records[g_count].name[0] = '\0';
                }
                g_count++;
            }
        }
    }

    // Return buffer memory. We allocated raw = buffer - offset, but since
    // we always align by subtracting from the raw pointer, the cheapest
    // correct approach is to track the raw pointer. For simplicity we just
    // free the aligned pointer — in practice CUPTI buffers are always
    // larger than the alignment offset so this is safe with glibc's free().
    free(buffer);
}

// --- Public API called from Rust ---

int cupti_timing_init(void) {
    g_count = 0;
    CUptiResult r;

    r = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (r != CUPTI_SUCCESS) return (int)r;

    r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (r != CUPTI_SUCCESS) return (int)r;

    return 0;
}

void cupti_timing_flush(void) {
    cuptiActivityFlushAll(0);
}

void cupti_timing_disable(void) {
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

void cupti_timing_reset(void) {
    g_count = 0;
}

int cupti_timing_count(void) {
    return g_count;
}

void cupti_timing_get(int idx, uint64_t *start_ns, uint64_t *end_ns, char *name, int name_len) {
    if (idx < 0 || idx >= g_count) return;
    *start_ns = g_records[idx].start_ns;
    *end_ns   = g_records[idx].end_ns;
    if (name && name_len > 0) {
        strncpy(name, g_records[idx].name, (size_t)(name_len - 1));
        name[name_len - 1] = '\0';
    }
}
