// CUPTI Event API shim — collects warp stall counters.
//
// The Event API is deprecated in CUDA 11+ (superseded by the Perfworks /
// Profiling API) but remains functional on all CUDA 10+ toolchains and is
// far simpler to embed than the full Perfworks pipeline.
//
// Events are summed across all SMs.  On Volta+ you may need to set
//   CUDA_AUTO_BOOST=0   and run as root, or set
//   NV_GPU_ENABLE_ADMIN_REQUIRED_COUNTERS=1
// if the driver requires elevated privileges for hardware counters.

#include <cupti.h>
#include <cuda.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_EVENTS 8

// Canonical warp-stall event names (sm_52+).
static const char *STALL_NAMES[MAX_EVENTS] = {
    "stall_memory_dependency",
    "stall_exec_dependency",
    "stall_inst_fetch",
    "stall_memory_throttle",
    "stall_sync",
    "stall_texture",
    "stall_other",
    "stall_not_selected",
};

typedef struct {
    char     name[64];
    uint64_t count;
} StallRecord;

static StallRecord      g_records[MAX_EVENTS];
static int              g_count  = 0;
static CUpti_EventGroup g_group  = NULL;
static CUpti_EventID    g_eids[MAX_EVENTS];
static int              g_neids  = 0;

// Returns 0 on success, negative on setup error, positive CUPTI status on
// CUPTI failure.  Must be called after a CUDA context exists (e.g. after
// cudarc's CudaDevice::new).
int stall_counters_init(void) {
    CUcontext ctx = NULL;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL) return -1;

    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -2;

    CUptiResult r = cuptiEventGroupCreate(ctx, &g_group, 0);
    if (r != CUPTI_SUCCESS) return (int)r;

    g_neids = 0;
    for (int i = 0; i < MAX_EVENTS; i++) {
        CUpti_EventID eid;
        if (cuptiEventGetIdFromName(dev, STALL_NAMES[i], &eid) != CUPTI_SUCCESS)
            continue;
        if (cuptiEventGroupAddEvent(g_group, eid) != CUPTI_SUCCESS)
            continue;
        g_eids[g_neids] = eid;
        strncpy(g_records[g_neids].name, STALL_NAMES[i], 63);
        g_records[g_neids].name[63] = '\0';
        g_records[g_neids].count    = 0;
        g_neids++;
    }

    if (g_neids == 0) {
        cuptiEventGroupDestroy(g_group);
        g_group = NULL;
        return -3;  // no stall events available on this GPU
    }

    r = cuptiEventGroupEnable(g_group);
    if (r != CUPTI_SUCCESS) {
        cuptiEventGroupDestroy(g_group);
        g_group = NULL;
        return (int)r;
    }

    g_count = 0;
    return 0;
}

// Read accumulated stall counts into g_records.  Call after kernel +
// synchronize.  Safe to call multiple times; values accumulate until
// stall_counters_disable() is called.
void stall_counters_read(void) {
    if (!g_group) return;
    g_count = 0;
    for (int i = 0; i < g_neids; i++) {
        size_t   bytes = sizeof(uint64_t);
        uint64_t val   = 0;
        cuptiEventGroupReadEvent(
            g_group, CUPTI_EVENT_READ_FLAG_NONE,
            g_eids[i], &bytes, &val);
        g_records[i].count = val;
        g_count++;
    }
}

void stall_counters_disable(void) {
    if (g_group) {
        cuptiEventGroupDisable(g_group);
        cuptiEventGroupDestroy(g_group);
        g_group = NULL;
    }
    g_neids = 0;
    g_count = 0;
}

int stall_counters_count(void) {
    return g_count;
}

void stall_counters_get(int idx, char *name, int name_len, uint64_t *out_count) {
    if (idx < 0 || idx >= g_count) return;
    if (name && name_len > 0) {
        strncpy(name, g_records[idx].name, (size_t)(name_len - 1));
        name[name_len - 1] = '\0';
    }
    if (out_count) *out_count = g_records[idx].count;
}
