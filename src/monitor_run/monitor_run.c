// Copyright (c) 2011-2026 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
//
// monitor_run.c - Wrap a subprocess (e.g. nvdla_runtime) and dump ESP monitor
// counters across the wrapped run.
//
// Usage:
//   monitor_run.exe <label> -- <cmd> [args...]
//   monitor_run.exe <label> -- "<sh -c command string>"
//
// Output (stdout, CSV; sweep script greps "^# " out before joining).
// One row per accelerator tile (SOC_NACC rows total). Per-acc counters apply
// only to that tile; SoC-wide counters (DDR, LLC, mem-tile coherence) are
// broadcast to every row of the same run.
//
// Columns:
//   label              user-supplied tag
//   acc_index          0..SOC_NACC-1
//   acc_invocations    accelerator "done" pulses (MON_ACC_INVOCATIONS)
//   acc_tot_cycles     non-idle cycles in tile FSM (MON_ACC_TOT_LO/HI)
//   acc_mem_cycles     subset stalled on memory (MON_ACC_MEM_LO/HI)
//   acc_tlb_cycles     TLB-loading cycles (MON_ACC_TLB_INDEX)
//   noc_inject_p0..p5  NoC packets injected by this acc tile, per NoC plane
//                      (planes are SoC-specific; typically:
//                       0=coh-req, 1=coh-fwd, 2=coh-rsp,
//                       3=DMA-req,  4=DMA-rsp/IO,  5=peripheral)
//   ddr_words          off-chip DRAM word transfers (sum over SOC_NMEM)
//   mem_coh_reqs       coherence requests arriving at memory tile
//   mem_coh_fwds       coherence forwards from LLC
//   mem_coh_rsps_rcv   coherence responses received by LLC
//   mem_coh_rsps_snd   coherence responses sent by LLC
//   mem_dma_reqs       non-coherent DMA requests at memory tile
//   mem_dma_rsps       non-coherent DMA responses at memory tile
//   mem_coh_dma_reqs   coherent DMA requests at LLC
//   mem_coh_dma_rsps   coherent DMA responses from LLC
//   llc_hits           LLC hits (sum over SOC_NMEM)
//   llc_misses         LLC misses (sum over SOC_NMEM)
//
// Also writes a full esp_monitor_print() dump to /tmp/<label>.mon.txt with
// every monitor field (DVFS, NoC queue-full, L2 stats per tile, etc.).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdint.h>

#include "monitors.h"
// soc_locs.h is included inside main(); same convention as libmonitors.c.

static unsigned long long lo_hi(unsigned int lo, unsigned int hi)
{
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}

int main(int argc, char **argv)
{
#include "soc_locs.h"

    int sep = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { sep = i; break; }
    }
    if (sep < 2 || sep + 1 >= argc) {
        fprintf(stderr,
            "usage: %s <label> -- <cmd> [args...]\n"
            "       %s <label> -- '<sh -c command string>'\n",
            argv[0], argv[0]);
        return 2;
    }

    const char *label = argv[1];
    int n_tail = argc - sep - 1;
    char **tail = &argv[sep + 1];

    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_ALL;

    esp_monitor_vals_t *vs = esp_monitor_vals_alloc();
    esp_monitor_vals_t *ve = esp_monitor_vals_alloc();
    if (!vs || !ve) {
        fprintf(stderr, "esp_monitor_vals_alloc failed\n");
        return 1;
    }

    esp_monitor(mon_args, vs);

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        if (n_tail == 1) {
            char *sh_argv[] = { (char *)"sh", (char *)"-c", tail[0], NULL };
            execvp(sh_argv[0], sh_argv);
        } else {
            execvp(tail[0], tail);
        }
        perror("execvp");
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }

    esp_monitor(mon_args, ve);
    esp_monitor_vals_t d = esp_monitor_diff(*vs, *ve);

    // SoC-wide memory-tile aggregates (single mem tile in this SoC, but loop
    // for portability)
    unsigned long long ddr_total       = 0;
    unsigned long long mem_coh_reqs    = 0;
    unsigned long long mem_coh_fwds    = 0;
    unsigned long long mem_coh_rsps_rcv = 0;
    unsigned long long mem_coh_rsps_snd = 0;
    unsigned long long mem_dma_reqs    = 0;
    unsigned long long mem_dma_rsps    = 0;
    unsigned long long mem_coh_dma_reqs = 0;
    unsigned long long mem_coh_dma_rsps = 0;
    unsigned long long llc_hits        = 0;
    unsigned long long llc_misses      = 0;

    for (int m = 0; m < SOC_NMEM; m++) {
        ddr_total        += (unsigned long long)d.ddr_accesses[m];
        mem_coh_reqs     += (unsigned long long)d.mem_reqs[m].coh_reqs;
        mem_coh_fwds     += (unsigned long long)d.mem_reqs[m].coh_fwds;
        mem_coh_rsps_rcv += (unsigned long long)d.mem_reqs[m].coh_rsps_rcv;
        mem_coh_rsps_snd += (unsigned long long)d.mem_reqs[m].coh_rsps_snd;
        mem_dma_reqs     += (unsigned long long)d.mem_reqs[m].dma_reqs;
        mem_dma_rsps     += (unsigned long long)d.mem_reqs[m].dma_rsps;
        mem_coh_dma_reqs += (unsigned long long)d.mem_reqs[m].coh_dma_reqs;
        mem_coh_dma_rsps += (unsigned long long)d.mem_reqs[m].coh_dma_rsps;
        llc_hits         += (unsigned long long)d.llc_stats[m].hits;
        llc_misses       += (unsigned long long)d.llc_stats[m].misses;
    }

    printf("# label,acc_index,acc_invocations,acc_tot_cycles,acc_mem_cycles,"
           "acc_tlb_cycles,"
           "noc_inject_p0,noc_inject_p1,noc_inject_p2,"
           "noc_inject_p3,noc_inject_p4,noc_inject_p5,"
           "ddr_words,"
           "mem_coh_reqs,mem_coh_fwds,mem_coh_rsps_rcv,mem_coh_rsps_snd,"
           "mem_dma_reqs,mem_dma_rsps,mem_coh_dma_reqs,mem_coh_dma_rsps,"
           "llc_hits,llc_misses\n");

    for (int a = 0; a < SOC_NACC; a++) {
        int tile = acc_locs[a].row * SOC_COLS + acc_locs[a].col;
        esp_acc_stats_t s = d.acc_stats[a];
        unsigned long long cyc = lo_hi(s.acc_tot_lo, s.acc_tot_hi);
        unsigned long long mem = lo_hi(s.acc_mem_lo, s.acc_mem_hi);

        // Per-plane NoC injects for this acc tile. NOC_PLANES is 6 in this
        // SoC; if it's ever different we still print 6 columns and zero-pad.
        unsigned long long p[6] = {0};
        for (int pi = 0; pi < NOC_PLANES && pi < 6; pi++)
            p[pi] = (unsigned long long)d.noc_injects[tile][pi];

        printf("%s,%d,%u,%llu,%llu,%u,"
               "%llu,%llu,%llu,%llu,%llu,%llu,"
               "%llu,"
               "%llu,%llu,%llu,%llu,"
               "%llu,%llu,%llu,%llu,"
               "%llu,%llu\n",
               label, a, s.acc_invocations, cyc, mem, s.acc_tlb,
               p[0], p[1], p[2], p[3], p[4], p[5],
               ddr_total,
               mem_coh_reqs, mem_coh_fwds, mem_coh_rsps_rcv, mem_coh_rsps_snd,
               mem_dma_reqs, mem_dma_rsps, mem_coh_dma_reqs, mem_coh_dma_rsps,
               llc_hits, llc_misses);
    }

    char dump_path[512];
    snprintf(dump_path, sizeof(dump_path), "/tmp/%s.mon.txt", label);
    FILE *fp = fopen(dump_path, "w");
    if (fp) {
        esp_monitor_print(mon_args, d, fp);
        fclose(fp);
        fprintf(stderr, "[monitor_run] full dump: %s\n", dump_path);
    }

    esp_monitor_free();
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
