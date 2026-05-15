/* ==========================================================================
 * fir_semi_parallel.c
 * Semi-parallel FIR filter -- coefficient decomposition across MPI ranks.
 *
 * Each rank owns a contiguous slice of the N filter coefficients.
 * Each rank computes partial dot products for every output sample,
 * accumulates them into a local array, then a single MPI_Reduce
 * sums partial results across all ranks.
 *
 * Analogue to: M-multiplier semi-parallel FIR (Xilinx Ch. 6)
 *   where P (MPI ranks) plays the role of M (hardware multipliers).
 *   Throughput: (P * f_clk) / N   <->   wall time per sample ~ 1/P
 *
 * Usage:  mpirun -n <P> ./fir_semi_parallel <N_taps> <L_samples>
 *
 * CPSC-375 -- High-Performance Computing, Spring 2026
 * Liu Restrepo Sanabria
 * ========================================================================== */

#include "fir_common.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <N_taps> <L_samples>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int  N = atoi(argv[1]);
    long L = atol(argv[2]);

    /* ------------------------------------------------------------------ */
    /* Coefficient slice owned by this rank                                */
    /* ------------------------------------------------------------------ */
    int base  = N / size;
    int extra = N % size;
    int coef_start = rank * base + (rank < extra ? rank : extra);
    int coef_count = base + (rank < extra ? 1 : 0);
    int coef_end   = coef_start + coef_count;   /* exclusive */

    /* ------------------------------------------------------------------ */
    /* Allocate                                                            */
    /* ------------------------------------------------------------------ */
    double *h        = (double *)malloc(N * sizeof(double));
    double *x        = (double *)malloc(L * sizeof(double));
    double *partials = (double *)calloc(L, sizeof(double));
    double *y        = NULL;
    if (rank == 0)
        y = (double *)calloc(L, sizeof(double));

    if (!h || !x || !partials || (rank == 0 && !y)) {
        fprintf(stderr, "[rank %d] Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ------------------------------------------------------------------ */
    /* Initialise (same seed on every rank -- each owns full arrays)       */
    /* ------------------------------------------------------------------ */
    generate_fir_coefficients(h, N, 0.25);
    generate_signal(x, L, 42);

    /* ------------------------------------------------------------------ */
    /* Timed computation                                                   */
    /* ------------------------------------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /*
     * Each rank computes partial inner products over its coefficient slice.
     * partials[n] = sum_{k=coef_start}^{coef_end-1} h[k] * x[n-k]
     *
     * All partials are then reduced (summed) at rank 0 to form y[n].
     * Batching: a single MPI_Reduce covers all L samples -- O(L) message,
     * independent of N.  This avoids calling Reduce once per sample.
     */
    for (long n = N - 1; n < L; n++) {
        double acc = 0.0;
        for (int k = coef_start; k < coef_end; k++)
            acc += h[k] * x[n - k];
        partials[n] = acc;
    }

    MPI_Reduce(partials, y, (int)L, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* ------------------------------------------------------------------ */
    /* Verification and report (rank 0 only)                              */
    /* ------------------------------------------------------------------ */
    if (rank == 0) {
        /* Load serial reference if available */
        char fname[64];
        snprintf(fname, sizeof(fname), "ref_N%d_L%ld.bin", N, L);
        FILE *f = fopen(fname, "rb");
        if (f) {
            double *ref = (double *)malloc(L * sizeof(double));
            fread(ref, sizeof(double), L, f);
            fclose(f);
            double err = verify(ref, y, L, N);
            fprintf(stderr, "[semi-parallel] max_abs_error = %.3e\n", err);
            free(ref);
        }
        printf("%d %ld %d %.6f\n", N, L, size, t_end - t_start);
    }

    free(h); free(x); free(partials);
    if (rank == 0) free(y);
    MPI_Finalize();
    return EXIT_SUCCESS;
}