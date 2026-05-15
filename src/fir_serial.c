/* ==========================================================================
 * fir_serial.c
 * Sequential (MAC) FIR filter -- single-process reference implementation.
 *
 * Usage:  mpirun -n 1 ./fir_serial <N_taps> <L_samples>
 *
 * Writes output to stdout:
 *   N  L  time_seconds
 *
 * CPSC-375 -- High-Performance Computing, Spring 2026
 * Liu Restrepo Sanabria
 * ========================================================================== */

#include "fir_common.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N_taps> <L_samples>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int  N = atoi(argv[1]);   /* number of filter taps  */
    long L = atol(argv[2]);   /* signal length (samples)*/

    /* ------------------------------------------------------------------ */
    /* Allocate and initialise                                             */
    /* ------------------------------------------------------------------ */
    double *h = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(L * sizeof(double));
    double *y = (double *)calloc(L, sizeof(double));

    if (!h || !x || !y) {
        fprintf(stderr, "[serial] Memory allocation failed.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    generate_fir_coefficients(h, N, 0.25);   /* cutoff = Fs/4 */
    generate_signal(x, L, 42);

    /* ------------------------------------------------------------------ */
    /* Timed filtering                                                     */
    /* ------------------------------------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Direct-form MAC FIR: y[n] = sum_{k=0}^{N-1} h[k] * x[n-k] */
    for (long n = N - 1; n < L; n++) {
        double acc = 0.0;
        for (int k = 0; k < N; k++)
            acc += h[k] * x[n - k];
        y[n] = acc;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    printf("%d %ld %.6f\n", N, L, t_end - t_start);

    /* Save reference output for verification by parallel implementations */
    char fname[64];
    snprintf(fname, sizeof(fname), "ref_N%d_L%ld.bin", N, L);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(y, sizeof(double), L, f);
        fclose(f);
    }

    free(h); free(x); free(y);
    MPI_Finalize();
    return EXIT_SUCCESS;
}