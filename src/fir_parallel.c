/* ==========================================================================
 * fir_parallel.c
 * Parallel FIR filter -- data decomposition with nearest-neighbour
 * halo exchange (MPI_Sendrecv).
 *
 * The input signal of length L is divided into P contiguous chunks.
 * Before filtering, each rank receives N-1 "halo" samples from its
 * left neighbour via MPI_Sendrecv.  The full filter (all N taps) is
 * then applied locally to the extended chunk.
 *
 * Analogue to: fully parallel systolic FIR (Xilinx Ch. 5)
 *   Each clock cycle all N taps fire simultaneously.
 *   Here each rank applies all N taps to its chunk of the signal.
 *   Communication volume: O(N) per rank (halo exchange, independent of L).
 *
 * Usage:  mpirun -n <P> ./fir_parallel <N_taps> <L_samples>
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
    /* Decompose signal across ranks                                       */
    /* Each rank gets chunk = ceil(L/P) samples; last rank may get fewer  */
    /* ------------------------------------------------------------------ */
    long chunk = (L + size - 1) / size;          /* samples per rank       */
    long my_start = (long)rank * chunk;
    long my_end   = my_start + chunk;
    if (my_end > L) my_end = L;
    long my_len   = my_end - my_start;

    int left_rank  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* ------------------------------------------------------------------ */
    /* Allocate                                                            */
    /* ------------------------------------------------------------------ */
    double *h       = (double *)malloc(N * sizeof(double));
    double *x_full  = NULL;
    if (rank == 0)
        x_full = (double *)malloc(L * sizeof(double));

    double *local_x = (double *)malloc(my_len * sizeof(double));
    double *halo    = (double *)calloc(N - 1, sizeof(double));
    double *buf     = (double *)malloc((my_len + N - 1) * sizeof(double));
    double *local_y = (double *)calloc(my_len, sizeof(double));
    double *y       = NULL;
    if (rank == 0)
        y = (double *)calloc(L, sizeof(double));

    if (!h || !local_x || !halo || !buf || !local_y ||
        (rank == 0 && (!x_full || !y))) {
        fprintf(stderr, "[rank %d] Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ------------------------------------------------------------------ */
    /* Initialise                                                          */
    /* ------------------------------------------------------------------ */
    generate_fir_coefficients(h, N, 0.25);

    if (rank == 0)
        generate_signal(x_full, L, 42);

    /* ------------------------------------------------------------------ */
    /* Scatter signal to all ranks                                         */
    /* Using a simple approach: rank 0 sends each chunk individually.     */
    /* For large L, MPI_Scatterv would be more efficient.                 */
    /* ------------------------------------------------------------------ */
    if (rank == 0) {
        memcpy(local_x, x_full, my_len * sizeof(double));
        for (int r = 1; r < size; r++) {
            long r_start = (long)r * chunk;
            long r_end   = r_start + chunk;
            if (r_end > L) r_end = L;
            long r_len   = r_end - r_start;
            MPI_Send(x_full + r_start, (int)r_len, MPI_DOUBLE,
                     r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_x, (int)my_len, MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ------------------------------------------------------------------ */
    /* Timed section: halo exchange + local filtering                     */
    /* ------------------------------------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /*
     * Halo exchange: each rank sends its first N-1 samples to the left
     * neighbour (which needs them to finish its right boundary), and
     * receives N-1 samples from its left neighbour into halo[].
     *
     * MPI_Sendrecv avoids deadlock and is more efficient than separate
     * MPI_Send / MPI_Recv pairs.
     */
    MPI_Sendrecv(
        local_x,    N - 1, MPI_DOUBLE, left_rank,  0,   /* send left  */
        halo,       N - 1, MPI_DOUBLE, left_rank,  0,   /* recv left  */
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    /*
     * Note: rank 0 has no left neighbour (left_rank = MPI_PROC_NULL),
     * so its halo[] remains zero-initialised -- equivalent to zero-padding
     * the signal before the first sample.
     */

    /* Build extended buffer: [halo | local_x] */
    memcpy(buf,         halo,    (N - 1) * sizeof(double));
    memcpy(buf + N - 1, local_x, my_len  * sizeof(double));

    /* Apply full N-tap filter to local chunk */
    for (long n = N - 1; n < my_len + N - 1; n++) {
        double acc = 0.0;
        for (int k = 0; k < N; k++)
            acc += h[k] * buf[n - k];
        local_y[n - (N - 1)] = acc;
    }

    /* Gather results at rank 0 */
    MPI_Gather(local_y, (int)my_len, MPI_DOUBLE,
               y,       (int)my_len, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* ------------------------------------------------------------------ */
    /* Verification and report (rank 0 only)                              */
    /* ------------------------------------------------------------------ */
    if (rank == 0) {
        char fname[64];
        snprintf(fname, sizeof(fname), "ref_N%d_L%ld.bin", N, L);
        FILE *f = fopen(fname, "rb");
        if (f) {
            double *ref = (double *)malloc(L * sizeof(double));
            fread(ref, sizeof(double), L, f);
            fclose(f);
            double err = verify(ref, y, L, N);
            fprintf(stderr, "[parallel] max_abs_error = %.3e\n", err);
            free(ref);
        }
        printf("%d %ld %d %.6f\n", N, L, size, t_end - t_start);
    }

    free(h); free(local_x); free(halo); free(buf); free(local_y);
    if (rank == 0) { free(x_full); free(y); }
    MPI_Finalize();
    return EXIT_SUCCESS;
}