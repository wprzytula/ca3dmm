#include "densematgen.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <vector>
#include <cassert>
#include <cmath>
#include <memory>

#ifdef PRINT
#define debug(x) x
#define DEBUG_INT(var) printf(#var "=%i\n", var)
#else
#define debug(x)
#define DEBUG_INT(var)
#endif

#ifdef NO_DECIMAL
#undef NO_DECIMAL
#define NO_DECIMAL true
#else
#define NO_DECIMAL false
#endif



using f = double;

namespace {

constexpr double const l = 0.95;
constexpr double const EPS = 0.0001;

constexpr int ceil(int x, int y) {
    return (x + y - 1) / y;
}


// Possibly go up from `what` to the first number divisible by `wrt`.
constexpr int pad(int const what, int const wrt) {
    int const quotient_ceiling = ceil(what, wrt);
    return wrt * quotient_ceiling;
}

#define norem_div(dividend, divisor) ({\
    int const res = (dividend) / (divisor);\
    assert(res * divisor == dividend);\
    res;\
})


// Borrowed from: https://stackoverflow.com/a/75458495
#define MPI_CHECK(n) __check_mpi_error(__FILE__, __LINE__, n)
inline void __check_mpi_error(const char *file, const int line, const int n)
{
    char errbuffer[MPI::MAX_ERROR_STRING];
    int errlen;

    if (n != MPI_SUCCESS)
    {
        MPI_Error_string(n, errbuffer, &errlen);
        printf("MPI-error: %s\n", errbuffer);
        printf("Location: %s:%i\n", file, line);
        MPI::COMM_WORLD.Abort(n);
    }
}

void print_array(char const *name, f const *arr, int const len) {
    printf("Array %s: \n", name);
    for (int i = 0; i < len; ++i) {
        printf("%.0f ", arr[i]);
    }
    printf("\n");
}

struct Config
{
    bool unused;

    int n, m, k;
    int p, p_n, p_m, p_k, p_all;
    int n_padded, m_padded, k_padded;

    int pk_group_procs_num = -1;
    int pk_groups_num = -1;
    int procs_num_per_chunk_along_k = -1;
    int pillars_per_pk_group = -1;
    int gidx = -1;

    int global_rank, pk_group_rank = -1, cannon_group_rank = -1;
    MPI_Comm pk_group_comm, cannon_group_comm, pk_groups_leaders_comm, cannon_groups_leaders_comm, pk_group_counterparts_comm;
    int pk_group_size = -1, cannon_group_size = -1, cannon_group_dim = -1;
    int cannon_groups_num = -1;
    int cannon_coords[2];

    int chunk_a_vertical_len = -1;
    int chunk_b_horizontal_len = -1;
    int chunk_along_k_len = -1;
    int a_chunk_size = -1;
    int b_chunk_size = -1;
    int c_chunk_size = -1;

    int left_neigh_rank = -1;
    int right_neigh_rank = -1;
    int up_neigh_rank = -1;
    int down_neigh_rank = -1;

    int pk_groups_leaders_rank = -1;
    int cannon_groups_leaders_rank = -1;

    bool is_cannon_group_leader = false;

    static int minimised_sum(int const m, int const n, int const k, int const p_m, int const p_n, int const p_k) {
        return p_m * k * n + p_n * m * k + p_k * m * n;
    }

    Config(int const n, int const m, int const k, int const p, int const global_rank)
        : n{n}, m{m}, k{k}, p{p}, global_rank{global_rank}
    {
        /* Solving the equation optimally */

        // Solve min(p_m*k*n + p_n*m*k + p_k*m*n) with constraints:
        // (a) l*p ≤ p_m*p_n*p_k ≤ p (l is a constant, e.g. 0.95; this ensures we
        // use
        //     as many processors as possible, but not necessarily all processors
        //     available)
        // (b) mod(max(p_m, p_n), min(p_m, p_n)) == 0 (this ensures the Cannon
        // algorithm can be performed). Among admissible, optimal solutions, pick
        // the one using as many processors as possible, thus maximizing
        // p_m*p_n*p_k.
        int opt_p_m = p;
        int opt_p_n = 1;
        int opt_p_k = 1;
        int max_p_prod = opt_p_m * opt_p_n * opt_p_k;
        int min_sum = minimised_sum(m, n, k, opt_p_m, opt_p_n, opt_p_k);

        double const l_p = l * p;

        for (int p_m = 1; p_m <= p; ++p_m)
        {
            if (p_m * p_m > p)
                break; // No point in continuing if p_m * p_m exceeds p
            int const p_div_p_m = p / p_m;
            for (int p_n = p_m; p_n <= p_div_p_m; p_n += p_m)
            {
                int const p_k = p / (p_m * p_n);
                int const p_prod = p_k * p_m * p_n;
                int const sum = minimised_sum(m, n, k, p_m, p_n, p_k);
                if (l_p <= p_prod && p_prod <= p &&
                    (sum < min_sum || (sum == min_sum && p_prod > max_p_prod)))
                {
                    min_sum = sum;
                    max_p_prod = p_prod;
                    opt_p_n = p_n;
                    opt_p_m = p_m;
                    opt_p_k = p_k;
                }
            }
        }
        p_n = opt_p_n;
        p_m = opt_p_m;
        p_k = opt_p_k;
        p_all = max_p_prod;
        unused = global_rank >= p_all;

        m_padded = pad(m, p_m);
        n_padded = pad(n, p_n);
        // k_padded = pad(k, p_k);


        /* P_k groups config & intracommunication */

        pk_groups_num = p_k;
        pk_group_procs_num = p_m * p_n;

        chunk_a_vertical_len = norem_div(m_padded, p_m);
        chunk_b_horizontal_len = norem_div(n_padded, p_n);
        procs_num_per_chunk_along_k = norem_div(pk_group_procs_num, std::max(p_m, p_n));
        k_padded = pad(k, p_k * procs_num_per_chunk_along_k);
        pillars_per_pk_group = norem_div(k_padded, p_k);
        chunk_along_k_len = norem_div(pillars_per_pk_group, procs_num_per_chunk_along_k);
        a_chunk_size = chunk_a_vertical_len * chunk_along_k_len;
        b_chunk_size = chunk_b_horizontal_len * chunk_along_k_len;
        c_chunk_size = chunk_a_vertical_len * chunk_b_horizontal_len;

        gidx = global_rank / pk_group_procs_num;

        if (unused) {
            debug(
                printf("Process %i: unused, early return.", global_rank);
            )
            return;
        }

        MPI_CHECK(MPI_Comm_split(
            MPI_COMM_WORLD,
            global_rank < p_all ? gidx : MPI_UNDEFINED,
            global_rank,
            &pk_group_comm
        ));
        if (pk_group_comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Comm_size(pk_group_comm, &pk_group_size));
            assert(pk_group_size == p_m * p_n);
        }
        MPI_CHECK(MPI_Comm_set_errhandler(pk_group_comm, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Comm_rank(pk_group_comm, &pk_group_rank));


        /* Cannon groups config & intracommunication */

        cannon_groups_num = std::max(p_m, p_n) / std::min(p_m, p_n);
        cannon_group_size = norem_div(pk_group_size, cannon_groups_num);
        cannon_group_dim = std::min(p_m, p_n);
        assert(cannon_group_size == cannon_group_dim * cannon_group_dim);

        MPI_CHECK(MPI_Comm_split(
            pk_group_comm,
            pk_group_rank / cannon_group_size,
            pk_group_rank,
            &cannon_group_comm
        ));
        MPI_CHECK(MPI_Comm_set_errhandler(cannon_group_comm, MPI_ERRORS_RETURN));
        {
            int cgrpsz;
            MPI_CHECK(MPI_Comm_size(cannon_group_comm, &cgrpsz));
            assert(cannon_group_size == cgrpsz);
        }
        MPI_CHECK(MPI_Comm_rank(cannon_group_comm, &cannon_group_rank));

        int dims[2] = {cannon_group_dim, cannon_group_dim};
        int periods[2] = {true, true};

        MPI_CHECK(MPI_Cart_create(cannon_group_comm, 2, dims, periods, false, &cannon_group_comm));
        MPI_CHECK(MPI_Comm_set_errhandler(cannon_group_comm, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Cart_coords(cannon_group_comm, cannon_group_rank, 2, cannon_coords));

        MPI_CHECK(MPI_Cart_shift(cannon_group_comm, 0, 1, &left_neigh_rank, &right_neigh_rank));
        MPI_CHECK(MPI_Cart_shift(cannon_group_comm, 1, 1, &up_neigh_rank, &down_neigh_rank));


        /* Matrix parts distribution communication */
        bool const is_pk_group_leader = global_rank % pk_group_size == 0;
        MPI_CHECK(MPI_Comm_split(
            MPI_COMM_WORLD,
            is_pk_group_leader,
            global_rank,
            &pk_groups_leaders_comm
        ));
        if (is_pk_group_leader) {
            MPI_CHECK(MPI_Comm_rank(pk_groups_leaders_comm, &pk_groups_leaders_rank));
            int sz;
            MPI_CHECK(MPI_Comm_size(pk_groups_leaders_comm, &sz));
            assert(sz == pk_groups_num);
        } else {
            pk_groups_leaders_comm = MPI_COMM_NULL;
        }

        is_cannon_group_leader = global_rank % cannon_group_size == 0;
        MPI_CHECK(MPI_Comm_split(
            MPI_COMM_WORLD,
                // all non-leaders get color 0, all leaders get color equal to idx of their pk_group.
            is_cannon_group_leader ? (global_rank / pk_group_size) : MPI_UNDEFINED,
            global_rank,
            &cannon_groups_leaders_comm
        ));
        if (is_cannon_group_leader) {
            MPI_CHECK(MPI_Comm_rank(cannon_groups_leaders_comm, &cannon_groups_leaders_rank));
        } else {
            cannon_groups_leaders_comm = MPI_COMM_NULL;
        }

        /* Results reduction communication */
        MPI_CHECK(MPI_Comm_split(
            MPI_COMM_WORLD,
            global_rank % pk_group_size,
            global_rank,
            &pk_group_counterparts_comm
        ));
        {
            int sz;
            MPI_CHECK(MPI_Comm_size(pk_group_counterparts_comm, &sz));
            assert(sz == pk_groups_num);
        }

    }

#define SEP "\n"
    void print() const
    {
        printf("\n\tCONFIG:" SEP "\tGLOBAL:" SEP "n=%i," SEP "m=%i," SEP "k=%i," SEP "p=%i" SEP "--->" SEP "p_m=%i," SEP "p_n=%i," SEP "p_k=%i," SEP "p_all=%i" SEP
               "(prod:%i, sum: %i)," SEP
               "k_padded=%i," SEP "m_padded=%i," SEP "n_padded=%i," SEP
               "pk_groups_num=%i, " SEP "pk_group_procs_num=%i," SEP "pillars_per_pk_group=%i," SEP "pk_group_size=%i," SEP
               "cannon_groups_num=%i," SEP "cannon_group_size=%i," SEP
               "procs_num_per_chunk_along_k=%i," SEP "chunk_a_vertical_len=%i," SEP "chunk_b_horizontal_len=%i," SEP "chunk_along_k_len=%i," SEP
               "\tLOCAL:" SEP "global_rank=%i," SEP "gidx=%i," SEP "pk_group_rank=%i," SEP "cannon_group_rank=%i," SEP
               "left_neigh_rank=%i," SEP "right_neigh_rank=%i," SEP "up_neigh_rank=%i," SEP "down_neigh_rank=%i," SEP
               "pk_groups_leaders_rank=%i," SEP "cannon_groups_leaders_rank=%i\n\n",
               n, m, k, p, p_m, p_n, p_k, p_all,
               p_n * p_m * p_k, minimised_sum(m, n, k, p_m, p_n, p_k),
               k_padded, m_padded, n_padded,
               pk_groups_num, pk_group_procs_num, pillars_per_pk_group, pk_group_size,
               cannon_groups_num, cannon_group_size,
               procs_num_per_chunk_along_k, chunk_a_vertical_len, chunk_b_horizontal_len, chunk_along_k_len,
               global_rank, gidx, pk_group_rank, cannon_group_rank,
               left_neigh_rank, right_neigh_rank, up_neigh_rank, down_neigh_rank,
               pk_groups_leaders_rank, cannon_groups_leaders_rank
        );
    }

/* CANNON ALGORITHM */
    void preskew_A(f *buf) const {
        int preskew_dest, preskew_src;
        MPI_Cart_shift(
            cannon_group_comm,
            0,
            -cannon_coords[1], // left
            &preskew_src,
            &preskew_dest
        );
        debug(
            printf("Issuing Preskew A Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, preskew_dest, cannon_group_rank, preskew_src);
        )
        MPI_Sendrecv_replace(
            buf,
            chunk_a_vertical_len * chunk_along_k_len,
            MPI_DOUBLE,
            preskew_dest,
            0,
            preskew_src,
            0,
            cannon_group_comm,
            MPI_STATUS_IGNORE
        );
        debug(
            printf("Completed Preskew A Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, preskew_dest, cannon_group_rank, preskew_src);
        )
    }

    void preskew_B(f *buf) const {
        int preskew_dest, preskew_src;
        MPI_Cart_shift(
            cannon_group_comm,
            1,
            -cannon_coords[0], // up
            &preskew_src,
            &preskew_dest
        );
        debug(
            printf("Issuing Preskew B Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, preskew_dest, cannon_group_rank, preskew_src);
        )
        MPI_Sendrecv_replace(
            buf,
            chunk_b_horizontal_len * chunk_along_k_len,
            MPI_DOUBLE,
            preskew_dest,
            0,
            preskew_src,
            0,
            cannon_group_comm,
            MPI_STATUS_IGNORE
        );
        debug(
            printf("Completed Preskew B Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, preskew_dest, cannon_group_rank, preskew_src);
        )
    }

    void cannon_step_A(f *buf) const {
        debug(
            printf("Issuing A Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, left_neigh_rank, cannon_group_rank, right_neigh_rank);
        )
		MPI_Sendrecv_replace(buf, chunk_a_vertical_len * chunk_along_k_len, MPI_DOUBLE, left_neigh_rank, 0, right_neigh_rank, 0, cannon_group_comm, MPI_STATUS_IGNORE);
        debug(
            printf("Completed A Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, left_neigh_rank, cannon_group_rank, right_neigh_rank);
        )
    }

    void cannon_step_B(f *buf) const {
        debug(
            printf("Issuing B Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, up_neigh_rank, cannon_group_rank, down_neigh_rank);
        )
        MPI_Sendrecv_replace(buf, chunk_b_horizontal_len * chunk_along_k_len, MPI_DOUBLE, up_neigh_rank, 0, down_neigh_rank, 0, cannon_group_comm, MPI_STATUS_IGNORE);
        debug(
            printf("Completed B Sendrecv_replace in pkgroup %i: %i <- %i <- %i\n", gidx, up_neigh_rank, cannon_group_rank, down_neigh_rank);
        )
    }

    void multiply_locally(f const *A, f const *B, f *C) const {
        // Multiply locally
        for (int i = 0; i < chunk_a_vertical_len; ++i) {
            for (int j = 0; j < chunk_b_horizontal_len; ++j) {
                f sum = 0;
                for (int k = 0; k < chunk_along_k_len; ++k) {
                    sum += A[i * chunk_along_k_len + k] * B[j * chunk_along_k_len + k];
                }
                C[i * chunk_b_horizontal_len + j] += sum;
            }
        }
    }

    void cannon_algorithm(f *A, f *B, f *C) const {
        if (cannon_group_size > 1) {
            preskew_A(A);
            debug(
                print_array("A chunk after preskew", A, a_chunk_size);
            )
            preskew_B(B);
            debug(
                print_array("B chunk after preskew", B, b_chunk_size);
            )
        }
        multiply_locally(A, B, C);
        for (int _shift = 1; _shift < cannon_group_dim; ++_shift) {
            // TODO: Init both async and wait for them
            cannon_step_A(A);
            debug(
                print_array("A chunk after cannon step", A, a_chunk_size);
            )
            cannon_step_B(B);
            debug(
                print_array("B chunk after cannon step", B, b_chunk_size);
            )
            multiply_locally(A, B, C);
        }
    }

/* MATRIX GENERATION AND DISTRIBUTION */
    void generate_matrix_A_part(f *A, int const seed, int const pk_group_idx) const {
        for (int r = 0; r < m_padded; r++) {
            for (int c = 0; c < pillars_per_pk_group; c++) {
                int const real_matrix_row = r;
                int const real_matrix_col = pk_group_idx * pillars_per_pk_group + c;
                bool const out_of_bounds = real_matrix_row >= m || real_matrix_col >= k;
                const f entry = out_of_bounds ?
                    ({
                        debug(
                            printf("Generating 0 for A[%i,%i]\n", real_matrix_row, real_matrix_col);
                        )
                        0;
                    }) :
                    ({
                        f const entry = generate_double(seed, real_matrix_row, real_matrix_col);
                        debug(
                            printf("Generating entry=%3.0f for A[%i,%i]\n", entry, real_matrix_row, real_matrix_col);
                        )
                        entry;
                    });

                int const chunk_col = c / chunk_along_k_len;
                int const chunk_col_offset = c % chunk_along_k_len;
                int const chunk_row = r / chunk_a_vertical_len;
                int const chunk_row_offset = r % chunk_a_vertical_len;
                int const chunk_idx = chunk_col * p_m + chunk_row;
                int const chunk_offset = chunk_row_offset * chunk_along_k_len + chunk_col_offset;

                // debug(
                //     printf("Placing entry in chunk no %i, at offset %i: A[%i]\n",
                //            chunk_idx, chunk_offset, chunk_idx * chunk_size + chunk_offset);
                // )
                A[chunk_idx * a_chunk_size + chunk_offset] = entry;
            }
        }
    }

    void generate_matrix_B_part(f *B, int const seed, int const pk_group_idx) const {
        int const chunk_size = chunk_b_horizontal_len * chunk_along_k_len;

        for (int chunk_col = 0; chunk_col < p_n; ++chunk_col) {
            for (int chunk_row = 0; chunk_row < procs_num_per_chunk_along_k; ++chunk_row) {
                for (int chunk_col_offset = 0; chunk_col_offset < chunk_b_horizontal_len; chunk_col_offset++) {
                    for (int chunk_row_offset = 0; chunk_row_offset < chunk_along_k_len; chunk_row_offset++) {

                        int const real_matrix_row = pk_group_idx * pillars_per_pk_group + chunk_row * chunk_along_k_len + chunk_row_offset;
                        int const real_matrix_col = chunk_col * chunk_b_horizontal_len + chunk_col_offset;
                        bool const out_of_bounds = real_matrix_col >= n || real_matrix_row >= k;

                        const f entry = out_of_bounds ?
                            ({
                                debug(
                                    printf("Generating 0 for B[%i,%i]\n", real_matrix_row, real_matrix_col);
                                )
                                0;
                            }) :
                            ({
                                f const entry = generate_double(seed, real_matrix_row, real_matrix_col);
                                debug(
                                    printf("Generating entry=%3.0f for B[%i,%i]\n", entry, real_matrix_row, real_matrix_col);
                                )
                                entry;
                            });

                        int const chunk_idx = chunk_col * procs_num_per_chunk_along_k + chunk_row;
                        int const chunk_offset = chunk_col_offset * chunk_along_k_len + chunk_row_offset;

                        debug(
                            printf("Placing entry %.0f in chunk no %i, at offset %i: B[%i]\n",
                                    entry, chunk_idx, chunk_offset, chunk_idx * b_chunk_size + chunk_offset);
                        )
                        B[chunk_idx * b_chunk_size + chunk_offset] = entry;
                    }
                }
            }
        }
    }

    void distribute_to_pk_groups(
        f *A_B_chunks,
        int const seed,
        int const pk_group_vals,
        char const *name,
        void (Config::*generate)(f*, int, int) const
    ) const {
        if (global_rank == 0) {
            for (int pk_group_idx = 1; pk_group_idx < p_k; ++pk_group_idx) {
                (this->*generate)(A_B_chunks, seed, pk_group_idx);
                MPI_CHECK(MPI_Send(
                    A_B_chunks,
                    pk_group_vals,
                    MPI_DOUBLE,
                    pk_group_idx,
                    0,
                    pk_groups_leaders_comm
                ));
            }
            (this->*generate)(A_B_chunks, seed, 0);
            debug(
                print_array(name, A_B_chunks, pk_group_vals);
            )
        } else if (pk_group_rank == 0) {
            MPI_CHECK(MPI_Recv(
                A_B_chunks,
                pk_group_vals,
                MPI_DOUBLE,
                0,
                0,
                pk_groups_leaders_comm,
                MPI_STATUS_IGNORE
            ));
            debug(
                print_array(name, A_B_chunks, pk_group_vals);
            )
        }
    }

    void distribute_A_to_pk_groups(f *A_B_chunks, int const seed_a) const {
        int const pk_group_vals_a = pillars_per_pk_group * m_padded;
        distribute_to_pk_groups(A_B_chunks, seed_a, pk_group_vals_a, "A chunks for pk_group", &Config::generate_matrix_A_part);
    }

    void distribute_B_to_pk_groups(f *A_B_chunks, int const seed_b) const {
        int const pk_group_vals_b = pillars_per_pk_group * n_padded;
        distribute_to_pk_groups(A_B_chunks, seed_b, pk_group_vals_b, "B chunks for pk_group", &Config::generate_matrix_B_part);
    }

    void replicate_among_cannon_groups(f *A_B_chunks, int const cannon_group_chunks_size) const {
        if (cannon_groups_leaders_comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Bcast(
                A_B_chunks,
                cannon_group_chunks_size,
                MPI_DOUBLE,
                0,
                cannon_groups_leaders_comm
            ));
        }
    }

    void distribute_among_cannon_groups(f *A_B_chunks, int const cannon_group_chunks_size) const {
        if (cannon_groups_leaders_comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Scatter(
                A_B_chunks,
                cannon_group_chunks_size,
                MPI_DOUBLE,
                A_B_chunks,
                cannon_group_chunks_size,
                MPI_DOUBLE,
                0,
                cannon_groups_leaders_comm
            ));
        }
    }

    void distribute_in_cannon_groups(f const* A_B_chunks, f *chunk, int const chunk_size) const {
        debug(
            if (is_cannon_group_leader)
                print_array("A/B chunks for my cannon group", A_B_chunks, chunk_size * cannon_group_size);
        )
        MPI_CHECK(MPI_Scatter(
            A_B_chunks,
            chunk_size,
            MPI_DOUBLE,
            chunk,
            chunk_size,
            MPI_DOUBLE,
            0,
            cannon_group_comm
        ));
    }

    int compute_ge(f const* C_chunk, f const ge_val) const {
        // Compute locally
        int const chunk_row_idx = global_rank % p_m;
        int const chunk_col_idx = global_rank / p_m;

        int const chunk_row_offset = chunk_row_idx * chunk_a_vertical_len;
        int const chunk_col_offset = chunk_col_idx * chunk_b_horizontal_len;
        debug(
            printf("rank %i: chunk idx: (row=%i, col=%i), offset: (row=%i, col=%i)\n",
                global_rank, chunk_row_idx, chunk_col_idx, chunk_row_offset, chunk_col_offset
            );
        )

        // TODO: fix case: --np 2 build/ca3dmm 6 6 6 -s 3,2 -g 55
        int count = 0;
        for (int i = 0; i < chunk_a_vertical_len; ++i) {
            if (chunk_row_offset + i >= m) break; // out of bounds
            for (int j = 0; j < chunk_b_horizontal_len; ++j) {
                if (chunk_col_offset + j >= n) break; // out of bounds
                if (C_chunk[i * chunk_b_horizontal_len + j] >= ge_val) {
                    ++count;
                }
            }
        }

        // Reduce in pk_groups
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            &count,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD
        ));

        // Now, `count` contains the answer.
        return count;
    }

    std::vector<std::vector<f>> populate_A(int const seed_a) const {
        // Step 1: Allocate and populate matrix A
        std::vector<std::vector<f>> A(m, std::vector<f>(k));

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                A[i][j] = generate_double(seed_a, i, j);
            }
        }

        return A;
    }

    std::vector<std::vector<f>> populate_B(int const seed_b) const {
        // Step 1: Allocate and populate matrix B
        std::vector<std::vector<f>> B(k, std::vector<f>(n));

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                B[i][j] = generate_double(seed_b, i, j);
            }
        }

        return B;
    }

    std::vector<std::vector<f>> compute_expected_C(std::vector<std::vector<f>> const& A, std::vector<std::vector<f>> const& B) const {
        // Step 2: Allocate matrix C
        std::vector<std::vector<f>> C(m, std::vector<f>(n));

        // Step 3: Multiply matrices A and B to obtain C
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                f sum = 0.0;
                for (int r = 0; r < k; ++r) {
                    sum += A[i][r] * B[r][j];
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    void print_result_matrix(f const *C_chunk) const {
        if (global_rank == 0)
            printf("%d %d\n", m, n);
        // For row in C matrix:
        std::unique_ptr<f[]> chunk_row_up{new f[chunk_b_horizontal_len]};
        for (int r = 0; r < m; ++r) {
            // For chunk row in row:
            for (int chunk_col_idx = 0; chunk_col_idx < p_n; ++chunk_col_idx) {
                // Find the owner of the chunk row.
                int const owner_pk_group_rank = chunk_col_idx * p_m + r / chunk_a_vertical_len;

                f *chunk_row;
                if (owner_pk_group_rank == pk_group_rank) {
                    // If I own this row, broadcast to others
                    int const row_offset_in_chunk = r % chunk_a_vertical_len * chunk_b_horizontal_len;
                    chunk_row = std::remove_const_t<f*>(C_chunk + row_offset_in_chunk);
                } else {
                    chunk_row = chunk_row_up.get();
                }

                // Broadcast chunk row.
                MPI_CHECK(MPI_Bcast(
                    chunk_row,
                    chunk_b_horizontal_len,
                    MPI_DOUBLE,
                    owner_pk_group_rank,
                    pk_group_comm
                ));

                if (global_rank == 0) {
                    // For elem in chunk row:
                    for (int i = 0; i < chunk_b_horizontal_len; ++i) {
                        // Print elem
                        if (NO_DECIMAL) {
                            printf("%3.0f ", chunk_row[i]);
                        } else {
                            printf("%f ", chunk_row[i]);
                        }
                    }
                }
            }
            // Print newline at the row end
            if (global_rank == 0)
                printf("\n");
        }
    }

    int expected_ge(std::vector<std::vector<f>> const& C, f const ge_value) const {
        // Step 4: Count the number of elements in C >= ge
        int count = 0;  // Counter for the elements
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (C[i][j] >= ge_value) {
                    count++;
                }
            }
        }

        return count;
    }

    void print_expected_matrix(std::vector<std::vector<f>> const& M) const {
        int const rows = M.size();
        int const cols = M[0].size();
        printf("%d %d\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (NO_DECIMAL) {
                    printf("%3.0f ", M[i][j]);
                } else {
                    printf("%f ", M[i][j]);
                }
            }
            printf("\n");
        }
    }

/* COMPLETE ALGORITHM */
    void multiply(int const seed_a, int const seed_b, bool const verbose,
                  bool const ge, double const ge_value) const
    {
    // Algorithm:
    // 2. Organize processes into pk groups, each group has pm×pn processes.
    // 3. Assign A to pk groups in a block-column order; assign B to pk groups in a block-row order.
    //    Denote by Ai the block-column of A assigned to group i; and by Bi the block-row of B assigned to group i.
    // 4. Within the i-th group: organize processes into c=max(pm,pn)/min(pm,pn) Cannon groups. For example,
    // assume pn>pm. Bi is then not replicated. Ai is replicated c times — each Cannon group stores its complete copy of Ai.
    // 5. Perform the Cannon's algorithm in each Cannon group and in each of the pk groups to get Ci.
    // 6. Reduce C=∑ki=1Ci.

        debug(
            if (global_rank == 0) {
                print();
            }
        )

        /* Distribute to pk groups */
        int const pk_group_vals_a = pillars_per_pk_group * m_padded;
        int const pk_group_vals_b = pillars_per_pk_group * n_padded;

        std::unique_ptr<f[]> A_B_chunks;
        if (is_cannon_group_leader) {
            A_B_chunks = std::make_unique<f[]>(std::max(pk_group_vals_a, pk_group_vals_b));
        }
        std::unique_ptr<f[]> A_chunk{new f[a_chunk_size]};
        std::unique_ptr<f[]> B_chunk{new f[b_chunk_size]};
        std::unique_ptr<f[]> C_chunk{new f[c_chunk_size]()};

        {
            f *chunks = A_B_chunks.get();
            bool const cannon_groups_placed_horizontally = p_m < p_n;

            distribute_A_to_pk_groups(chunks, seed_a);
            if (cannon_groups_num > 1) {
                if (cannon_groups_placed_horizontally) {
                    replicate_among_cannon_groups(chunks, pk_group_vals_a);
                } else {
                    distribute_among_cannon_groups(chunks, pk_group_vals_a / cannon_groups_num);
                }
            }
            distribute_in_cannon_groups(chunks, A_chunk.get(), a_chunk_size);
            debug(
                printf("p%i: ", global_rank); print_array("A chunk", A_chunk.get(), a_chunk_size);
            )

            distribute_B_to_pk_groups(chunks, seed_b);
            if (cannon_groups_num > 1) {
                if (!cannon_groups_placed_horizontally) {
                    replicate_among_cannon_groups(chunks, pk_group_vals_b);
                } else {
                    distribute_among_cannon_groups(chunks, pk_group_vals_b / cannon_groups_num);
                }
            }
            distribute_in_cannon_groups(chunks, B_chunk.get(), b_chunk_size);
            debug(

                printf("p%i: ", global_rank); print_array("B chunk", B_chunk.get(), b_chunk_size);
            )
        }


        /* Perform Cannon algorithm */
        cannon_algorithm(A_chunk.get(), B_chunk.get(), C_chunk.get());

        /* Reduce matrix C to pk_group 0. */
        if (pk_groups_num > 1) {
            MPI_CHECK(MPI_Allreduce(
                MPI_IN_PLACE,
                C_chunk.get(),
                c_chunk_size,
                MPI_DOUBLE,
                MPI_SUM,
                pk_group_counterparts_comm
            ));
        }
        // At this point, the whole C is distributed among all pk_groups.

        auto const expected_A = populate_A(seed_a);
        auto const expected_B = populate_B(seed_b);
        auto const expected_C = compute_expected_C(expected_A, expected_B);

        if (ge) {
            int const computed = compute_ge(C_chunk.get(), ge_value);
            int const expected = expected_ge(expected_C, ge_value);
            if (computed == expected ) {
                if (global_rank == 0) {
                    debug(
                        printf("computed ge=%i\n", computed);
                    )
                    printf("%i\n", computed);
                }
            } else {
                fprintf(stderr, "GE MISMATCH!: rank %i, expected=%i, computed=%i\n", global_rank, expected, computed);
            }
        } else if (verbose) {
            if (global_rank == 0) {
                // debug({
                    printf("A matrix:\n");
                    print_expected_matrix(expected_A);

                    printf("B matrix:\n");
                    print_expected_matrix(expected_B);
                // })
                    printf("Expected matrix:\n");
                    print_expected_matrix(expected_C);

                    printf("Computed matrix:\n");
                // })
            }
            print_result_matrix(C_chunk.get());
        }
    }
};
} // namespace

#ifdef TEST
// UNIT TESTS
void test_pad() {
    assert(pad(11, 3) == 12);
    assert(pad(12, 3) == 12);
    assert(pad(3, 3) == 3);
    assert(pad(23, 4) == 24);
    assert(pad(23, 5) == 25);
}

int main(int argc, char *argv[])
{
    test_pad();

    MPI_Init(&argc, &argv);
    int rank = -1;
    MPI_Comm_rank(MPI::COMM_WORLD, &rank);
    int num_proc = -1;
    MPI_Comm_size(MPI::COMM_WORLD, &num_proc);

    Config const conf{323, 123, 231, num_proc, rank};
    if (rank == 37)
        conf.print();

    MPI_Finalize();
    return 0;
}

#else
static void usage(char const *progname)
{
    std::cerr << "Usage: " << progname << " n m k -s seeds [-g ge_value] [-v]\n";
}

int main(int argc, char *argv[])
{
    int p = 0;
    int rank = 0;

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    static char const *const delim = ",";
    char *token = nullptr;

    int n = 0, m = 0, k = 0;

    char *seeds = nullptr, *ge_value_str = nullptr;
    bool verbose = false;

    int option;
    // Process command-line arguments using getopt
    while ((option = getopt(argc, argv, "s:g:v")) != -1)
    {
        switch (option)
        {
        case 's':
            seeds = optarg;
            break;
        case 'g':
            ge_value_str = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    double const ge_value = ge_value_str ? std::stod(ge_value_str) : 0.;
    if (ge_value_str != nullptr && verbose) {
        std::cerr << "Bad combination of optargs: -g and -v are mutually-exclusive." << '\n';
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    // Parse remaining positional arguments n, m, and k
    if (optind + 2 > argc)
    {
        usage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    n = std::stoi(argv[optind++]);
    m = std::stoi(argv[optind++]);
    k = std::stoi(argv[optind++]);

    Config const conf{n, m, k, p, rank};
    if (conf.unused) {
        goto end;
    }

    // Print the parsed values
    debug(
        if (rank == 0) {
            std::cout << "n: " << n << '\n';
            std::cout << "m: " << m << '\n';
            std::cout << "k: " << k << '\n';
            std::cout << "seeds: " << seeds << '\n';
            std::cout << "ge_value: " << ge_value << '\n';
            std::cout << "verbose: " << std::boolalpha << verbose << '\n';
        }
    )

    if (seeds != nullptr)
        token = std::strtok(seeds, delim);

    while (token != nullptr)
    {
        int first = std::stoi(token);

        token = std::strtok(nullptr, delim);
        if (token != nullptr)
        {
            int second = std::stoi(token);

            debug(
                std::cout << "Pair: " << first << delim << second << std::endl;
            )
            conf.multiply(first, second, verbose, !!ge_value_str, ge_value);

            token = std::strtok(nullptr, delim);
        }
        else
        {
            std::cerr << "Odd number of seeds. Unpaired: " << first << '\n';
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    end:
    MPI_CHECK(MPI_Finalize());
    return 0;
}
#endif // TEST
