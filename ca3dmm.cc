#include "densematgen.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <vector>
#include <cassert>
#include <cmath>

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
struct Config
{
    bool unused;

    int n, m, k;
    int p, p_n, p_m, p_k, p_all;
    int n_padded, m_padded, k_padded;

    int pk_groups_num = -1;
    int procs_num_per_chunk_along_k = -1;
    int pillars_per_pk_group = -1;
    int gidx = -1;

    int global_rank, pk_group_rank = -1, cannon_group_rank = -1;
    MPI_Comm pk_group_comm, cannon_group_comm;
    int pk_group_size = -1, cannon_group_size = -1, cannon_group_dim = -1;
    int cannon_groups_num = -1;
    int cannon_coords[2];

    int chunk_a_vertical_len = -1;
    int chunk_b_horizontal_len = -1;
    int chunk_along_k_len = -1;

    int left_neigh_rank = -1;
    int right_neigh_rank = -1;
    int up_neigh_rank = -1;
    int down_neigh_rank = -1;

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
        k_padded = pad(k, p_k);


        /* P_k groups config & intracommunication */

        pk_groups_num = p_k;
        int const pk_group_procs_num = p_m * p_n;

        chunk_a_vertical_len = norem_div(m_padded, p_m);
        chunk_b_horizontal_len = norem_div(n_padded, p_n);
        procs_num_per_chunk_along_k = norem_div(pk_group_procs_num, std::max(p_m, p_n));
        pillars_per_pk_group = norem_div(k_padded, p_k);
        // printf("%d, %d\n", pillars_per_pk_group, procs_num_per_chunk_along_k);
        chunk_along_k_len = norem_div(pillars_per_pk_group, procs_num_per_chunk_along_k);

        gidx = global_rank / pk_group_procs_num;

        if (unused)
            return;

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
    }

    void print() const
    {
        printf("\n\tCONFIG: GLOBAL: n=%i, m=%i, k=%i, p=%i ---> p_m=%i, p_n=%i, p_k=%i, p_all=%i (prod:"
               "%i, sum: %i), k_padded=%i, m_padded=%i, n_padded=%i, "
               "pillars_per_pk_group=%i, pk_group_size=%i, "
               "cannon_groups_num=%i, cannon_group_size=%i, "
               "procs_num_per_chunk_along_k=%i, chunk_a_vertical_len=%i, chunk_b_horizontal_len=%i, chunk_along_k_len=%i, "
               "LOCAL: global_rank=%i, gidx=%i, pk_group_rank=%i, cannon_group_rank=%i, "
               "left_neigh_rank=%i, right_neigh_rank=%i, up_neigh_rank=%i, down_neigh_rank=%i\n",
               n, m, k, p, p_m, p_n, p_k, p_all, p_n * p_m * p_k,
               minimised_sum(m, n, k, p_m, p_n, p_k), k_padded, m_padded, n_padded,
               pillars_per_pk_group, pk_group_size,
               cannon_groups_num, cannon_group_size,
               procs_num_per_chunk_along_k, chunk_a_vertical_len, chunk_b_horizontal_len, chunk_along_k_len,
               global_rank, gidx, pk_group_rank, cannon_group_rank,
               left_neigh_rank, right_neigh_rank, up_neigh_rank, down_neigh_rank
        );
    }

    void preskew_A(f *buf) const {
        int preskew_dest, preskew_src;
        MPI_Cart_shift(
            cannon_group_comm,
            0,
            -cannon_coords[0], // left
            &preskew_src,
            &preskew_dest
        );
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
    }

    void preskew_B(f *buf) const {
        int preskew_dest, preskew_src;
        MPI_Cart_shift(
            cannon_group_comm,
            1,
            -cannon_coords[1], // up
            &preskew_src,
            &preskew_dest
        );
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
    }

    void cannon_step_A(f *buf) const {
		MPI_Sendrecv_replace(buf, chunk_a_vertical_len * chunk_along_k_len, MPI_DOUBLE, left_neigh_rank, 0, right_neigh_rank, 0, cannon_group_comm, MPI_STATUS_IGNORE);
    }

    void cannon_step_B(f *buf) const {
        MPI_Sendrecv_replace(buf, chunk_b_horizontal_len * chunk_along_k_len, MPI_DOUBLE, up_neigh_rank, 0, down_neigh_rank, 0, cannon_group_comm, MPI_STATUS_IGNORE);
    }

    void multiply_locally(f const *A, f const *B, f *C) const {
        // Multiply locally
        for (int i = 0; i < chunk_a_vertical_len; ++i) {
            for (int j = 0; j < chunk_b_horizontal_len; ++j) {
                int sum = 0;
                for (int k = 0; k < chunk_along_k_len; ++k) {
                    sum += A[i * chunk_along_k_len + k] * B[j * chunk_along_k_len + k];
                }
                C[i * chunk_b_horizontal_len + j] += sum;
            }
        }
    }

    void cannon_algorithm(f *A, f *B, f *C) const {
        preskew_A(A);
        preskew_B(B);
        multiply_locally(A, B, C);
        for (int shift = 0; shift < cannon_group_dim; ++shift) {
            // TODO: Init both async and wait for them
            cannon_step_A(A);
            cannon_step_B(B);
            multiply_locally(A, B, C);
        }
    }


    static void generate_matrix(int const i, int const j, int const seed)
    {
        for (int r = 0; r < i; r++)
        {
            for (int c = 0; c < j; c++)
            {
                const double entry = generate_double(seed, r, c);
                std::cout << entry << " ";
            }
            std::cout << std::endl;
        }
    }

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


    // 2. Organize processes into pk groups, each group has pm×pn processes.

        // matrix A
        generate_matrix(m, k, seed_a);
        generate_matrix(k, n, seed_b);
    }

};

static void usage(char const *progname)
{
    std::cerr << "Usage: " << progname << " n m k -s seeds [-g ge_value] [-v]\n";
}
}

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
int main(int argc, char *argv[])
{
    int p = 0;
    int rank = 0;

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    static char const *const delim = ",";
    char *token;

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

    double const ge_value = ge_value_str ? std::stod(ge_value_str) : 0.;

    Config const conf{n, m, k, p, rank};
    if (conf.unused) {
        goto end;
    }
    conf.print();

    // Print the parsed values
    // std::cout << "n: " << n << '\n';
    // std::cout << "m: " << m << '\n';
    // std::cout << "k: " << k << '\n';
    // std::cout << "seeds: " << seeds << '\n';
    // std::cout << "ge_value: " << ge_value << '\n';
    // std::cout << "verbose: " << std::boolalpha << verbose << '\n';

    token = std::strtok(seeds, delim);

    while (token != nullptr)
    {
        int first = std::stoi(token);

        token = std::strtok(nullptr, delim);
        if (token != nullptr)
        {
            int second = std::stoi(token);

            // std::cout << "Pair: " << first << delim << second << std::endl;
            // conf.multiply(first, second, verbose, !!ge_value_str, ge_value);

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