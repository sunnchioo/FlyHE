#include <cassert>
#include <iostream>
#include <random>
#include <tfhe++.hpp>

int main() {
    constexpr uint32_t num_test = 1000;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> binary(0, 1);

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebkfft<TFHEpp::lvl01param>(sk);
    ek.emplaceiksk<TFHEpp::lvl10param>(sk);
    for (int test = 0; test < num_test; test++) {
        bool p = binary(engine) > 0;
        TFHEpp::TLWE<TFHEpp::lvl1param> tlwe =
            TFHEpp::tlweSymEncrypt<TFHEpp::lvl1param>(
                p ? TFHEpp::lvl1param::μ : -TFHEpp::lvl1param::μ,
                TFHEpp::lvl1param::α, sk.key.lvl1);
        TFHEpp::TLWE<TFHEpp::lvl1param> bootedtlwe;
        TFHEpp::GateBootstrapping(bootedtlwe, tlwe, ek);
        bool p2 =
            TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(bootedtlwe, sk.key.lvl1);
        assert(p == p2);
    }
    std::cout << "Passed" << std::endl;
}

// #include <array>
// #include <cassert>
// #include <chrono>
// #include <iostream>
// #include <random>
// #include <tfhe++.hpp>

// int main() {
//     constexpr uint32_t num_test = 100;
//     std::random_device seed_gen;
//     std::default_random_engine engine(seed_gen());
//     std::uniform_int_distribution<uint32_t> binary(0, 1);

//     TFHEpp::SecretKey sk;
//     TFHEpp::EvalKey ek;
//     ek.emplacebkfft<TFHEpp::lvl01param>(sk);
//     ek.emplaceiksk<TFHEpp::lvl10param>(sk);
//     // ek.emplacebkfft<TFHEpp::lvl02param>(sk);
//     // ek.emplaceiksk<TFHEpp::lvl20param>(sk);

//     std::array<TFHEpp::TLWE<TFHEpp::lvl1param>, num_test> tlwe, bootedtlwe;
//     std::array<bool, num_test> p;

//     for (int i = 0; i < num_test; i++) {
//         p[i] = binary(engine) > 0;
//         tlwe[i] = TFHEpp::tlweSymEncrypt<TFHEpp::lvl1param>(
//             p[i] ? TFHEpp::lvl1param::μ : -TFHEpp::lvl1param::μ,
//             TFHEpp::lvl1param::α, sk.key.lvl1);
//     }

//     auto start = std::chrono::system_clock::now();

//     for (int test = 0; test < num_test; test++) {
//         TFHEpp::GateBootstrapping(bootedtlwe[test], tlwe[test], ek);
//     }

//     auto end = std::chrono::system_clock::now();

//     for (int i = 0; i < num_test; i++) {
//         bool p2 = TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(bootedtlwe[i], sk.key.lvl1);
//         assert(p[i] == p2);
//     }

//     std::cout << "Passed" << std::endl;

//     double elapsed =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//     std::cout << "Average time: " << elapsed / num_test << " ms per gate bootstrapping" << std::endl;
// }

// #include <cassert>
// #include <iostream>
// #include <random>
// #include <tfhe++.hpp>

// int main() {
//     constexpr uint32_t num_test = 1000;
//     std::random_device seed_gen;
//     std::default_random_engine engine(seed_gen());
//     std::uniform_int_distribution<uint32_t> binary(0, 1);

//     TFHEpp::SecretKey sk;
//     TFHEpp::EvalKey ek;
//     ek.emplacebkfft<TFHEpp::lvl01param>(sk);
//     ek.emplaceiksk<TFHEpp::lvl10param>(sk);
//     for (int test = 0; test < num_test; test++) {
//         bool p = binary(engine) > 0;
//         TFHEpp::TLWE<TFHEpp::lvl1param> tlwe =
//             TFHEpp::tlweSymEncrypt<TFHEpp::lvl1param>(
//                 p ? TFHEpp::lvl1param::μ : -TFHEpp::lvl1param::μ,
//                 TFHEpp::lvl1param::α, sk.key.lvl1);
//         TFHEpp::TLWE<TFHEpp::lvl1param> bootedtlwe;
//         TFHEpp::GateBootstrapping(bootedtlwe, tlwe, ek);
//         bool p2 =
//             TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(bootedtlwe, sk.key.lvl1);
//         assert(p == p2);
//     }
//     std::cout << "Passed" << std::endl;
// }

// #include <array>
// #include <cassert>
// #include <chrono>
// #include <iostream>
// #include <random>
// #include <tfhe++.hpp>

// int main() {
//     constexpr uint32_t num_test = 100;
//     std::random_device seed_gen;
//     std::default_random_engine engine(seed_gen());
//     std::uniform_int_distribution<uint32_t> binary(0, 1);

//     TFHEpp::SecretKey sk;
//     TFHEpp::EvalKey ek;
//     ek.emplacebkfft<TFHEpp::lvl02param>(sk);
//     ek.emplaceiksk<TFHEpp::lvl20param>(sk);

//     std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, num_test> tlwe, bootedtlwe;
//     std::array<bool, num_test> p;

//     for (int i = 0; i < num_test; i++) {
//         p[i] = binary(engine) > 0;
//         tlwe[i] = TFHEpp::tlweSymEncrypt<TFHEpp::lvl2param>(
//             p[i] ? TFHEpp::lvl2param::μ : -TFHEpp::lvl2param::μ,
//             TFHEpp::lvl2param::α, sk.key.lvl2);
//     }

//     auto start = std::chrono::system_clock::now();

//     for (int test = 0; test < num_test; test++) {
//         TFHEpp::GateBootstrapping(bootedtlwe[test], tlwe[test], ek);
//     }

//     auto end = std::chrono::system_clock::now();

//     for (int i = 0; i < num_test; i++) {
//         bool p2 = TFHEpp::tlweSymDecrypt<TFHEpp::lvl2param>(bootedtlwe[i], sk.key.lvl2);
//         assert(p[i] == p2);
//     }

//     std::cout << "Passed" << std::endl;

//     double elapsed =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//     std::cout << "Average time: " << elapsed / num_test << " ms per gate bootstrapping" << std::endl;
// }
