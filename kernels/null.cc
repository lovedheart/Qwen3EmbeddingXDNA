//===- null.cc ------------------------------------------------ C++ -----===//
// P0 microbench kernel: pure DMA. step touches the object (1B read),
// flush zeroes the drain object. Measures launch floor + shim bandwidth.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef OB
#define OB 3456
#endif

extern "C" {

__attribute__((noinline)) void null_reset() {}

__attribute__((noinline)) void null_step(const uint8_t *__restrict obj) {
    volatile uint8_t v = obj[0];
    (void)v;
}

__attribute__((noinline)) void null_flush(uint8_t *__restrict obj) {
    for (int i = 0; i < OB; i += 64) {
        aie::store_v(obj + i, aie::zeros<uint8_t, 64>());
    }
}

}
