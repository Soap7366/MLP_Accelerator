#!/bin/sh
set -eu
cd "$(dirname "$0")"
gcc -O3 -Wall -Wextra -Werror -shared -fPIC -o dma_fast.so dma_fast.c
gcc -O3 -Wall -Wextra -Werror -shared -fPIC -o quiet_dma.so quiet_dma.c native_timing.c

gcc -O2 -Wall -Wextra -Werror -shared -fPIC read_existing.c -o read_existing.so
