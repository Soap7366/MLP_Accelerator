# PS / DMA protocol v1

The selected dual-clock configuration uses PS FCLK0=100 MHz for DMA and FCLK1 for NPU computation. AXIS asynchronous FIFOs convert between domains; a two-stage XPM synchronizer carries the GPIO arm level. A rising edge on the synchronized `arm` arms exactly one command. While busy extra arm edges are ignored. Reset cancels all in-flight operations. The GPIO must be returned low before a new request; PS software keeps it high until the response completes, allowing the synchronizer to observe it reliably.

PS sequence: allocate aligned buffers, prepare S2MM receive, pulse arm from 0 to 1, send MM2S packet, wait for send/receive completion, validate response. Return arm to 0 before the next request. DMA registers use AXI-Lite through PS GP0; DMA memory channels access DDR through HP0.

Request: 129 little-endian 32-bit beats. Beat zero is `0x4d4c5000 | mode`, where mode=0 inference or 1 feature dump. Following 128 beats contain 512 signed INT8 values, first value in bits 7:0. All TKEEP=15; TLAST only on beat 128. An invalid request is drained to TLAST and returns an error response without running inference.

Response header: three 32-bit beats. Word 0 = `0x4d4c0000 | (error<<2) | (mode<<1) | prediction`; word 1 = compute cycles from accepted core start through final feature write; word 2 = number of meaningful payload bytes. Error response is header only. Successful inference payload contains two signed logits, padded to one 32-bit beat (16 bytes total). Feature payload contains 128 FC1, 32 FC2 and 2 FC3 INT8 values, padded by two zero bytes (176 bytes total). All response TKEEP=15; TLAST marks the final word. Padding is not a feature. Prediction ties select 0.

Measure PS end-to-end elapsed time separately from core cycle count. Never label DMA/software latency as core execution time. No model parameters or floating point scales are sent per sample.

The optimized build disables both DMA data realignment engines, uses 64-bit memory ports matching HP0, and uses a 10-bit transfer-length field. Addresses must be aligned to eight bytes. Requests are still 516 bytes, inference responses 16 bytes and feature responses 176 bytes, all within the 1023-byte limit. The batch input buffer uses a 520-byte row stride: the four padding bytes after each request are not transmitted. A batch contains separate DMA transactions; it is never submitted as one transfer exceeding the limit.

PS timing scopes: `run_experiment.py` measures an individual Python/API call including cache synchronization. `run_batch.py` prepares 42 packets, synchronizes the buffers once per mode, and runs 42 separately armed DMA requests in a native loop. Its per-sample transaction time excludes batch cache synchronization, while batch wall time includes synchronization and the Python/native boundary. Neither timing scope includes input preparation or output validation. Report the scopes explicitly.

## Final read-only extension
A single 32-bit 0x4d4c5001 word (TKEEP=15, TLAST=1) reads retained PL features without input writes or core start. The previous 516-byte requests remain supported. Before any completed inference since reset, the command returns a 12-byte error header with zero cycles/payload. Successful reads return 176 bytes and the PREVIOUS inference cycle count. Only the last completed sample is retained. The last_cycles register now persists across commands; error responses report zero without discarding that stored count.
