# Memory layout

Targets assume aggressive MCU limits (for example 32KB RAM, 128KB flash). `CodegenConfig` and `DispatcherConfig` encode stack, heap, and buffer sizes used when generating linker flags and HAL-oriented stubs.

Tune `max_stack_size`, `max_heap_size`, and `buffer_size` to match your SoC.
