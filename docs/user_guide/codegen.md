# Code generation

TinyRL emits CMake/Make projects, Rust `no_std` crates, Arduino libraries, and TVM-oriented placeholder artifacts. You still need **target toolchains** (for example `arm-none-eabi-gcc`) to produce binaries.

```bash
python codegen.py <model_path> <env_id>
```

See [Code generation API](../api/codegen.md) for `CodegenConfig` and `CodegenTrainer`. CI includes a smoke path when `arm-none-eabi-gcc` is available.
