# Rust integration

Codegen can emit a `no_std` crate layout. Cross-compile with a target that matches `CodegenConfig.rust_target`, for example:

```bash
cargo build --release --target thumbv8m.main-none-eabihf
```

You must install the corresponding Rust target and have a suitable linker/flash workflow for your hardware.

See [Code generation](../user_guide/codegen.md).
