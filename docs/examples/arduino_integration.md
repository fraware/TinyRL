# Arduino integration

1. Train and compress a policy (see [Quick Start](../quickstart.md)).
2. Run `codegen.py` with options that match your board (for example `CodegenConfig.arduino_board`).
3. Open the generated library layout in the Arduino IDE or `arduino-cli`, and build with the correct core for your MCU.

See [Code generation](../user_guide/codegen.md) and the [codegen API](../api/codegen.md).
