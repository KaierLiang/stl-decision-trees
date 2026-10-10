# Bundled data

`naval_dataset.pkl` is the exact file supplied alongside the original scripts.
It contains a four-element list:

| Array | Shape | Description |
| --- | --- | --- |
| `X_train` | `(1600, 2, 61)` | Training trajectories |
| `y_train` | `(1600,)` | Original labels `-1` / `1` |
| `X_test` | `(400, 2, 61)` | Test trajectories |
| `y_test` | `(400,)` | Original labels `-1` / `1` |

Axes are sample, feature, and time. The loader maps `-1` to `0` without rescaling
or resampling signals. The original data generation source, channel units, and
any data-specific upstream terms are not documented in the supplied files.

SHA-256 of the supplied pickle:
`f7d8a3fddfe9f0a4b7131da3832e3b7777a338822cde40d1fa2ca39529e25d34`.

The model's optional augmentation is documented in [semantics](../docs/semantics.md).
It adds a binary channel and relabelled copies; those are not raw measurements.
