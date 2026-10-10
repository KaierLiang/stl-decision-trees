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
redistribution terms were not supplied with this attachment and are not invented
here. Verify them before any public release.

SHA-256 of the supplied pickle:
`f7d8a3fddfe9f0a4b7131da3832e3b7777a338822cde40d1fa2ca39529e25d34`.

`ecg200_data.pickle` is referenced by `legacy/main_belong.py` but was not included.
The ECG preset is provided for use when the original file becomes available.

The model's optional augmentation is documented in [semantics](../docs/semantics.md).
It adds a binary channel and relabelled copies; those are not raw measurements.
