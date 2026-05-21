# models/

BreathCoach attaches to a frozen **NanoPitch** pitch-tracking backbone. That
backbone is a separate, Smule-confidential artifact and is **not** distributed
with this repository.

To run the project, place a NanoPitch-compatible backbone here:

```
models/nanopitch/
├── model.py     # defines a `NanoPitch` nn.Module (conv1, conv2, gru1-3, dense_vad, dense_pitch)
└── best.pth     # trained backbone weights
```

Everything under `models/nanopitch/` is gitignored, so these files stay local
and never get pushed to the public repo.

Once they're in place, the app finds them automatically — `config.py` resolves
the backbone in this order:

1. `$NANOPITCH_CHECKPOINT` / `$NANOPITCH_SRC_DIR` environment variables
2. `models/nanopitch/best.pth` and `models/nanopitch/model.py` (this folder)
3. otherwise the app errors with a helpful message

So with the two files above present, you can just run:

```bash
python -m nanobreath.deployment.serve
```

with no flags and no environment variables.

## Bring-your-own backbone

Any module exposing the same interface works. The wrapper in
`src/nanobreath/model/joint.py` expects these attributes on the backbone:
`conv1`, `conv2`, `gru1`, `gru2`, `gru3`, `dense_vad`, `dense_pitch`, and a
`gru_size` attribute. It produces a 384-d per-frame feature vector
(`conv2 + gru1 + gru2 + gru3`, each 96-d) that the BreathHead consumes.
