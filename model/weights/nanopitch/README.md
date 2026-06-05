# models/nanopitch/ — local-only backbone (gitignored)

BreathCoach attaches to a frozen NanoPitch pitch-tracking backbone. These
weights are proprietary and are NOT committed to the public repo, so this
directory is gitignored.

`nanobreath.config` auto-discovers this location, so with these two files in
place the project runs standalone — no env vars needed:
- `model.py`   — NanoPitch `nn.Module` definition (class `NanoPitch`)
- `best.pth`   — trained backbone checkpoint

To use a backbone elsewhere, set `NANOPITCH_SRC_DIR` and `NANOPITCH_CHECKPOINT`.
