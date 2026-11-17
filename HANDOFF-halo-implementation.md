# Handoff: minimal, exact prediction halo

Status: ready to implement. This document is a handoff, not part of the docs
site. The theory is already written up in
`docs/chapters/advanced/halo.md` (untracked in the worktree
`panseg-z-slice-invariance` at the time of writing — check whether that session
committed it, and copy it into your worktree if needed). Read that page first;
this document does not repeat the derivation, only the decisions, the target
behaviour, and the work plan.

Tracking: [kreshuklab/panseg#639](https://github.com/kreshuklab/panseg/issues/639)
(halo too small), parent [#637](https://github.com/kreshuklab/panseg/issues/637)
(edge artefacts), sibling [#638](https://github.com/kreshuklab/panseg/issues/638)
(auto halo 0 for small images).

## Mission

Replace the current "conv radius sum" halo (`44` for the standard UNet3D) with
the **minimal halo that makes tiled prediction exact**, derived from the
architecture, and constrain the patch/stride calculation so that minimal value
is actually valid. Document the resulting size increments in the model
description.

Definition of done: with the default (automatic) settings, the slow
z-slice-invariance test predicts **64/64 identical slices with max deviation
0.0**, at a smaller halo than the alignment-free fallback (56), and the
BioImage.IO description declares a size step equal to the per-axis increment.

## What is wrong (one paragraph)

`panseg/functionals/model_geometry.py::model_geometry` adds
`2**level * (kernel_size // 2)` per convolution. That is the classic symmetric
receptive-field radius, `S = 44`. The real reach is position-dependent because
`MaxPool(k=2, s=2)` shifts the support right by one cell per level and
nearest-neighbour upsampling shifts it left or right depending on the parity of
the output index at each level. With `P = 2**(levels-1) = 8`:

```
L(phi) = S + ((phi + 2) mod P)      44 … 51
R(phi) = S + ((1 - phi) mod P)      44 … 51
```

So 44 is the *minimum* reach, 51 is the worst case, and the halo requirement is
a per-position condition, not `h >= max reach`. See the doc for the numbers and
the reference implementation (`reach_profile`).

## Decisions already made (do not re-litigate)

1. **Alignment is required, not optional.** The model input, the patch, the
   stride and the halo are all multiples of the per-axis phase period
   `P = increment = min_input[axis]` (8 for the standard 3D net, 1 in z for 2D).
   Rationale: `M = patch + 2*halo` must be a multiple of `P` (otherwise the
   encoder does not halve exactly and the reach grows to 52–58), and the crop
   phase must match the untiled run across overlapping patches
   (`halo ≡ patch_origin (mod P)`, with origins multiples of the stride), so
   `halo % P == 0` and `stride % P == 0`.
2. **Minimal halo, computed, not hard-coded.** The result for the standard
   4-level UNet3D is **48** (patch a multiple of 8, margin 2 voxels on each
   side). Do not hard-code 48: compute it from the reach table so anisotropic
   pools/kernels, other depths and 2D models work. For the standard net the
   table from the doc is:

   | `Q mod 8` | 0 | 2 | 4 | 6 |
   | --- | --- | --- | --- | --- |
   | smallest exact `h` | **48** | 47 | 46 | 49 |

   These are the exact minima for a given patch residue, but the ones with
   `h % 8 != 0` break phase consistency between overlapping patches. **The
   supported/automatic configuration is `patch ≡ 0 (mod P)`, `halo % P == 0`,
   which yields 48.**
3. **Never silently drop the halo to 0** when the volume fits one patch (#638).
   Pad the volume and keep the halo; if the padded model input does not fit,
   shrink the patch (tile) instead.
4. **`halo` keeps a single public meaning: the minimal exact halo for an aligned
   input.** The old value 44 becomes an internal quantity (call it `context`
   or `S`), not the public halo.
5. **Increment is documented per axis in `ParameterizedSize`.** Input axes
   declare `step = increment`; that is what makes the minimal halo safe for
   external consumers (they can only pick input sizes ≡ 0 mod `P`).

## Target rules

Per axis (z, y, x), with `S = context`, `P = increment`, `Q = patch`,
`h = halo`, `M = Q + 2h`:

- **R1** `M % P == 0`, `Q % P == 0`, `h % P == 0`, `stride % P == 0`.
- **R2** Every kept output position `p ∈ [h, h+Q-1]` is *clean*: its support
  lies strictly inside the model input, i.e. `p - L(p) > 0` and
  `p + R(p) < M - 1` (strict, per the doc; the first/last input voxel's conv
  already reads the zero padding).
- **R3** The halo is the smallest multiple of `P` satisfying R2 (48 for the
  standard net). If the caller passes a larger halo, honour it.
- **R4** An input shape that violates R1 is rejected with a message naming the
  axis, the offending size and the required increment — do not silently round
  user-supplied sizes; round only in the *derivation* path
  (`derive_patch_and_halo_shapes`, `find_feasible_patch_and_halo_shapes`).
- **R5** The alignment-free fallback (`S + P - 1`, rounded up to a multiple of
  `P` → 56) is available only for callers that explicitly opt out of alignment
  (e.g. a user-supplied patch). It must not be the default.

Expected numbers for `generic_confocal_3D_unet`:

| quantity | today | target |
| --- | --- | --- |
| `ModelGeometry` context / old `halo` | 44 | 44 (kept as `context`) |
| public `halo` / default `patch_halo` | 44 | **48** |
| `effective_min_input` (input minimum) | 89 | rounded up to a multiple of 8 → **104** |
| BioImage.IO input `ParameterizedSize` | `(min=89, step=1)` | `(min=104, step=8)` |
| BioImage.IO output `halo` | 44 | **48** |
| alignment-free fallback | — | 56 |

## Implementation plan

Work on a fresh worktree per `docs/agents/conventions.md` (symlink `AGENTS.md`,
`docs/agents/`, copy the theory doc). Suggested order:

### 1. `panseg/functionals/model_geometry.py` (core, test-first)

- Split the single traversal into two steps, keeping the "resolved in exactly
  one place" principle:
  1. `_layer_sequence(model)` → an ordered list of spatial layers with per-axis
     attributes (conv radius, pool kernel/stride, upsample). Reuse the existing
     module inspection (`_spatial_layer`, `InterpolateUpsampling`), but keep the
     per-axis data instead of collapsing to one number. `nn.MaxPool3d` without
     an explicit stride means `stride = kernel`; assert that assumption or read
     `.stride`.
  2. `_reach_table(layers, P)` → per axis, per phase `φ ∈ [0, P)`, the pair
     `(L, R)`. Replay the propagation rules from the doc on supports for a
     model input of `P * 32` and read the interior; `P ≤ 16` in practice, so
     this is cheap and exact. Do **not** assume two convs per level or `k = 3`.
- `ModelGeometry` fields: keep `min_input`, add `increment` (per-axis `P`),
  `context` (per-axis `S`), and the per-phase reach table. Provide:
  - `minimal_halo(patch) -> tuple[int, int, int]` — smallest multiple of
    `increment` satisfying R2 for `patch` (R3).
  - `max_halo(patch) -> ...` or `fallback_halo` — `S + P - 1` rounded up (R5).
  - `valid_input(shape) -> bool` / `check_patch(patch, halo)` — extend to
    enforce R1 **and** R2, with axis-specific messages (R4).
  - Keep `model_input_shape`, `tensor_input_shape`, `is_2d`.
- Decide whether the public `halo` attribute disappears or becomes a method. Do
  not keep an attribute named `halo` that means "44": it is the main source of
  the confusion. Grep every use before renaming (`panseg/`, `tests/`).
- `effective_min_input`: make it a *valid* size, i.e.
  `round_up_to_increment(max(min_input, 2*halo + 1))` per axis (104 for the
  standard net), or add a separate `declared_min_input` if rounding here is too
  surprising. The BioImage.IO path and `pad_to_effective_minimum` both need a
  size that satisfies the declared step.

### 2. Patch / halo derivation

- `panseg/functionals/prediction/utils/size_finder.py`
  - `derive_patch_and_halo_shapes`: round the patch down to a multiple of
    `increment` (but never below `effective_min_input`), and return the
    minimal halo from the geometry instead of `(0,0,0)`. If the volume fits,
    still return the volume shape and a **nonzero** halo (#638); if the padded
    model input exceeds the budget, shrink the patch rather than the halo.
  - `find_feasible_patch_and_halo_shapes`: keep the OOM probe, but probe only
    shapes that satisfy R1 (round before probing), and never return a shape
    that fails R1/R2.
- `panseg/functionals/prediction/prediction.py`
  - `patch_halo is None` → `geometry.minimal_halo(patch)`; when `patch` is also
    `None`, derive both together (the current code resolves the halo before the
    patch, so this ordering needs care).
  - `_check_patch_axes` / `geometry.check_patch`: add the increment check.
  - `_warn_inefficient_auto_patch`: keep, it is now more relevant.
- `panseg/functionals/prediction/utils/utils.py::get_stride_shape`: round the
  stride to a multiple of `increment` (e.g. floor to the grid, minimum one
  increment) so overlapping patches share the output phase. `SliceBuilder` then
  inherits aligned starts; assert it in `ArrayDataset`/`ArrayPredictor` rather
  than rounding silently there.
- Remove or rework `get_patch_halo` (`utils/utils.py`), which reads a static
  template value.

### 3. Model description (`ParameterizedSize`)

`panseg/functionals/training/biio.py`:

- `_input_axis(...)` currently declares `ParameterizedSize(min=minimum, step=1)`
  (line ~237). Change to `step=increment[axis]` and make `minimum` a valid size
  under that step (see `effective_min_input` above). `ParameterizedSize` has
  exactly `min` and `step`, so this is the intended place to document the
  supported increments.
- `_output_axis(...)`: pass the new minimal halo (48 for the standard net).
  This is only sound for consumers because the input axes now declare
  `step = P`; state that in the docstring.
- `make_model_description`: update the docstring (it currently speaks of
  `geometry.halo`/`effective_min_input` as if 44/89 were final).
- `pad_to_effective_minimum`: pads to `tensor_effective_min_input()`, which must
  now be step-valid; keep the packaged `test_in`/`test_out` consistent
  (the existing tests assert this).
- `panseg/resources/models_zoo.yaml` / zoo records: `recommended_patch_size`
  values are already multiples of 8 ([80, 160, 160], [1, 256, 256]). Optionally
  add a `patch_size_increment` field next to `recommended_patch_size`; treat it
  as a follow-up if it widens the change too much.

### 4. Viewer

`panseg/viewer_napari/widgets/prediction.py`: `update_halo` should show the
minimal halo for the current patch and the current model, and disable/adjust
the halo spinbox consistently with the increment (the widget already has
mode-dependent enable/disable logic). `patch_halo` default changes from 44 to
48. Do not silently round user input in the widget without telling them.

## Tests

- `tests/functionals/test_model_geometry.py`: replace the 44 expectations with
  the new API; assert the reach table from the doc (exact numbers) for the
  standard net, the anisotropic conv test, 2D (z increment 1, z halo 0),
  `minimal_halo` == 48 for patch multiples of 8, and the per-residue minima
  (47/46/49) if you expose them. Keep/adapt the `effective_min_input` tests.
- New unit tests for R1/R2: an invalid model input (e.g. `M = 156`) raises;
  a clean and an unclean `(patch, halo)` pair are classified correctly.
- `tests/core/test_zoo.py`: `(44,44,44)` / `(0,44,44)` → the new halo.
- `tests/widgets/test_prediction.py`: 44 → 48.
- `tests/functionals/training/test_biio.py`: the halo literals are synthetic
  `ModelGeometry` fixtures; update only where they encode the old public
  meaning. Add an assertion that the declared input axes have
  `step == increment` and that `min` is a multiple of `step`.
- `tests/functionals/prediction/test_size_finder.py`: add "every derived
  patch/halo satisfies R1" and "a volume that fits one patch still gets a
  nonzero halo" (#638).
- `tests/tasks/test_prediction_tasks.py::test_unet_prediction_task_z_slice_invariance`:
  must pass with the defaults. Also parametrize the empirically verified rows
  as fast(er) regression coverage if feasible:

  | patch | halo | model input | expected |
  | --- | --- | --- | --- |
  | 64 | 48 | 160 | 64/64, dev 0.0 |
  | 60 | 46 | 152 | 64/64, dev 0.0 |
  | 62 | 49 | 160 | 64/64, dev ≈ 1e-9 |
  | 62 | 47 | 156 | not clean (M not a multiple of 8) |
  | 64 | 44 | 152 | not clean |

  The last two rows are the negative controls; keep them as explicit
  expectations if you can keep the runtime reasonable (`@pytest.mark.slow`).
- Add a guard test for the *reference* implementation: the doc's
  `reach_profile` and the production traversal must agree on the phase table
  for the standard net and for at least one anisotropic architecture.

## Verification

Per `docs/agents/conventions.md`:

```bash
/home/kriedmiller/software/miniforge3/envs/panseg-fork/bin/python -c "import panseg; print(panseg.__file__)"   # must be your worktree
make test
make test ARGS="-m slow tests/tasks/test_prediction_tasks.py::test_unet_prediction_task_z_slice_invariance"
# widget tests need Xvfb: make test ARGS="tests/widgets/test_prediction.py"
```

Lint with the repo's ruff configuration (pre-commit: `--select I` +
`ruff-format`); no typechecker is configured, gate on no *new* pyright
diagnostics. Do not run a docs build from a worktree unless needed; if you do,
delete `site/` afterwards.

## Acceptance criteria

1. `model_geometry(UNet3D(...)).minimal_halo((64,128,128)) == (48,48,48)` (or the
   equivalent new API), and the 2D case gives z halo 0.
2. The default `unet_prediction` path derives an aligned patch, a minimal halo
   and an aligned stride without user input, and the slow z-invariance test
   passes with max deviation 0.0.
3. Any path that would produce `M % P != 0` raises a message naming the axis,
   the size, and the required increment.
4. A volume that fits one patch still gets mirror padding and a nonzero halo
   (#638).
5. `make_model_description` declares `step = increment` on input axes and the
   minimal halo on output axes; packaged test tensors satisfy the declared
   sizes.
6. No reference to the old `44` public halo remains in user-facing strings,
   widget defaults or the BioImage.IO description.

## Out of scope / do not do

- Do not re-derive the theory; use `docs/chapters/advanced/halo.md`.
- Do not make the halo data-dependent. It is architecture-only.
- Do not change training. Training crops have no halo; the only training-side
  output is the model description.
- Do not change `S = 44` handling for existing saved weights: geometry is
  derived from the architecture, not from checkpoints.
- Do not implement ConvTranspose/deconv support "while you are there"; the
  current architectures use nearest interpolation. Add a clear `NotImplemented`
  path if an unknown spatial layer appears.

## Open questions for the next session

1. **API shape.** Should `ModelGeometry` keep a `halo` attribute (as the minimal
   halo) or replace it with `context` + `minimal_halo(patch)`? The latter is
   clearer but touches many call sites and the BioImage.IO description.
2. **Where to enforce the increment.** Validating in `check_patch` is one
   place; rounding in `derive_patch_and_halo_shapes` is another. Decide who
   rounds (derivation) and who rejects (validation) and keep it to one place
   per concern.
3. **`effective_min_input` rounding.** Round it up to a multiple of the
   increment (104), or add a separate "declared minimum" for the description so
   the internal "smallest runnable input" keeps its current meaning (89)?
4. **Output halo semantics.** 48 is only safe for consumers because the input
   axes declare `step = 8`. Confirm that `bioimageio.spec` consumers honour
   `ParameterizedSize` on the *input* axis when they tile, and decide whether
   the description should otherwise declare the fallback 56.
5. **Zoo metadata.** Add `patch_size_increment` to the zoo records, or is the
   BioImage.IO `ParameterizedSize` enough?
6. **Odd patch sizes.** The per-residue minima include odd patches (doc table).
   Should `minimal_halo` support them (returning a non-aligned halo), or reject
   them with a clear message? Recommended: reject for now, keep the aligned
   path only.
7. **Performance.** Measure the model-input growth for the recommended patches
   (44 → 48 adds 8 voxels per axis, ~15 % more voxels for 80³) and document it.
   If the fallback 56 is ever used, it is ~50 % more.
8. **Small volumes and mirror padding.** `mirror_pad` uses
   `np.pad(mode="reflect")`. Check what happens when an axis is shorter than
   the halo (single-slice volumes, #638). If reflect cannot pad that far,
   decide between repeated reflection, edge padding, and refusing the input —
   but do not silently fall back to halo 0.
