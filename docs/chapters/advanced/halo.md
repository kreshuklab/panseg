# The receptive field and the prediction halo

PanSeg predicts large volumes patch by patch. Each patch is cut with extra
context on every side — the *halo* — because a convolutional network can only
produce a correct value for a voxel if it sees every input voxel that voxel
depends on. Too small a halo leaves visible artefacts at patch borders.

This page derives from the architecture alone how much context a patch needs.
For the 4-level 3D U-Net of the PanSeg models the classic symmetric radius 44 is
a lower bound, not the receptive field: the reach is 44–51 voxels per side
depending on the position's phase, the smallest halo that is exact for aligned
patches is 48, and the smallest input with an artefact-free interior is 104.

!!! note
    Everything follows from the layer types and their hyperparameters, not from
    the weights or the data. The input *size* matters only through its remainder
    modulo the phase period `P`.

## Notation

Every quantity is defined along one spatial axis (z, y or x). The *value*
column gives the number for the PanSeg 3D U-Net: 4 levels, 3×3 convolutions,
two convolutions per level.

| Symbol | Meaning | Value |
| --- | --- | --- |
| `n_lev` | Number of U-Net levels. | 4 |
| `ℓ` | Level index, from `0` (full resolution) to `n_lev - 1` (deepest). | 0 … 3 |
| `c`, `k`, `r` | Convolutions per level (a `DoubleConv`); kernel size; radius `k // 2`. | 2; 3; 1 |
| `s_ℓ` | Stride of level `ℓ`, measured in input voxels: `2^ℓ`. | 1, 2, 4, 8 |
| `k_pool`, `s_pool` | Encoder pooling kernel; its stride, which equals `k_pool` because `MaxPool` is created without an explicit stride. A pooling cell reads the `k_pool` input cells below it, beginning at its own index times `s_pool`. | 2; 2 |
| `P` | Phase period along an axis: the total downsampling, `2^(n_lev - 1)`. For this architecture it is also the axis entry of `min_input`, because every pooling layer halves. It is the number of output positions after which the network's behaviour repeats. | 8 |
| `S` | Conv radius sum: every convolution's radius scaled by its level stride, `Σ r · s_ℓ` over all layers. The classic lower bound on each side's reach. | 44 |
| `Q`, `h`, `M` | Patch size; halo per side (`patch_halo`); model input `M = Q + 2h`, the length of the tensor fed to the network. Only `Q` is kept in the prediction. | problem-specific; `Q + 2h` |
| `p`, `φ` | Index of an output voxel inside the model input, `0 ≤ p < M`; its phase `φ = p mod P`, which records whether the voxel sits on an even or an odd cell at each level. | —; 0 … 7 |
| `L(p)`, `R(p)` | Left and right reach of output voxel `p`, in input voxels: the output depends on input positions `p - L(p) … p` and `p … p + R(p)`. | 44 … 51 |
| clean | An output is *clean* when its support — the set of input positions that influence it — lies strictly inside the model input, i.e. it contains neither the first nor the last input voxel. Only a clean output is free of the patch's artificial boundary. | — |

## The architecture

Level 0 applies `c` size-preserving convolutions of kernel `k` and padding `r`.
Every deeper encoder level first max-pools with `k_pool = 2` and `s_pool = 2`,
then applies the same two convolutions. The decoder walks back from level
`n_lev - 2`: it nearest-neighbour-upsamples the previous decoder output by two
to the matching encoder size, concatenates that encoder feature map, and applies
`c` convolutions. A final 1×1 convolution changes channels, not layout.

## The classic halo bound

The naive bound scales every convolution's radius by its level stride and sums,
`S = Σ r · s_ℓ`:

```
encoder:  2 · (1 + 2 + 4 + 8) = 30
decoder:  2 · (4 + 2 + 1)     = 14
                                  = 44
```

That is correct for a stack of stride-1 convolutions and is the *smallest* reach
this network ever has, but it is not the receptive field: 44 is never enough.

## Computing the exact reach

The real reach follows from five propagation rules, applied in the order the
architecture applies them. Let `q` be an output position at the level being
processed, not necessarily full resolution. Each rule maps `q` to the level-0
input positions it depends on; its support is the union over everything feeding
it.

| Layer | Support of output position `q` |
| --- | --- |
| convolution, kernel `k`, padding `r` | union of the supports of positions `q - r … q + r` |
| max/avg pool, kernel `k_pool`, stride `s_pool` | union of the supports of positions `s_pool · q … s_pool · q + k_pool - 1` |
| nearest upsampling from size `s_in` to size `s_out` | the support of source position `floor(q · s_in / s_out)` |
| concatenation with an encoder skip | union with the support of the encoder feature at the same position |
| 1×1 convolution | unchanged |

The rules are exact: they do not assume shift invariance, so they capture
pooling and nearest-neighbour upsampling. They cover size-preserving
convolutions only (odd kernel `k`, `padding = k // 2`, stride and dilation 1,
zero padding), plus pooling without padding, with unit dilation and in floor
mode. Convolutions and pooling outside these rules raise `NotImplementedError`
instead of being approximated; so does an unrecognised module that exposes a
kernel, stride, padding, output size, scale or resize factor, or an unrecognised
container — fail closed rather than under-approximate the reach.
`panseg.functionals.model_geometry.layers` maps model modules to this vocabulary
(`Encoder`, `Decoder`, `InterpolateUpsampling`, convolutions, pooling,
containers) and `.replay` applies the rules per axis and per phase; that replay
is the authoritative source, not a re-derivation for one specific depth.

## The result

In the interior of a model input whose size is a multiple of `P`:

```
L(φ) = S + ((φ + 2) mod P)
R(φ) = S + ((1 - φ) mod P)
```

With `S = 44` and `P = 8`:

| `φ = p mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `L(p)` | 46 | 47 | 48 | 49 | 50 | 51 | 44 | 45 |
| `R(p)` | 45 | 44 | 51 | 50 | 49 | 48 | 47 | 46 |

* the **minimum** is `S = 44`, the classic symmetric radius;
* the **maximum** is `S + P - 1 = 51` per side;
* the total width `L(p) + R(p)` is 91 or 99 voxels, never `2 · S = 88`.

Pooling reaches one cell further right per level, `P - 1 = 7` voxels in total
(the encoder ends at `L = 30`, `R = 37`). Nearest-neighbour upsampling shifts
the support the other way for odd positions, and because adjacent outputs share a
source the reach is not a radius-times-stride sum; the shift's direction depends
on the binary digits of `p`, and the encoder skips stop the shifts from
cancelling, so the aligned reach is `S + P - 1`.

## The model input size matters

The reach above assumes every pooling layer halves exactly, i.e. `M` is a
multiple of `P`. Otherwise encoder sizes are rounded down and the decoder's
`floor(q · s_in / s_out)` is no longer 1/2, so the upsampling is not uniform and
the reach grows:

| `M mod P` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| worst reach per side | **51** | 52 | 53 | 54 | 55 | 56 | 57 | 58 |

Only `M mod P = 0` attains the minimum; the absolute size cannot reduce the
receptive field, only the residue class prevents it from growing.

## How big must the halo be

The prediction is exact when every kept output voxel is clean. For every `p` in
the crop `[h, h + Q - 1]` that means

```
p - L(p) > 0        and        p + R(p) < M - 1
```

The inequalities are strict, because the conv at the first or last input voxel
already reads zero padding. This is a per-position condition, not
`h ≥ max reach`: a position with reach 51 is harmless as long as it is 51
voxels from the tensor edge.

For `M ≡ 0 (mod P)` and `Q ≥ P`, substituting the reach into the two crop ends
gives

```
h > S + ((h + 2) mod P)                       crop start,  p = h
h > S + ((1 - ((h + Q - 1) mod P)) mod P)     crop end,    p = h + Q - 1
```

The end sits at phase `φ = (h + Q - 1) mod P`, so its bound also depends on the
residue of `Q`, and neither bound is monotone in `h`. For `h` a multiple of `P`
both agree and read `h > S + 2`, so the smallest such halo is 48 (with
`M ≡ 0 (mod P)` this also requires `Q ≡ 0 (mod P)`). Other residues differ and
must be solved together with the constraint on `M`; the exact propagation gives:

| `Q mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smallest valid `h` | **48** | **48** | **47** | **47** | **46** | 50 | 49 | 49 |
| `M mod 8` at that `h` | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| margin (left, right) | 2, 2 | 2, 3 | 2, 2 | 2, 3 | 2, 2 | 2, 3 | 2, 2 | 2, 3 |

Rows with `M ≡ 0 (mod P)` are the ones the bounds predict; odd patch sizes have
an odd `M`, a slightly inflated reach and one extra voxel of margin on the right.
The global minimum is `S + 2 = 46`, at `Q ≡ 4 (mod 8)`. Patches shorter than `P`
are not covered: with so few phases the crop need not contain all of them, so it
can be aligned to the cheap ones (e.g. `Q = 4`, `h = 46`).

* **48 is the minimum for `Q ≡ 0 (mod 8)`**, which covers every patch whose
  dimensions are multiples of 8. The other residues reach their minima only with
  halos that are not multiples of `P`; for `Q ≡ 4 (mod 8)` no multiple of `P`
  works, because `M = Q + 2h` keeps the residue 4. The zoo's `[40, 220, 220]`
  recommendation is such a case (220 ≡ 4 (mod 8)) and needs either the
  per-residue halo 46 or a patch rounded to 216 or 224.
* **44 is never enough.** For `Q = 64` the two outermost crop voxels on each
  side depend on input 6 voxels outside the model input: the support would have
  to extend past the border, not merely touch it.
* **46 is not automatically enough either.** With `Q = 64` it gives `M = 156`,
  not a multiple of `P`, so the reach grows and the crop is still not clean; with
  `Q = 60` the same `h = 46` gives `M = 152` and is exact.

For a guarantee independent of alignment the bounds above do not apply: the
reach itself grows (to 58 per side) and the halo must be re-derived. Applying
the exact propagation to a fixed halo across every patch size gives **53** as
the smallest alignment-free halo (52 still fails for `Q ≡ 7 (mod 8)`), **56**
rounded up to a multiple of `P`. PanSeg does not use it: it aligns the domain, so
the declared BioImage.IO step is the contract.

## Practical rule

For a model whose per-axis phase period is `P` — the product of the pooling
strides, which the package calls `increment` — and whose conv radius sum is `S`,
compute the reach table per phase from the architecture, as above, or use the
closed form when it applies. Then:

1. pad the raw volume up to a multiple of `P` on every spatial axis, so the
   patch, the stride and `patch + 2 · halo` are all multiples of `P`. Otherwise
   the reach grows (`M mod P != 0` above) and overlapping patches compute the
   same voxel at different phases;
2. apply the halo on **every** spatial axis, including axes that fit in a single
   patch — the halo surrounds the model input, not just the tiled patch — and
   use the smallest multiple of `P` whose aligned crop is clean.
   Together with 1 this forces `patch % P == 0`, and for the standard net the
   smallest such halo is 48. If the exact minimum matters and the phase may vary
   between patches, use the per-residue table instead — it allows 46 or 47 for
   some patch sizes, at the cost of that phase consistency;
3. manual patch/halo overrides are used verbatim: no rounding, no alignment
   check, no clamping. Artefacts and inefficiency then are the caller's choice.

A 2D U-Net does not pool along z, so the z entry of `P` is 1, the z reach has a
single phase, and the z halo is 0. The y and x axes behave exactly like the z
axis of the 3D model: 44 as the classic radius, 48 when the patch dimension is a
multiple of 8.

!!! note
    `GroupNorm` couples all spatial positions of a feature map, so for models
    with `gcr` layers "exact" means free of *convolution-border* artefacts: the
    support argument bounds what a convolution can see, not the normalization
    statistics. The z-slice test below is unaffected because its data is
    constant along z.

## Verification

The z-slice invariance test builds a volume from one z slice repeated 64 times,
so every predicted z slice must be identical too; a slice can only differ by
seeing a patch's artificial border. With a patch depth and halo satisfying the
rules above the measured maximum deviation is at most `6e-8` (float32 noise) and
the test, which asserts equality within `1e-7`, passes; a model input that is
not a multiple of `P` leaves deviations up to `2 · 10^-4`:

| patch `Q` | halo `h` | model input `M` | identical slices (within `1e-4`) |
| --- | --- | --- | --- |
| 64 | 44 | 152 | 60–63 / 64 |
| 64 | 46 | 156 | 60–62 / 64 |
| 60 | 46 | 152 | **64 / 64** |
| 62 | 49 | 160 | **64 / 64** |
| 64 | 48 | 160 | **64 / 64** |

## The contract

Architecture-derived geometry is resolved in exactly one place,
`panseg.functionals.model_geometry`; its facade docstring defines the four
quantities this page explains:

* `min_input` — the smallest shape each axis accepts, the product of the pooling
  kernels that downsample it;
* `increment` — the phase period `P`, the product of the pooling strides; derived
  patches, strides and the minimal halo are multiples of it;
* `halo` — the smallest multiple of `increment` whose aligned crop is clean (48
  for the standard 3D net; 0 on an axis the model does not downsample);
* `effective_min_input` — `max(min_input, 2 · halo + 1)` rounded up to
  `increment`, the smallest tile that runs and still leaves a clean interior
  (104 for the standard 3D net).

Prediction reflect-pads the raw volume up to a multiple of `increment` on every
spatial axis, then by default tiles it with an aligned patch and stride, and
crops the result back to the raw shape. Defaults come from the package; manual
patch and halo overrides are used verbatim. The regression test for this
contract is
`tests/tasks/test_prediction_tasks.py::test_unet_prediction_task_z_slice_invariance`.
For the classic receptive-field argument behind `S`, see
<https://doi.org/10.6028/jres.126.009>.
