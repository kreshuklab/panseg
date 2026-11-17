# The receptive field and the prediction halo

PanSeg predicts large volumes patch by patch. Each patch is cut from the volume
with extra context on every side — the *halo* — because a convolutional network
can only produce a correct value for a voxel if it sees every input voxel that
voxel depends on. A halo that is too small leaves visible artefacts at patch
borders.

This page derives, from the architecture alone, how many voxels of context a
patch really needs. For the 4-level 3D U-Net used by the PanSeg models the
answer is: the value computed today, 44 voxels per side, is a lower bound, not
the receptive field. The receptive field reaches 51 voxels per side, and the
smallest halo that is exact for the usual aligned patch sizes is 48.

!!! note
    Everything below follows from the layer types and their hyperparameters.
    Neither the trained weights nor the input data play a role. The input
    *size* matters only through its remainder modulo the phase period `P`,
    which is discussed separately below.

## Notation

Every quantity is defined along one spatial axis (z, y or x). The *value*
column gives the number for the PanSeg 3D U-Net: 4 levels, 3×3 convolutions,
two convolutions per level.

| Symbol | Meaning | Value |
| --- | --- | --- |
| `n_lev` | Number of U-Net levels. | 4 |
| `ℓ` | Level index, from `0` (full resolution) to `n_lev - 1` (deepest). | 0 … 3 |
| `c` | Number of convolutions per level (a `DoubleConv`). | 2 |
| `k` | Convolution kernel size. | 3 |
| `r` | Convolution radius, `k // 2`. | 1 |
| `s_ℓ` | Stride of level `ℓ`, measured in input voxels: `2^ℓ`. | 1, 2, 4, 8 |
| `k_pool` | Kernel size of the encoder pooling. | 2 |
| `s_pool` | Stride of the encoder pooling; equals `k_pool` because `MaxPool` is created without an explicit stride. Each pooling cell reads the `k_pool` input cells below it, beginning with the one at its own index times `s_pool`. | 2 |
| `P` | Phase period along an axis: the total downsampling, `2^(n_lev - 1)`. For this architecture it is also the axis entry of `min_input`, because every pooling layer halves; it is the number of output positions after which the network's behaviour repeats. | 8 |
| `S` | Conv radius sum: every convolution's radius scaled by its level stride, `Σ r · s_ℓ` over all layers. This is the value `ModelGeometry.halo` returns today. | 44 |
| `Q` | Patch size along the axis: the part of the predicted volume that is kept. | problem-specific |
| `h` | Halo per side along the axis (`patch_halo`). | problem-specific |
| `M` | Model input size along the axis, `M = Q + 2h`: the length of the tensor that is actually fed to the network. | `Q + 2h` |
| `p` | Index of an output voxel inside the model input, `0 ≤ p < M`. | — |
| `φ` | Phase of an output voxel, `φ = p mod P`. It records whether the voxel sits on an even or an odd cell at each level. | 0 … 7 |
| `L(p)` | Left reach of output voxel `p`, in input voxels: the output depends on input positions `p - L(p) … p`. | 44 … 51 |
| `R(p)` | Right reach of output voxel `p`: the output depends on input positions `p … p + R(p)`. | 44 … 51 |
| support | The set of input positions whose value can influence a given output position. | — |
| clean | An output position is *clean* when its support lies strictly inside the model input, i.e. it neither contains the first nor the last input voxel. Only a clean output is free of the patch's artificial boundary. | — |

## The architecture

The encoder walks the levels from `ℓ = 0` to `ℓ = n_lev - 1`. Level 0 starts with
`c` convolutions of kernel `k` and padding `r`; every deeper level first applies
a `k_pool = 2` max pool with stride `s_pool = 2`, then the same two convolutions.
Every convolution preserves its input size.

The decoder walks back from level `n_lev - 2` to level 0. At each decoder level the
previous decoder output is upsampled by a factor of two with nearest-neighbour
interpolation to the size of the matching encoder feature map, concatenated with
that encoder feature map, and then passed through `c` convolutions. A final 1×1
convolution maps to the output channels and does not change the spatial layout.

## How the halo is computed today

`panseg.functionals.model_geometry.model_geometry` makes a single pass over the
modules. It tracks the level `ℓ`, multiplies `min_input` by the pooling kernel
when it meets a pool, decrements `ℓ` when it meets an upsampling layer, and adds

```
halo[axis] += 2**level * (kernel_size // 2)
```

for every convolution. In the notation above that is `S = Σ r · s_ℓ`:

```
encoder:  2 · (1 + 2 + 4 + 8) = 30
decoder:  2 · (4 + 2 + 1)     = 14
                                  = 44
```

This is the classic symmetric receptive-field radius. It is correct for a
network made only of stride-1 convolutions, and it is the *smallest* reach the
network ever has — but it is not the receptive field.

## Computing the real receptive field

The real reach follows from five propagation rules, applied in the order in
which the architecture applies the layers. Let `q` denote an output position at
the level currently being processed, which need not be the full-resolution
level. Each rule maps `q` to the set of level-0 input positions it depends on;
the support of `q` is the union over everything that feeds it.

| Layer | Support of output position `q` |
| --- | --- |
| convolution, kernel `k`, padding `r` | union of the supports of positions `q - r … q + r` |
| max/avg pool, kernel `k_pool`, stride `s_pool` | union of the supports of positions `s_pool · q … s_pool · q + k_pool - 1` |
| nearest upsampling from size `s_in` to size `s_out` | the support of source position `floor(q · s_in / s_out)` |
| concatenation with an encoder skip | union with the support of the encoder feature at the same position |
| 1×1 convolution | unchanged |

The rules are exact: they do not assume that the network is shift-invariant, so
they also capture the fact that pooling and nearest-neighbour upsampling are
not. The following function applies them along one axis:

```python
def reach_profile(model_input, n_levels=4):
    """Per-position (left, right) reach of a PanSeg UNet3D along one axis."""
    def conv(support, size):
        return {
            p: set().union(*(support.get(p + d, set()) for d in (-1, 0, 1)))
            for p in range(size)
        }

    def pool(support, size):
        return {
            j: support.get(2 * j, set()) | support.get(2 * j + 1, set())
            for j in range(size)
        }

    def upsample(support, src_size, dst_size):
        return {
            q: set(support.get((q * src_size) // dst_size, set()))
            for q in range(dst_size)
        }

    sizes = [model_input]
    for _ in range(n_levels - 1):
        sizes.append(sizes[-1] // 2)

    enc = {}
    for level, size in enumerate(sizes):
        support = (
            {i: {i} for i in range(size)}
            if level == 0
            else pool(enc[level - 1], size)
        )
        support = conv(support, size)
        support = conv(support, size)
        enc[level] = support

    dec = enc[n_levels - 1]
    for level in range(n_levels - 2, -1, -1):
        size = sizes[level]
        up = upsample(dec, sizes[level + 1], size)
        joined = {q: up[q] | enc[level].get(q, set()) for q in range(size)}
        joined = conv(joined, size)
        dec = conv(joined, size)

    return [(p - min(dec[p]), max(dec[p]) - p) for p in range(sizes[0])]
```

## The result

Evaluated in the interior of a model input whose size is a multiple of `P`, the
reach of this architecture is

```
L(φ) = S + ((φ + 2) mod P)
R(φ) = S + ((1 - φ) mod P)
```

Because `L` and `R` depend on `p` only through its phase, they are written as
functions of `φ` here. With `S = 44` and `P = 8`:

| `φ = p mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `L(p)` | 46 | 47 | 48 | 49 | 50 | 51 | 44 | 45 |
| `R(p)` | 45 | 44 | 51 | 50 | 49 | 48 | 47 | 46 |

So the reach is not one number:

* the **minimum** is `S = 44`, the value computed today;
* the **maximum** is `S + P - 1 = 51` per side;
* the total width `L(p) + R(p)` is 91 or 99 voxels, never `2 · S = 88`.

The encoder already shows the mechanism. Its reaches at the last level are
`L = 30` and `R = 37` in input voxels:

| `ℓ` | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| encoder `L` | 2 | 6 | 14 | 30 |
| encoder `R` | 2 | 7 | 17 | 37 |
| `s_ℓ` | 1 | 2 | 4 | 8 |

Three properties of the architecture cause the difference.

**Pooling is asymmetric.** A pooling cell at position `q` reads positions `2q`
and `2q + 1`, so it extends one full cell further to the right than to the
left. At level `ℓ` that is `s_ℓ` input voxels, and at the deepest level the
accumulated asymmetry is

```
Σ 2^ℓ for ℓ = 0 … n_lev - 2  =  P - 1  =  7
```

which is exactly `R - L = 37 - 30` in the encoder table.

**Nearest-neighbour upsampling is asymmetric in the opposite direction.** Output
position `q` takes the value of source position `floor(q / 2)`. For even `q`
the source is exactly below the output; for odd `q` it sits `s_ℓ` voxels to the
left, so the support of that position grows to the left and shrinks to the
right. Which of the two happens depends only on the parity of the position at
that level. The decoder contributes up to `P - 1` further voxels to one side,
and the sign is decided by the binary digits of `p` — hence the dependence on
`φ = p mod P`.

**Nearest-neighbour upsampling is also many-to-one.** Two adjacent output
positions share one source position, so a convolution neighbourhood of two
cells can reach a *new* source index in fewer than two source steps. This is
why the reach cannot be written as "convolution radius times stride" summed
over layers: that sum is only a lower bound. For an `M` that is a multiple of
`P` the upper bound `S + P - 1` follows from the two asymmetries above together
with the concatenated encoder skips, which prevent the left and right shifts
from cancelling; the next section shows that it grows further when `M` is not.

## The model input size matters

The reach above assumes that every pooling layer halves exactly, i.e. that `M`
is a multiple of `P`. When it is not, the encoder sizes are rounded down and the
decoder's `floor(q · s_in / s_out)` ratio is no longer 1/2, so the upsampling
stops being uniform and the reach grows. Sweeping `M`:

| `M mod P` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| worst reach per side | **51** | 52 | 53 | 54 | 55 | 56 | 57 | 58 |

Only `M mod P = 0` attains the minimum, and every `M` with that residue behaves
identically. The absolute size of the model input therefore cannot reduce the
receptive field — only the residue class can prevent it from growing further.

## How big must the halo be

The prediction is exact when every kept output voxel is clean. With the reach
from the previous section that means, for every `p` in the crop
`[h, h + Q - 1]`,

```
p - L(p) > 0        and        p + R(p) < M - 1
```

The inequalities are strict because the conv at the first or last input voxel
already reads zero padding. This is a per-position condition, not
`h ≥ max reach`. A position with reach 51 is harmless as long as it is at least
51 voxels away from the tensor edge, and inside a patch that is usually the
case.

For the well-behaved case `M ≡ 0 (mod P)` with `Q ≥ P`, substituting the reach
formula into the two ends of the crop gives

```
h > S + ((h + 2) mod P)                       crop start,  p = h
h > S + ((1 - ((h + Q - 1) mod P)) mod P)     crop end,    p = h + Q - 1
```

These are *not* two forms of the same condition: the crop end sits at the phase
`φ = (h + Q - 1) mod P`, so its bound also depends on the residue of `Q`.
Neither bound is monotone in `h`, because the residue terms change with the
residue class of `h`. For `Q ≡ 0 (mod P)`, where the end bound simplifies to
`h > S + ((2 - h) mod P)`:

| residue of `h` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop-start bound, `S + ((h + 2) mod 8)` | 46 | 47 | 48 | 49 | 50 | 51 | 44 | 45 |
| crop-end bound, `S + ((2 - h) mod 8)` | 46 | 45 | 44 | 51 | 50 | 49 | 48 | 47 |

For a halo that is a multiple of `P` both bounds agree and read `h > S + 2`, so
the smallest such halo is 48 (with `M ≡ 0 (mod P)` this case requires
`Q ≡ 0 (mod P)` as well). For every other residue the two bounds differ and have
to be solved together with the constraint on `M` below.

The two bounds, together with `M ≡ 0 (mod P)` and `Q ≥ P` — the latter so that
every phase actually occurs in the crop — are what the per-residue table below
enumerates. Evaluating the exact support propagation for a patch of each residue
class gives the smallest halo:

| `Q mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smallest valid `h` | **48** | **48** | **47** | **47** | **46** | 50 | 49 | 49 |
| `M mod 8` at that `h` | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| margin (left, right) | 2, 2 | 2, 3 | 2, 2 | 2, 3 | 2, 2 | 2, 3 | 2, 2 | 2, 3 |

The rows with `M ≡ 0 (mod P)` are the ones the two bounds above predict; the odd
patch sizes have an odd `M`, a slightly inflated reach, and one extra voxel of
margin on the right. In every case the minimum is `S + 2 = 46`, reached at
`Q ≡ 4 (mod 8)`. Patches shorter than `P` are not covered by the table: with
so few phases the crop no longer contains all of them and can be aligned to the
cheap ones instead (for example `Q = 4` with `h = 46`).

In particular:

* **48 is the minimum for `Q ≡ 0 (mod 8)`**, which covers every patch whose
  dimensions are multiples of 8. The other residues reach their minima only with
  halos that are not multiples of `P`; for `Q ≡ 4 (mod 8)` no multiple of `P`
  works at all, because `M = Q + 2h` keeps the residue 4. The zoo's
  `[40, 220, 220]` recommendation is such a case (220 ≡ 4 (mod 8)) and needs
  either the per-residue halo 46 or a patch rounded to 216 or 224;
* **44 is never enough.** For `Q = 64` the two outermost crop voxels on each
  side depend on input that lies 6 voxels outside the model input; the support
  is not merely at the border, it would have to extend past it;
* **46 is not automatically enough either.** With `Q = 64` it gives `M = 156`,
  which is not a multiple of `P`, so the reach grows and the crop is still not
  clean. With `Q = 60` the same `h = 46` gives `M = 152` and is exact.

For a guarantee that does not depend on the alignment, the bounds above do not
apply, because they assume `M ≡ 0 (mod P)`: the reach itself grows (to 58 per
side in the worst case) and the halo has to be re-derived. Applying the exact
support propagation to a fixed halo across every patch size gives **53** as the
smallest alignment-free halo for this architecture (52 still fails for
`Q ≡ 7 (mod 8)`). Rounded up to the next multiple of `P` that is **56**, the
value used below.

## Practical rule

For a model whose per-axis `min_input` is `P` and whose conv radius sum is `S`:

1. compute the reach table per phase from the architecture, as above, or use the
   closed form when it applies;
2. require `M % P == 0` for the model input, i.e. `patch + 2 · halo` is a
   multiple of `min_input` on every axis, otherwise the reach grows;
3. require `halo % P == 0` so that every crop starts at phase 0 of its model
   input. Together with 2 this forces `patch % P == 0`, and the smallest such
   halo is the first multiple of `P` that is at least `S + 2`, which is 48 here.
   Align the patch origins (i.e. the stride) to `P` as well: otherwise
   overlapping patches compute the same voxel at different phases, and a voxel
   that is exact in one patch can be inexact in the next. If the exact minimum
   matters, use the per-residue table above instead — it allows 46 or 47 for
   some patch sizes, at the cost of that phase consistency;
4. if the patch or the stride is not aligned to `P`, fall back to the
   alignment-free 53, rounded up to a multiple of `P` → 56.

For the 4-level 3D U-Net of the PanSeg models this yields

| Quantity | Value |
| --- | --- |
| `S` (today's `ModelGeometry.halo`) | 44 |
| worst-case reach per side, `S + P - 1` | 51 |
| smallest halo, aligned (`patch` a multiple of 8) | 48 |
| smallest alignment-free halo | 53 |
| safe halo without alignment | 56 |

A 2D U-Net does not pool along z, so the z entry of `P` is 1, the z reach has a
single phase, and the z halo is 0. The y and x axes behave exactly like the z
axis of the 3D model: 44 today, 48 when the patch dimension is a multiple of 8.

## Verification

The derivation was checked against the real `generic_confocal_3D_unet` on a
volume made of one z slice repeated 64 times. Every z slice of the input is
identical, so every predicted z slice must be identical too; the only way a
slice can differ is by seeing the artificial border of a patch. With a patch
depth and a halo satisfying the rules above the maximum deviation is at most
`1e-9`, while a model input that is not a multiple of `P` leaves deviations up
to `2 · 10^-4`:

| patch `Q` | halo `h` | model input `M` | identical slices (atol `1e-4`) |
| --- | --- | --- | --- |
| 64 | 44 | 152 | 60–63 / 64 |
| 64 | 46 | 156 | 60–62 / 64 |
| 60 | 46 | 152 | **64 / 64** |
| 62 | 49 | 160 | **64 / 64** |
| 64 | 48 | 160 | **64 / 64** |

## References

* `panseg.functionals.model_geometry` — the implementation that computes `S`
  today, and the source of the notation used here.
* For the classic receptive-field argument behind `S`, see
  <https://doi.org/10.6028/jres.126.009>.
* The z-slice invariance test in `tests/tasks/test_prediction_tasks.py`
  (`test_unet_prediction_task_z_slice_invariance`) is the regression test for
  this page.
* Tracking issues: [kreshuklab/panseg#637](https://github.com/kreshuklab/panseg/issues/637)
  (edge artefacts) and [#639](https://github.com/kreshuklab/panseg/issues/639)
  (halo too small).
