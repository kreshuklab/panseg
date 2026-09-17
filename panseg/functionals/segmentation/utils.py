import numpy as np
from elf.segmentation import compute_boundary_mean_and_length
from elf.segmentation.multicut import transform_probabilities_to_costs


def shift_affinities(affinities, offsets):
    rolled_affs = []
    for i, _ in enumerate(offsets):
        offset = offsets[i]
        shifts = tuple([int(off / 2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf

        padded_inverted_affs = np.pad(
            affinities, pad_width=((0, 0),) + tuple(padding), mode="constant"
        )

        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1])
            for ax in range(3)
        )

        padded_inverted_affs = np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[
            crop_slices
        ]
        rolled_affs.append(padded_inverted_affs)
        del padded_inverted_affs

    rolled_affs = np.stack(rolled_affs)
    return rolled_affs


def compute_mc_costs(boundary_pmaps, rag, superpixels, beta):
    # compute the edge costs
    features = compute_boundary_mean_and_length(
        rag, segmentation=superpixels, input_=boundary_pmaps
    )
    costs, sizes = features[:, 0], features[:, 1]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by interpreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.
    # in addition, we weight the costs by the size of the corresponding edge

    costs = transform_probabilities_to_costs(costs, edge_sizes=sizes, beta=beta)
    return costs
