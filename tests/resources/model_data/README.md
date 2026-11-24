# Model test data

Small crops of the public datasets PanSeg's models were trained on, so the tests run each real model on the kind of image it was made for. Each file holds `raw` and `label`, both with `element_size_um` (ZYX). In `label`, 0 is anything that isn't a cell: background, or walls not assigned to a cell.

`make_crops.py` rebuilds the two crops in this folder from their sources; it reads about 75 MB over the network.

| File | Content | Shape, voxel size (ZYX, µm) | Source | License | Models |
|---|---|---|---|---|---|
| `shoot_meristem_cells.h5` | Arabidopsis shoot apical meristem, cell walls (acylYFP), confocal, with the published segmentation | 100 × 128 × 128 uint8, 0.25 isotropic | Willis et al. 2016, PNAS 113(51), doi:10.17863/CAM.7793: `PNAS.zip`, stack `8hrs_plant2`, rescaled to 0.25 µm with the z-stretching factor the dataset gives | CC BY 4.0 | `emotional-cricket`, `confocal_3D_unet_sa_meristem_cells` |
| `lateral_root_cells.h5` | Arabidopsis lateral root primordium, cell walls, light-sheet, with ground-truth cells | 32 × 192 × 192 uint8, 0.25 × 0.1625 × 0.1625 | Wolny et al. 2020, eLife 9:e57613, OSF doi:10.17605/OSF.IO/UZQ3W: test set, `Movie1_t00006_crop_gt.h5` | PlantSeg data of the Kreshuk lab; cite Wolny et al. 2020 | `thoughtful-turtle`, `generic_light_sheet_3D_unet`, `lightsheet_*_unet_root_*` |
| `../sample_ovule.h5` | Arabidopsis ovule, cell walls, confocal, with ground-truth cells | 32 × 128 × 128 uint8, 0.235 × 0.15 × 0.15 | Wolny et al. 2020, the OSF ovules dataset; where it was cut from wasn't recorded, but it matches that dataset's half-resolution files | as above | `generic_confocal_3D_unet`, `confocal_*_unet_ovules_ds2x`, `passionate-t-rex`, `pioneering-rhino` |
| `../data/{train,val}/113*_mini.h5` | Arabidopsis ovule nuclei (TO-PRO-3), confocal, with gold-standard nuclei | 100 × 100 × 100 uint8, 0.284 × 0.127 × 0.127 | Go-Nuclear, doi:10.1101/2024.02.19.580954, BioImage Archive S-BIAD1026: stacks 1135 to 1137 | CC0 | `philosophical-panda`, `efficient-chipmunk`, `PlantSeg_3Dnuc_platinum`; 1135 is held out from their training |

None of the new crops was trained on: `8hrs_plant2` is in `emotional-cricket`'s validation set and not its training set (its `config_train.yml`, Zenodo 7768142), and the lateral root file is from the test set.
