# PanSeg introduction

!!! tip "PlantSeg is now PanSeg!"
    The project was renamed to **PanSeg** instead of **PlantSeg** to highlight
    its capabilities beyond plant cell segmentation!  

!!! warning "Technical Issues"
    The renaming process is ongoing, and you might encounter technical issues
    with the new version until this process is finished.
    We are aware of issues with the windows installer and the in-app updater.
    If you encounter issues, please use PlantSeg <= 2.0.0rc14 for now, thank you!

PanSeg is a tool for 3D and 2D segmentation.
The methods used are very generic and can be used for any instance segmentation workflow,
but they are tuned towards cell segmentation in plant tissue. The tool is fundamentally composed of two main steps.

<figure markdown="span">
  ![Main Figure](https://github.com/kreshuklab/panseg/raw/assets/images/main_figure_nologo.png)
  <figcaption>Figure: PanSeg Main Workflow</figcaption>
</figure>

* ***Cell boundary prediction***: A convolutional neural network (CNN) is utilized to perform voxel-wise boundary classification. This network is adept at filtering out diverse types and intensities of noise, homogenizing signal strength, and correcting imaging defects such as blurred or missing cell boundaries. This step ensures a high-quality boundary prediction which is crucial for accurate segmentation.

* ***Cell Segmentation as graph partitioning***: The boundary prediction from the first step serve as the basis for automated segmentation. PanSeg implements four distinct algorithms for this task, each with unique features tailored to different segmentation needs. This graph partitioning approach is particularly effective for segmenting densely packed cells.

<figure markdown="span">
  ![Main Figure](./logos/panseg2gui.png)
  <figcaption>Figure: PanSeg v2 Interface</figcaption>
</figure>

For a detailed description of the methods employed in PanSeg, please refer to our [manuscript](https://elifesciences.org/articles/57613). If you find PanSeg useful in your research, please consider citing our work:

```bibtex
@article{wolny2020accurate,
  title={Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
  author={Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, S{\"o}ren and Wilson-S{\'a}nchez, David and Lymbouridou, Rena and others},
  journal={Elife},
  volume={9},
  pages={e57613},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

## PanSeg & PlantSeg

This project stated 2020 as `PlantSeg`, but its capabilities are not
restricted to plant cells! To better reflect that, for the 2.0 release
we changed the name to `PanSeg`, referring to plant and animal cells.
