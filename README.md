# Semantic Segmentation with Noisy Boundary Annotations

Implemented boundary detection based on "Devil is in the Edges: Learning Semantic Boundaries from Noisy Annotations" (see [link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf)), generalized it to 3-dimension cases.

### Implementation

- [x] Basic 2D/3D CASENet network with weighted multilabel BCE loss

- [x] 2D/3D Geodesic active contour inference
- [x] Iterative update between network training and level-set refinement
- [x] 3D UNET (obsolete code)
- [ ] NMS loss and direction losstr

### Configuration example 

1. Setup configuration for 2D CASENet with 2D level set: training ([link](./resources/train_config_case2D.yaml)), testing ([link](./resources/test_config_case2D.yaml))
2. Setup configuration for 3D CASENet with 3D level set: trianing ([link](./resources/train_config_case3D.yaml))
3. obsolete configuration code: UNet3D traning([link](./resources/train_config_unet3D.yaml)), testing ([link](./resources/test_config_unet3D.yaml))

### Usage

**Clone this repo**

```bash
git clone http://gitlab.bj.sensetime.com/shenrui/edgeDL.git
cd edgeDL
```

**Install dependencies**

Require Python 3.6+ and Pytorch 1.0+. Please install dependencies by

```bash
conda env create -f environment.yml
```

**Preprocessing**

Resample the data into same resolution. This code requires Free Surfer mri_convert (see [link](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert), Free Surfer installation [guide](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)). 

```bash
./utils/resample.sh
```

Generate file lists for traning, validation and testing sets.

```bash
python data2txt.py
```

**Traning**

Setup configuration file and run

```bash
python train_casenet.py --config PATH_TO_CONFIG_FILE
```

**Testing**

Setup configuration file and run

```bash
python predict_casenet.py --config PATH_TO_CONFIG_FILE
```

### Loss function

1. Weighted multilabel BCE loss

   $\mathcal{L}_{BCE}(\theta) = - \sum_k\sum_m\{\beta y_k^m\log f_k(m|x,\theta) + (1-\beta) (1-y_k^m)\log(1 - f_k(m|x,\theta))\}$

   where

   $\beta$ : non-edge pixels/voxels ratio, $\beta = \frac{|Y^-|}{|Y|}$

   $k$ : class

   $m$ : pixel/voxel

2. NMS loss (edge thinning layers, to be implemented)

   $\mathcal{L}_{NSM}(\theta) = -\sum_k\sum_p \log h_k(p|x,\theta)$

   where

   $h_k(p|x,\theta) = \frac{\exp(f_k(p|x,\theta)/\tau)}{\sum_{t=-L}^L \exp(f_k(p_t|x,\theta)/\tau)}$ for normalization

   $x(p_t) = x(p) + t · \cos \vec{d_p} $ , $y(p_t) = y(p) + t · \sin \vec{d_p}$

   $p$ : gt boundary pixel/voxel

   $\vec{d_p}$ : normal direction at $p$ computed from gt boundary map

   $t \in \{-L, -L+1, ... L\}$

   ##### Notes for implementation

   - Normal direction: use a fixed convolutoonal layer to estimate second derivatives, and then use trigonometry function to compute normal direction from the gt boundary map
   - code reference: edgesNMS([link](https://github.com/pdollar/edges/blob/master/private/edgesNmsMex.cpp))

3. Direction Loss (to be implemented)

   $\mathcal{L}_{Dir}(\theta) = \sum_k\sum_p ||\cos ^{-1} <\vec{d_p}, \vec{e_p}(\theta)>||$

   where

   $\vec{e_p}(\theta)$ : normal direction at p computed from prediction map

### Level Set

1. Level set evolution

   $\frac{\partial \phi}{\partial t} = g_k(\kappa + c)|\nabla\phi| + \nabla g_k · \nabla \phi$

   solved by morphological approach (see [link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.3979&rep=rep1&type=pdf))

2. Energy (edge) map for level set alignment

   $g_k = \frac{1}{\sqrt{1+\alpha f_k}}+\frac{\lambda}{\sqrt{1+\alpha \sigma(y_k)}}$

   where

   $f_k$ : probability map predicted by neural network

   $\sigma(y_k)$ : (previous) ground truth annotation smoothed by gaussian filter with $\sigma$

### Github reference:

1. STEAL ([link](https://github.com/nv-tlabs/STEAL))
2. edges ([link](https://github.com/pdollar/edges))
3. Morphsnakes ([link](https://github.com/pmneila/morphsnakes))