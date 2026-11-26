# RSPTrack: Language-Assisted Visual Tracking with\\ Layer-wise Rectified Sparse Attention



## Overview

#### Our work

Visual object tracking is challenged by occlusion, appearance changes, and similar-object interference. Language offers complementary semantic cues, but text annotations are often incomplete or missing. Existing vision-language joint encoding methods face two challenges: 1) category attributes may be shared by similar objects, while background clutter in templates weakens instance discrimination; 2) unreliable local cues increase reliance on context, yet dense interactions introduce distractions and excessive sparsity discards critical information. We propose RSPTrack, a language-assisted tracking framework integrating pre-fusion target-conditioned modulation with query-specific information selection during joint encoding, so semantic cues influence both search representations and information aggregation. For the first challenge, Target-aware Visual Prompt Modulation (TVPM) combines template foreground appearance and text to generate latent prompts for target, boundary, and background roles, using gated and differential residuals to modulate representations without increasing sequence length. For the second, Layer-wise Rectified Sparse Attention (LRSA) selects relevant key blocks per query while retaining local and essential semantic connections, with periodic dense attention restoring global interactions to recover missed context, balancing focused aggregation with contextual completeness.

![framework](asset/framework.png)



#### Experimental Performance

![experiment](asset/experiment.png)



## Installation

```bash
conda create -n rsptrack python=3.8 -y
conda activate rsptrack
pip install -r requirements.txt
```



## Usage

#### Data Preparation

Our RSPTrack is trained on LaSOT, TNL2K, RefCOCOg, GOT-10k, and TrackingNet datasets.
Put these tracking datasets in [./data]. It should look like:

```text
${RSPTrack_ROOT}
 -- data
     -- Lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- GOT10k
         |-- test
         |-- train
         |-- val
     -- COCO
         |-- annotations
         |-- images
     -- Trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
     -- TNL2K
         -- train
             |-- Arrow_Video_ZZ04_done
             |-- Assassin_video_1-Done
             |-- Assassin_video_2-Done
             ...
         -- test
             |-- advSamp_Baseball_game_002-Done
             |-- advSamp_Baseball_video_01-Done
             |-- advSamp_Baseball_video_02-Done
             ...

```

You can also use (which is what we use by default)：

```text
data/
├── Lasot_lmdb/
├── GOT10k_lmdb/
├── COCO_lmdb/
├── Trackingnet_lmdb/
├── TNL2K/
│   ├── train_set/
│   └── test_set/
```

`--use_lmdb 1` enables the use of LMDB for LaSOT, GOT-10k, COCO, and TrackingNet, while TNL2K continues to be read from the original files.



#### Set project paths

Run the following command to set paths for this project

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

After running this command, you can also modify paths by editing these two files

```bash
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```



#### train

The backbone  of RSPTrack are initialized with pre-trained weights from [**Fast-iTPN**](https://github.com/sunsmarterjie/iTPN).
Please download the **fast_itpn_base_clipl_e1600.pt** and place them in [./pretrained/itpn].

You can run the following command to train RSPTrack:

```bash
python tracking/train.py --config RSPTrack-B --save_dir ./output --use_lmdb 1
```



#### test and evaluation

First, you need to set the paths for the various evaluation benchmarks in `./lib/test/evaluation/local.py` , and prepare the model weights for evaluation. Then, run the following command to perform evaluation on different benchmarks.

- LaSOT

```bash
python tracking/test.py tracker_name tracker_param --dataset_name lasot
python tracking/analysis_results.py --dataset_name lasot
```

- LaSOT_ext

```bash
python tracking/test.py tracker_name tracker_param --dataset_name lasot_extension_subset
python tracking/analysis_results.py --dataset_name lasot_extension_subset
```

- TNL2K

```bash
python tracking/test.py tracker_name tracker_param --dataset_name tnl2k
python tracking/analysis_results.py --dataset_name tnl2k
```



## Model Zoo

The trained models, and the raw tracking results are provided in ==[......]==



## Acknowledgement

Our work has been significantly inspired by the following open-source repositories: [SAVLT](https://github.com/zj5559/SAVLT), [SUTrack](https://github.com/chenxin-dlut/SUTrack), [Fast-iTPN](https://github.com/sunsmarterjie/iTPN), and [CoCoOP](https://github.com/KaiyangZhou/CoOp). We gratefully acknowledge the insights and foundations they provided, which have been instrumental in shaping this project.

