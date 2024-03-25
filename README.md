# 《MutualFormer: Multi-Modality Representation Learning via Cross-Diffusion Attention》IJCV 2024

## Abstract 
Aggregating multi-modality data to obtain reliable data representation attracts more and more attention. Recent studies demonstrate
that Transformer models usually work well for multi-modality tasks.
Existing Transformers generally either adopt the Cross-Attention (CA)
mechanism or simple concatenation to achieve the information interaction among different modalities which generally ignore the issue
of modality gap. In this work, we re-think Transformer and extend
it to MutualFormer for multi-modality data representation. Rather
than CA in Transformer, MutualFormer employs our new design of
Cross-Diffusion Attention (CDA) to conduct the information communication among different modalities. Comparing with CA, the main
advantages of the proposed CDA are three aspects. First, the cross-affinities in CDA are defined based on the individual modality affinities in the metric space which thus can naturally avoid the issue
of modality/domain gap in feature based CA definition. Second,
CDA provides a general scheme which can either be used for multi-modality representation or serve as the post-optimization for existing
CA models. Third, CDA is implemented efficiently. We successfully
apply the MutualFormer on different multi-modality learning tasks
(i.e., RGB-Depth SOD, RGB-NIR object ReID). Extensive experiments demonstrate the effectiveness of the proposed MutualFormer.

## Architecture of MutualFormer and Cross-Affinity Attention (CDA) module
![overview](https://github.com/SissiW/MutualFormer/blob/main/MutualFormer_overview.png)

![CDA](https://github.com/SissiW/MutualFormer/blob/main/CDA.png)

## Results on RGB-D SOD datasets
![results](https://github.com/SissiW/MutualFormer/blob/main/SOD_results.png)

![qul_results](https://github.com/SissiW/MutualFormer/blob/main/SOD_qualitative_results.png)

More results on other tasks can be found in the paper. The saliency maps produced by MutualFormer-based RGB-D SOD method can be downloaded in Baidu Drive [RGBD_SOD_Results](https://pan.baidu.com/s/1hsypcBSrfOcrZz9u0tNJAA)(password: 5jdf).

## Datasets
The RGB-D SOD datasets can be downloaded to click Baidu Drive ([Data](https://pan.baidu.com/s/1zFiy7c3P4sb_w0cOVtQucA) (password: 92cc)

## Installation
python3.6+, pytorch>=1.6, tensorboard

## Download Pre-trained Models
[Baidu Drive](https://pan.baidu.com/s/1UWnpjNaaCTSUB2sOtJqZng)
提取码：yd8w

## Config
```
sh train_meta_QSFormer.sh
```


## Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@article{wang2023few,
  title={Few-Shot Learning Meets Transformer: Unified Query-Support Transformers for Few-Shot Classification},
  author={Wang, Xixi and Wang, Xiao and Jiang, Bo and Luo, Bin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
This project is built upon [DeepEMD](https://github.com/icoz69/DeepEMD). We also reference some code from [DETR](https://github.com/facebookresearch/detr). Thanks to the contributors of these great codebases.
