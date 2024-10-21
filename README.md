<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (CIKM'24) Rethinking Attention Mechanism for Spatio-Temporal Modeling: A Decoupling Perspective in Traffic Flow Prediction </b></h2>
</div>

## Model

The input traffic flow tensor is decomposed into trend and seasonal parts through Trend Decomposition (De-Trend) module, and those two parts are modeled separately to extract temporal correlation. The input traffic topology graph derives a pattern matrix through Traffic Pattern Extraction (TS-Pattern) module, and that pattern matrix assists in extracting spatial correlation

<div align=center>
<img src="https://github.com/HCHHOH/DEC-Former/blob/main/img/overview.png" width='80%'>
</div>

## Main Results

The performance of DEC-Former was evaluated on four real-world traffic flow datasets, PeMS03, PeMS04, PeMS07 and PeMS08.

<div align=center>
<img src="https://github.com/HCHHOH/DEC-Former/blob/main/img/performance.png" width='75%'>
</div>

## Requirements
- python 3.7
- numpy == 1.19.2
- scipy == 1.7.3
- pandas == 1.1.5
- torch == 1.10.1
- scikit-learn==0.24.0


## Data
The dataset used in the paper can be downloaded from [BaiDuYun](https://pan.baidu.com/s/1jy-IUNTZUUYHNXtx7BpM-w?pwd=y1ef).

## Citation

```
@inproceedings{DEC-Former,
  author = {Yu, Qi and Ding, Weilong and Zhang, Hao and Yang, Yang and Zhang, Tianpu},
  title = {Rethinking Attention Mechanism for Spatio-Temporal Modeling: A Decoupling Perspective in Traffic Flow Prediction},
  year = {2024},
  isbn = {9798400704369},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3627673.3679571},
  doi = {10.1145/3627673.3679571},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages = {3032–3041},
  numpages = {10},
  keywords = {attention mechanism, spatio-temporal representation, traffic prediction},
  location = {Boise, ID, USA},
  series = {CIKM '24}
}

```

