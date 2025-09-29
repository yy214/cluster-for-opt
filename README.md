# Clustering for stochastic gradient descent

This is the code associated with the end of studies internship report (2025) of Shun Ye CHEN from ENSTA Paris.

The report is unfortunately only available to members of the ENSTA community on https://bibnum.ensta.fr


## Datasets used
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
Available at: 
  [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
  [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
  [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE, 86(11):2278–2324, November 1998.
MNIST database available at http://yann.lecun.com/exdb/mnist/.

R. Mohammad, F. Thabtah and L. Mccluskey.
An assessment of features related to phishing websites using an automated technique
International Conference for Internet Technology and Secured Transactions, December 2012
Downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#phishing.


## Sources / inspirations for the algorithms

```bibtex
@misc{allenzhu2016exploitingstructurestochasticgradient,
    title={Exploiting the Structure: Stochastic Gradient Methods Using Raw Clusters}, 
    author={Zeyuan Allen-Zhu and Yang Yuan and Karthik Sridharan},
    year={2016},
    eprint={1602.02151},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/1602.02151}, 
}
@misc{faghri2020studygradientvariancedeep,
    title={A Study of Gradient Variance in Deep Learning}, 
    author={Fartash Faghri and David Duvenaud and David J. Fleet and Jimmy Ba},
    year={2020},
    eprint={2007.04532},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2007.04532}, 
}
@INPROCEEDINGS{8682527,
    author={Yuan, Kun and Ying, Bicheng and Sayed, Ali H.},
    booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={COVER: A Cluster-based Variance Reduced Method for Online Learning}, 
    year={2019},
    volume={},
    number={},
    pages={3102-3106},
    keywords={Convergence;Indexes;Steady-state;Probability distribution;Clustering algorithms;Standards;Optimization;Online leaming;Streaming data;Internal structure;Variance reduction;SGD;SAGA},
    doi={10.1109/ICASSP.2019.8682527}
}
@misc{zhao2014acceleratingminibatchstochasticgradient,
    title={Accelerating Minibatch Stochastic Gradient Descent using Stratified Sampling}, 
    author={Peilin Zhao and Tong Zhang},
    year={2014},
    eprint={1405.3080},
    archivePrefix={arXiv},
    primaryClass={stat.ML},
    url={https://arxiv.org/abs/1405.3080}, 
}
```