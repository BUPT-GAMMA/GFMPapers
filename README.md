# GFMPapers

Must-read papers on graph foundation models (GFMs)

![](https://img.shields.io/github/last-commit/BUPT-GAMMA/GFMpapers?color=blue) ![](https://img.shields.io/badge/PRs-Welcome-red) 

This list is currently maintained members in BUPT GAMMA Lab. 

We thank all the great [contributors](#contributors) very much.

### Contents

- [GFMPapers](#gfmpapers)
    - [Contents](#contents)
    - [Keywords Convention](#keywords-convention)
  - [0. Survey Papers](#0-survey-papers)
  - [1. GNN-based Papers](#1-gnn-based-papers)
  - [2. LLM-based Papers](#2-llm-based-papers)
  - [3. GNN+LLM-based Papers](#3-gnnllm-based-papers)
  - [Contributors](#contributors)


### Keywords Convention

![](https://img.shields.io/badge/xxx-DCE7F1) backbone architecture

![](https://img.shields.io/badge/yyy-EAD8D9) Pretraining

![](https://img.shields.io/badge/zzz-D8D0E1) Adaptation

The meaning of each tag can be referred to in the "Towards Graph Foundation Models: A Survey and Beyond" paper.

## 0. Survey Papers
1. **Towards Graph Foundation Models: A Survey and Beyond**. *Jiawei Liu, Cheng Yang, Zhiyuan Lu, Junze Chen, Yibo Li, Mengmei Zhang, Ting Bai, Yuan Fang, Lichao Sun, Philip S. Yu, Chuan Shi*. arXiv 2023.10. [[pdf](https://arxiv.org/pdf/2310.11829.pdf)]


## 1. GNN-based Papers

1. **All in One: Multi-Task Prompting for Graph Neural Networks**. *Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, Jihong Guan*. KDD 2023. [[pdf](https://www.researchgate.net/profile/Jia-Li-127/publication/371608827_All_in_One_Multi-Task_Prompting_for_Graph_Neural_Networks/links/648c2270c41fb852dd0a4f62/All-in-One-Multi-Task-Prompting-for-Graph-Neural-Networks.pdf)] ![](https://img.shields.io/badge/GCN/GAT/Graph_Transformer-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. **PRODIGY: Enabling In-context Learning Over Graphs**. *Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec*. arXiv 2023.5. [[pdf](https://arxiv.org/pdf/2305.12600.pdf)] ![](https://img.shields.io/badge/GCN/GAT-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Supervised-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. **Deep graph infomax**. *Petar Veliˇckovi´c, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm*. arXiv 2018.9. [[pdf](https://arxiv.org/pdf/1809.10341)] [[code](https://github.com/PetarV-/DGI)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Cross--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Deep graph contrastive representation learning**. *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, and Liang Wang*. arXiv 2020.6. [[pdf](https://arxiv.org/pdf/2006.04131.pdf)] [[code](https://github.com/CRIPAC-DIG/GRACE)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Same--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
1. **Variational graph auto-encoders**. *Thomas N Kipf and Max Welling*. arXiv 2016.11. [[pdf](https://arxiv.org/pdf/1611.07308.pdf)] [[code](https://github.com/tkipf/gae)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Graph_Reconstruction/Property_Prediction-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
1. **Ma-gcl: Model augmentation tricks for graph contrastive learning**. *Xumeng Gong, Cheng Yang, and Chuan Shi*. AAAI 2023. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/download/25547/25319)] [[code](https://github.com/GXM1141/MA-GCL)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)


## 2. LLM-based Papers

1. **Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning**. *H. Zhao, S. Liu, C. Ma, H. Xu, J. Fu, Z.-H. Deng L. Kong, and Q. Liu*. Arxiv 2023.05.[[pdf](https://scholar.google.com/scholar_url?url=https://www.biorxiv.org/content/biorxiv/early/2023/06/01/2023.05.30.542904.full.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=8390578571473859304&ei=8QRCZbaEHIz5yATYnbjoDQ&scisig=AFWwaebONdO5ia5yjK3p4wA-pOf1)] ![](https://img.shields.io/badge/transformer-DCE7F1)
2. **Meta-Transformer: A Unified Framework for Multimodal Learning**. *Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, Xiangyu Yue*. Arxiv 2023.07.[[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2307.10802&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=11077145075852511910&ei=cgZCZZG8AcaKywSmq6XgCw&scisig=AFWwaeaAZA6AHtPdRvX0JnyNhv1F)] [[code](https://github.com/invictus717/MetaTransformer)] ![](https://img.shields.io/badge/transformer-DCE7F1)
3. **Natural language is all a graph needs**. *R Ye, C. Zhang, R. Wang, S. Xu, and Y. Zhang*. 2023.08 [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2308.07134&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=14935989239849530960&ei=WgdCZbaJFtqk6rQPkv-skA8&scisig=AFWwaeZJKMJktGJOJmeusMs1l5k1)] [[code](https://github.com/agiresearch/InstructGLM)] ![](https://img.shields.io/badge/Graph_to_token_+_Flan_T5,LLaMA-DCE7F1) ![](https://img.shields.io/badge/MLM,LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1) 
4. **Evaluating large language models on graphs: Performance insights and comparative analysis**. *Liu and B. Wu*. Arxiv 2023.08. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2308.11224&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=13367291863109264530&ei=JQlCZcecFqKQ6rQPwPS0qAw&scisig=AFWwaebcL4UoKZKs-b3HfIKmzeoB)] [[code](https://github.com/ayame1006/llmtograph)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs,Vicuna-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
5. **Can language models solve graph problems in natural language?** *H. Wang, S. Feng, T. He, Z Tan, X. Han, and Y. Tsvetkov*. Arxiv 2023.05. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2305.10037&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=10660384245119063422&ei=UgpCZeaYLKKQ6rQPwPS0qAw&scisig=AFWwaeb83Q4qbndJ3rQeda6SsHVD)] [[code](https://github.com/arthur-heng/nlgraph)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
6. **Graphtext: Graph reasoning in text space**. *. Zhao, L. Zhuo, Y. Shen, M. Qu, K. Liu, M Bronstein, Z. Zhu, and J. Tang*. Arxiv 2023.10. [[pdf](https://arxiv.org/pdf/2310.01089.pdf)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
7. **Can large language models empower molecular property prediction?** *Qian, H. Tang, Z. Yang, H Liang, and Y. Liu*. Arxiv 2023.07. [[pdf](https://arxiv.org/pdf/2307.07443.pdf)] [[code](https://github.com/chnq/llm4mol)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
8. **Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking** *J. Guo, L. Du, and H. Liu*. Arxiv 2023.05. [[pdf](https://arxiv.org/pdf/2305.15066.pdf)] ![](https://img.shields.io/badge/Graph_to_text_+_GPT_3-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning_+_Automatic_Prompt_Tuning-D8D0E1)
9. **Exploring the potential of large language models (llms) in learning on graphs** *Chen, H. Mao, H. Li, W. Jin, H. Wen, X. Wei, S. Wang, D. Yin, W. Fan, H. Liu, et al*. Arxiv 2023.07. [[pdf](https://arxiv.org/pdf/2307.03393.pdf)] [[code](https://github.com/CurryTang/Graph-LLM)] ![](https://img.shields.io/badge/Graph_to_text_+_Bert,sBert,LLaMa,GPTs-DCE7F1) ![](https://img.shields.io/badge/LM,MLM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning_+_Automatic_Prompt_Tuning-D8D0E1)

## 3. GNN+LLM-based Papers
1. **Simteg: A frustratingly simple approach improves
textual graph learning**. *K. Duan, Q. Liu, T.-S. Chua, S. Yan, W. T. Ooi, Q. Xie, and
J. He*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2308.02565.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM,TTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--EfficientFT-D8D0E1)
1. **Explanations as features: Llm-based features for text-attributed graphs**. *Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, Jihong Guan*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2305.19523.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9)  ![](https://img.shields.io/badge/Tuning-free_Prompting+Parameter--Efficient_FT-D8D0E1)
1. **Node feature extraction by self-supervised multi-scale neighborhood prediction**. *E. Chien, W. Chang, C. Hsieh, H. Yu, J. Zhang,
O. Milenkovic, and I. S. Dhillon*. ICLR 2022. [[pdf](https://arxiv.org/pdf/2111.00064.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/vanillal--FT-D8D0E1)
1. **Train your own GNN teacher: Graph-aware distillation on textual graphs**. *Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, Jihong Guan*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2304.10668.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Graphformers: Gnn-nested transformers for representation learning on textualgraph**. *EJ. Yang, Z. Liu, S. Xiao, C. Li, D. Lian, S. Agrawal, A. Singh, G. Sun, and X. Xie*. NIPS 2021. [[pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/f18a6d1cde4b205199de8729a6637b42-Paper.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Vanilla--FT-D8D0E1)
1. **Learning on large-scale text-attributed graphs via variational inference**. *J. Zhao, M. Qu, C. Li, H. Yan, Q. Liu, R. Li, X. Xie, and
J. Tang*. ICLR 2023. [[pdf](https://arxiv.org/pdf/2210.14709.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
1. **Congrat: Self-supervised contrastive pretraining for joint graph and text embeddings**. *. Brannon, S. Fulay, H. Jiang, W. Kang, B. Roy, J. Kabbara, and D. Roy*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2305.14321.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Augmenting low-resource text classification with graph-grounded pre-training and prompting**. *Wen and Y. Fang*. SIGIR 2023. [[pdf](https://arxiv.org/pdf/2305.03324.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/GTCL-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. **Graph-based modeling of online communities for fake news detection**. *Chandra, P. Mishra, H. Yannakoudakis, M. Nimishakavi,
M. Saeidi, and E. Shutova*. Arxiv 2020. [[pdf](hhttps://arxiv.org/pdf/2008.06274.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Text2mol: Cross-modal molecule retrieval with natural language queries**. *Edwards, C. Zhai, and H. Ji*. EMNLP 2021. [[pdf](https://aclanthology.org/2021.emnlp-main.47.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **A molecular multimodal foundation model associating molecule graphs with natural language**. *B. Su, D. Du, Z. Yang, Y. Zhou, J. Li, A. Rao, H. Sun, Z. Lu, and J.-R. Wen*. Arxiv 2022. [[pdf](https://arxiv.org/pdf/2209.05481.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Multi-modal molecule structure-text model for text-based retrieval and editing**. *S. Liu, W. Nie, C. Wang, J. Lu, Z. Qiao, L. Liu, J. Tang, C. Xiao, and A. Anandkumar*. Arxiv 2022. [[pdf](https://arxiv.org/pdf/2212.10789.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Enhancing activity prediction models in drug discovery with the ability to understand human language**. *P. Seidl, A. Vall, S. Hochreiter, and G. Klambauer*. PMLR 2023. [[pdf](https://arxiv.org/pdf/2303.03363.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. **Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt**. *J. Zhang*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2304.11116.pdf)] ![](https://img.shields.io/badge/LLM--centric-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9)  ![](https://img.shields.io/badge/Tuning--free_Prompting+Vanilla_FT-D8D0E1)




## Contributors

We thank all the contributors to this list. And more contributions are very welcome.

<a href="https://github.com/BUPT-GAMMA/GFMpapers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BUPT-GAMMA/GFMpapers" />
</a>

