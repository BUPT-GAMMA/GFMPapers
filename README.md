<h2 align="center"><b>GFMPapers: Must-read papers on graph foundation models (GFMs)</b></h2>

![](https://img.shields.io/github/last-commit/BUPT-GAMMA/GFMpapers?color=blue) ![](https://img.shields.io/badge/PRs-Welcome-red) 

This list is currently maintained members in BUPT GAMMA Lab. 

We thank all the great [contributors](#contributors) very much.

### Contents


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
1. [arXiv 2023.10] **Towards Graph Foundation Models: A Survey and Beyond**. [[pdf](https://arxiv.org/pdf/2310.11829.pdf)]

## 1. GNN-based Papers
1. [arXiv 2023.10] **HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous Graph Neural Networks** [[pdf](https://arxiv.org/abs/2310.15318)] ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. [arXiv 2023.10] **Prompt Tuning for Multi-View Graph Contrastive Learning** [[pdf](https://arxiv.org/abs/2310.10362)] ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. [KDD 2023] **All in One: Multi-Task Prompting for Graph Neural Networks**. [[pdf](https://www.researchgate.net/profile/Jia-Li-127/publication/371608827_All_in_One_Multi-Task_Prompting_for_Graph_Neural_Networks/links/648c2270c41fb852dd0a4f62/All-in-One-Multi-Task-Prompting-for-Graph-Neural-Networks.pdf)] ![](https://img.shields.io/badge/GCN/GAT/Graph_Transformer-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. [arXiv 2023.05] **PRODIGY: Enabling In-context Learning Over Graphs**. [[pdf](https://arxiv.org/pdf/2305.12600.pdf)] ![](https://img.shields.io/badge/GCN/GAT-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Supervised-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. [ICLR 2019] **Deep graph infomax**. [[pdf](https://arxiv.org/pdf/1809.10341)] [[code](https://github.com/PetarV-/DGI)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Cross--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [arXiv 2016.11] **Variational graph auto-encoders**. [[pdf](https://arxiv.org/pdf/1611.07308.pdf)] [[code](https://github.com/tkipf/gae)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Property_Prediction-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
1. [AAAI 2023] **Ma-gcl: Model augmentation tricks for graph contrastive learning**. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/download/25547/25319)] [[code](https://github.com/GXM1141/MA-GCL)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
4. [ICML 2020] **Deep graph contrastive representation learning.** [[pdf](https://arxiv.org/pdf/2006.04131.pdf)] [[code](https://github.com/CRIPAC-DIG/GRACE)] ![](https://img.shields.io/badge/GCN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
7. [KDD 2022] **GraphMAE: Self-supervised masked graph autoencoders.** [[pdf](https://dl.acm.org/doi/pdf/10.1145/3534678.3539321)] [[code](https://github.com/THUDM/GraphMAE)] ![](https://img.shields.io/badge/GAT-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
8. [WWW 2023] **GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner.** [[pdf](https://github.com/THUDM/GraphMAE2)] [[code](https://dl.acm.org/doi/pdf/10.1145/3543507.3583379)] ![](https://img.shields.io/badge/GAT-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
9. [KDD 2022] **Gppt: Graph pre-training and prompt tuning to generalize graph neural networks.** ![](https://img.shields.io/badge/GraphSAGE-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction/Cross--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
10. [CIKM 2023] **Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks.** [[pdf](https://github.com/WenZhihao666/VPGNN)] [[code](https://dl.acm.org/doi/pdf/10.1145/3583780.3615505)] ![](https://img.shields.io/badge/GraphSAGE-DCE7F1) ![](https://img.shields.io/badge/Cross--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
11. [KDD 2020] **Gpt-gnn: Generative pre-training of graph neural networks.** [[pdf](https://github.com/acbull/GPT-GNN)] [[code](https://dl.acm.org/doi/pdf/10.1145/3394486.3403237)] ![](https://img.shields.io/badge/HGT-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
12. [KDD 2021] **Pre-training on large-scale heterogeneous graph.** [[pdf](https://github.com/BUPT-GAMMA/PTHGNN)] [[code](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7891&context=sis_research)] ![](https://img.shields.io/badge/HGT-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
13. [CIKM 2021] **Contrastive pre-training of GNNs on heterogeneous graphs.** [[pdf](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7892&context=sis_research)] [[code]([https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7892&context=sis_research](https://github.com/BUPT-GAMMA/CPT-HG))] ![](https://img.shields.io/badge/HGT-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
14. [WWW 2023] **Graphprompt: Unifying pre-training and downstream tasks for graph neural networks.** [[pdf](https://dl.acm.org/doi/pdf/10.1145/3543507.3583386)] [[code](https://github.com/Starlien95/GraphPrompt)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
15. [KDD 2020] **Gcc: Graph contrastive coding for graph neural network pre-training.** [[pdf](https://arxiv.org/pdf/2006.09963.pdf)] [[code](https://github.com/THUDM/GCC)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
16. [NeurIPS 2020] **Graph contrastive learning with augmentations.** [[pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf)] [[code](https://github.com/Shen-Lab/GraphCL)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL-EAD8D9) ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
17. [arXiv 2023.04] **AdapterGNN: Efficient Delta Tuning Improves Generalization Ability in Graph Neural Networks.** [[pdf](https://arxiv.org/pdf/2304.09595.pdf)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Cross--Scale_CL/Same--Scale_CL/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
18. [KDD 2023] **A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability.** [[pdf](http://shichuan.org/doc/150.pdf)] [[code](https://github.com/BUPT-GAMMA/AAGOD)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Cross--Scale_CL/Same--Scale_CL/Supervised-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
19. [arXiv 2022.09] **Universal Prompt Tuning for Graph Neural Networks.** [[pdf](https://www.researchgate.net/profile/Taoran-Fang/publication/364110035_Prompt_Tuning_for_Graph_Neural_Networks/links/647abf7879a722376509c6a9/Prompt-Tuning-for-Graph-Neural-Networks.pdf)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Cross--Scale_CL/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
20. [arXiv 2023.02] **SGL-PT: A Strong Graph Learner with Graph Prompt Tuning.** [[pdf](https://arxiv.org/pdf/2302.12449.pdf)] ![](https://img.shields.io/badge/GIN-DCE7F1) ![](https://img.shields.io/badge/Same--Scale_CL/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
21. [arXiv 2020.01] **Graph-bert: Only attention is needed for learning graph representations.** [[pdf](https://arxiv.org/pdf/2001.05140.pdf)] [[code](https://github.com/anonymous-sourcecode/Graph-Bert)] ![](https://img.shields.io/badge/Graph_Transformer-DCE7F1) ![](https://img.shields.io/badge/Supervised/Graph_Reconstruction-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
22. [NeurIPS 2020] **Selfsupervised graph transformer on large-scale molecular data.** [[pdf](https://proceedings.neurips.cc/paper/2020/file/94aef38441efa3380a3bed3faf1f9d5d-Paper.pdf)] ![](https://img.shields.io/badge/Graph_Transformer-DCE7F1) ![](https://img.shields.io/badge/Property_Prediction-EAD8D9) ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
23. [arXiv 2023.05] **G-Adapter: Towards Structure-Aware Parameter-Efficient Transfer Learning for Graph Transformer Networks.** [[pdf](https://arxiv.org/pdf/2305.10329.pdf)] ![](https://img.shields.io/badge/Graph_Transformer-DCE7F1) ![](https://img.shields.io/badge/Supervised/Graph_Reconstruction/Property_Prediction-EAD8D9) ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)


## 2. LLM-based Papers
1. [arXiv 2023.10] **Talk Like a Graph: Encoding Graphs for Large Language Models** [[pdf](https://arxiv.org/abs/2310.04560)] ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1) 
1. [arXiv 2023.09] **Can LLMs Effectively Leverage Graph Structural Information: When and Why** [[pdf](https://arxiv.org/abs/2309.16595)] ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1) 
1. [arxiv 2023.05] **Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning**. [[pdf](https://scholar.google.com/scholar_url?url=https://www.biorxiv.org/content/biorxiv/early/2023/06/01/2023.05.30.542904.full.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=8390578571473859304&ei=8QRCZbaEHIz5yATYnbjoDQ&scisig=AFWwaebONdO5ia5yjK3p4wA-pOf1)] ![](https://img.shields.io/badge/transformer-DCE7F1)
2. [arxiv 2023.07] **Meta-Transformer: A Unified Framework for Multimodal Learning**. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2307.10802&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=11077145075852511910&ei=cgZCZZG8AcaKywSmq6XgCw&scisig=AFWwaeaAZA6AHtPdRvX0JnyNhv1F)] [[code](https://github.com/invictus717/MetaTransformer)] ![](https://img.shields.io/badge/transformer-DCE7F1)
3. [arXiv 2023.08] **Natural language is all a graph needs**. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2308.07134&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=14935989239849530960&ei=WgdCZbaJFtqk6rQPkv-skA8&scisig=AFWwaeZJKMJktGJOJmeusMs1l5k1)] [[code](https://github.com/agiresearch/InstructGLM)] ![](https://img.shields.io/badge/Graph_to_token_+_Flan_T5,LLaMA-DCE7F1) ![](https://img.shields.io/badge/MLM,LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1) 
4. [arxiv 2023.08] **Evaluating large language models on graphs: Performance insights and comparative analysis**. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2308.11224&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=13367291863109264530&ei=JQlCZcecFqKQ6rQPwPS0qAw&scisig=AFWwaebcL4UoKZKs-b3HfIKmzeoB)] [[code](https://github.com/ayame1006/llmtograph)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs,Vicuna-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
5. [arxiv 2023.05] **Can language models solve graph problems in natural language?** [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/2305.10037&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=10660384245119063422&ei=UgpCZeaYLKKQ6rQPwPS0qAw&scisig=AFWwaeb83Q4qbndJ3rQeda6SsHVD)] [[code](https://github.com/arthur-heng/nlgraph)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
6. [arxiv 2023.10] **Graphtext: Graph reasoning in text space**. [[pdf](https://arxiv.org/pdf/2310.01089.pdf)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
7. [arxiv 2023.07] **Can large language models empower molecular property prediction?** [[pdf](https://arxiv.org/pdf/2307.07443.pdf)] [[code](https://github.com/chnq/llm4mol)] ![](https://img.shields.io/badge/Graph_to_text_+_GPTs-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning-D8D0E1)
8. [arxiv 2023.05] **Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking** [[pdf](https://arxiv.org/pdf/2305.15066.pdf)] ![](https://img.shields.io/badge/Graph_to_text_+_GPT_3-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning_+_Automatic_Prompt_Tuning-D8D0E1)
9. [arxiv 2023.07] **Exploring the potential of large language models (llms) in learning on graphs** [[pdf](https://arxiv.org/pdf/2307.03393.pdf)] [[code](https://github.com/CurryTang/Graph-LLM)] ![](https://img.shields.io/badge/Graph_to_text_+_Bert,sBert,LLaMa,GPTs-DCE7F1) ![](https://img.shields.io/badge/LM,MLM-EAD8D9) ![](https://img.shields.io/badge/Manual_Prompt_Tuning_+_Automatic_Prompt_Tuning-D8D0E1)

## 3. GNN+LLM-based Papers
1. [arXiv 2023.10] **Label-free Node Classification on Graphs with Large Language Models (LLMs)** [[pdf](https://arxiv.org/abs/2310.04668)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) 
1. [arXiv_2023.09] **One for All: Towards Training One Graph Model for All Classification Tasks** [[pdf](https://arxiv.org/abs/2310.00149)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) 
1. [arXiv_2023.09] **Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs**.[[pdf](https://arxiv.org/abs/2309.02848)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) 
1. [arxiv 2023.08] **Simteg: A frustratingly simple approach improves 
textual graph learning**. [[pdf](https://arxiv.org/pdf/2308.02565.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM,TTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--EfficientFT-D8D0E1)
1. [arxiv 2023.05] **Explanations as features: Llm-based features for text-attributed graphs**. [[pdf](https://arxiv.org/pdf/2305.19523.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9)  ![](https://img.shields.io/badge/Tuning--free_Prompting+Parameter--Efficient_FT-D8D0E1)
1. [ICLR 2022] **Node feature extraction by self-supervised multi-scale neighborhood prediction**. [[pdf](https://arxiv.org/pdf/2111.00064.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/vanilla--FT-D8D0E1)
1. [arxiv 2023.04] **Train your own GNN teacher: Graph-aware distillation on textual graphs**. [[pdf](https://arxiv.org/pdf/2304.10668.pdf)] ![](https://img.shields.io/badge/GNN--centric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [NIPS 2021] **Graphformers: Gnn-nested transformers for representation learning on textualgraph**. [[pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/f18a6d1cde4b205199de8729a6637b42-Paper.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Vanilla--FT-D8D0E1)
1. [ICLR 2023] **Learning on large-scale text-attributed graphs via variational inference**. [[pdf](https://arxiv.org/pdf/2210.14709.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Vanilla_FT-D8D0E1)
1. [arxiv 2023.05] **Congrat: Self-supervised contrastive pretraining for joint graph and text embeddings**. [[pdf](https://arxiv.org/pdf/2305.14321.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [SIGIR 2023] **Augmenting low-resource text classification with graph-grounded pre-training and prompting**. [[pdf](https://arxiv.org/pdf/2305.03324.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/GTCL-EAD8D9)  ![](https://img.shields.io/badge/Prompt--Tuning-D8D0E1)
1. [arxiv 2020.08] **Graph-based modeling of online communities for fake news detection**. [[pdf](hhttps://arxiv.org/pdf/2008.06274.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [EMNLP 2021] **Text2mol: Cross-modal molecule retrieval with natural language queries**.  [[pdf](https://aclanthology.org/2021.emnlp-main.47.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [arxiv 2022.09] **A molecular multimodal foundation model associating molecule graphs with natural language**. [[pdf](https://arxiv.org/pdf/2209.05481.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [arxiv 2022.12] **Multi-modal molecule structure-text model for text-based retrieval and editing**.  [[pdf](https://arxiv.org/pdf/2212.10789.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTC-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [PMLR 2023] **Enhancing activity prediction models in drug discovery with the ability to understand human language**.  [[pdf](https://arxiv.org/pdf/2303.03363.pdf)] ![](https://img.shields.io/badge/Symmetric-DCE7F1) ![](https://img.shields.io/badge/MLM+GTCL-EAD8D9)  ![](https://img.shields.io/badge/Parameter--Efficient_FT-D8D0E1)
1. [arxiv 2023.04] **Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt**.  [[pdf](https://arxiv.org/pdf/2304.11116.pdf)] ![](https://img.shields.io/badge/LLM--centric-DCE7F1) ![](https://img.shields.io/badge/LM-EAD8D9)  ![](https://img.shields.io/badge/Tuning--free_Prompting+Vanilla_FT-D8D0E1)




## Contributors

We thank all the contributors to this list. And more contributions are very welcome.

<a href="https://github.com/BUPT-GAMMA/GFMpapers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BUPT-GAMMA/GFMpapers" />
</a>

