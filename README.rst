=======================================
Learning Optimal Features via Partial Invariance
=======================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:target: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:alt: License

.. image:: https://img.shields.io/badge/Preprint-ArXiv-blue.svg
	:target: https://arxiv.org/abs/2301.12067
	:alt: ArXiv

Official implementation of the experiments in the paper `"**Learning Optimal Features via Partial Invariance**" <https://arxiv.org/abs/2301.12067>`_ (AAAI'23). 

Each folder contains a different experiment. Please follow the instructions 
in the respective folder on how to run the experiments and reproduce the results. 
`This repository <https://github.com/MoulikChoraria/IbtihalFerwana/pirm>`_ is implemented in `PyTorch <https://pytorch.org/>`_.



Browsing the experiments
========================
Parts of the code are adapted from repositories  `here <https://github.com/Kel-Lu/time-waits-for-no-one>`_ [2]_  and `here <https://github.com/Weixin-Liang/MetaShift/>`_ [3]_. The folder structure is the following:

*    ``synthetic_experiment_pirm``: The `jupyter notebook <https://github.com/IbtihalFerwana/pirm/blob/main/synthetic_experiment_pirm.ipynb>`_ contains the code for synthetic experiment to demonstrate suppression of non-invariant features via IRM.

*    ``linear_regression_pirm``: The `jupyter notebook <https://github.com/IbtihalFerwana/pirm/blob/main/linear_regression_pirm.ipynb>`_ contains the experiments for replicating the results pertaining to the linear regression setting on the House Prices Dataset.

*    ``language``: The `folder <https://github.com/IbtihalFerwana/pirm/tree/main/language>`_ contains the experiments for demonstrating the performance benefits of partitioning on language classification tasks using deep language models.

*    ``metashift``: The `folder <https://github.com/IbtihalFerwana/pirm/tree/main/metashift>`_ contains the experiments for demonstrating the performance benefits of partitioning on image classification tasks on the MetaShift dataset, using Resnets.

Summary
==========================


Learning models that are robust to test-time distribution shifts is a key concern in domain generalization, with Invariant Risk Minimization (IRM) being one particular framework that aims to learn deep invariant features from multiple domains. A key assumption for its success requires that the underlying causal mechanisms/features remain invariant across domains and the true invariant features be sufficient to learn the optimal predictor. In practical problem settings, these assumptions are often not satisfied, which leads to IRM learning a sub-optimal predictor for that task. In this work, we propose the notion of partial invariance as a relaxation of the IRM framework. We demonstrate that learning invariant features from only a subset of available domains can often yield better predictors. We conduct several experiments, both in linear settings as well as with classification tasks in language and images with deep models, to support our findings.  


Citing
======
If you use this code, please cite [1]_:

*BibTeX*:: 

  @misc{ChorariaFMV2023,
  doi = {10.48550/ARXIV.2301.12067},
  url = {https://arxiv.org/abs/2301.12067},
  author = {Choraria, Moulik and Ferwana, Ibtihal and Mani, Ankur and Varshney, Lav R.},
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences},
  title = {Learning Optimal Features via Partial Invariance},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}}
  
References
==========

.. [1] Moulik Choraria, Ibtihal Ferwana, Ankur Mani, and Lav R. Varshney. **Learning Optimal Features via Partial Invariance**, To be presented at the 37th *AAAI Conference on Artificial Intelligence*, 2023.
.. [2] Kelvin Luu, Daniel Khashabi, Suchin Gururangan, Karishma Mandyam, Noah A. Smith, **Time Waits for No One! Analysis and Challenges of Temporal Misalignment**, arXiv, July 1, 2022.
.. [3] Weixin Liang, James Zou. “MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts” , *International Conference on Learning Representations (ICLR)*, 2022.
