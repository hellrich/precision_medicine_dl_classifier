# precision_medicine_dl_classifier
Document classification experiments on TREC precision medicine data

Final design is an adaption of [Yang et al. (NAACL 2016): "Hierarchical Attention Networks for Document Classification"](https://aclweb.org/anthology/N16-1174) with additional structured information. 
Result on 2017 PubMed during 10-fold crossvalidation were 78.14 (versus 74.96 for logistic regression with BoW and structured information); results on 2018 data were 75.98 (74.40 baseline).

Training code and models can be found inside precision_medicine_scripts, Notebooks were used during development and for evaluation
