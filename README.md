# precision_medicine_dl_classifier
Document classification experiments on TREC precision medicine data, used as a feature for information retrieval.

Final design is an adaption of [Yang et al. (NAACL 2016): "Hierarchical Attention Networks for Document Classification"](https://aclweb.org/anthology/N16-1174) with additional structured information (vectors representing entities/keywords) added to the document level representations. 
Accuracy on 2017 PubMed during 10-fold crossvalidation was 78.14 (versus 74.96 for logistic regression with BoW and structured information); on 2018 data 75.98 (74.40 baseline) could be achieved.

Training code and models can be found inside precision_medicine_scripts, Notebooks were used during development and for evaluation.
