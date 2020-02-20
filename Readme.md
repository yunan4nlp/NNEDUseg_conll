# RST EDU Segmenter #

This is the reimplement code[BERT + BiLSTM + CRF] for the paper:
[ToNy: Contextual embeddings for accurate multilingual discourse segmentation of full documents](https://www.aclweb.org/anthology/W19-2715.pdf)

We use CRF in our segmenter, and the original paper without CRF.

## Environment ##


Pytorch 1.0.0:
*https://download.pytorch.org/whl/cpu/torch_stable.html*

AllenNLP [CRF]:
*https://github.com/allenai/allennlp*

## Data ##

DISRPT2019 Shared Task</br>
*https://github.com/disrpt/sharedtask2019*

We only prepare the RST Tree Bank.</br>
*https://catalog.ldc.upenn.edu/LDC2002T07*



## External Resource ##
BERT</br>
*https://github.com/huggingface/transformers*</br>



## Performance ##

| eng.rst.rstdt |  P   |  R    | F   |
| :--: | :--: | :--: | :--: |
| Orignal Paper Test | 95.29   | 96.81 | 96.04 |
| Our Test | 95.56 | 96.46 | 96.01 |

