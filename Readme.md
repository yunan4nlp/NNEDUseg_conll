# RST EDU Segmenter #

This is the reimplement code[BERT + BiLSTM] for the paper:
[ToNy: Contextual embeddings for accurate multilingual discourse segmentation of full documents](https://www.aclweb.org/anthology/W19-2715.pdf)


## Environment ##


Pytorch 1.0.0:
*https://download.pytorch.org/whl/cpu/torch_stable.html*

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
| Our Test | 95.49 | 96.55 | 96.02 |

