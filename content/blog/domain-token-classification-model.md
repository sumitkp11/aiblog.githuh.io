---
title: Domain specific Token Classification Model
description: Let’s see the following points covered in the course provided by Nvidia's Deep Learning Institue on Transformers-based Natural Language Processing
date: 2026-01-21
tags: ["nvidia", "transformers", "model", "token"]
---
## Domain specific token classification model
In the following lines, we will see how to fine-tine a pre-trained language model to perform token classification for specific domains.
We will develop an NER model that finds disease names in medical disease abstracts.

We are going to use the NCBI dataset, which contains set of 793 PubMed abstracts, annotated by 14 annotators.

### Download Data
```
import os
import wget

# set data path
DATA_DIR = "data/NCBI"
os.makedirs(DATA_DIR, exist_ok=True)

with open(f'{DATA_DIR}/NCBI_corpus_testing.txt') as f: 
    sample_text=f.readline()
    
print(sample_text)
```

This shows how the NCBI dataset looks like:

```
9288106	Clustering of missense mutations in the <category="Modifier">ataxia-telangiectasia</category> gene in a <category="SpecificDisease">sporadic T-cell leukaemia</category>.	<category="SpecificDisease">Ataxia-telangiectasia</category> ( <category="SpecificDisease">A-T</category> ) is a <category="DiseaseClass">recessive multi-system disorder</category> caused by mutations in the ATM gene at 11q22-q23 ( ref . 3 ) . The risk of <category="DiseaseClass">cancer</category> , especially <category="DiseaseClass">lymphoid neoplasias</category> , is substantially elevated in <category="Modifier">A-T</category> patients and has long been associated with chromosomal instability . By analysing <category="Modifier">tumour</category> DNA from patients with <category="SpecificDisease">sporadic T-cell prolymphocytic leukaemia</category> ( <category="SpecificDisease">T-PLL</category> ) , a rare <category="DiseaseClass">clonal malignancy</category> with similarities to a <category="SpecificDisease">mature T-cell leukaemia</category> seen in <category="SpecificDisease">A-T</category> , we demonstrate a high frequency of ATM mutations in <category="SpecificDisease">T-PLL</category> . In marked contrast to the ATM mutation pattern in <category="SpecificDisease">A-T</category> , the most frequent nucleotide changes in this <category="DiseaseClass">leukaemia</category> were missense mutations . These clustered in the region corresponding to the kinase domain , which is highly conserved in ATM-related proteins in mouse , yeast and Drosophila . The resulting amino-acid substitutions are predicted to interfere with ATP binding or substrate recognition . Two of seventeen mutated <category="SpecificDisease">T-PLL</category> samples had a previously reported <category="Modifier">A-T</category> allele . In contrast , no mutations were detected in the p53 gene , suggesting that this <category="Modifier">tumour</category> suppressor is not frequently altered in this <category="DiseaseClass">leukaemia</category> . Occasional missense mutations in ATM were also found in <category="Modifier">tumour</category> DNA from patients with <category="SpecificDisease">B-cell non-Hodgkins lymphomas</category> ( <category="SpecificDisease">B-NHL</category> ) and a <category="Modifier">B-NHL</category> cell line . The evidence of a significant proportion of loss-of-function mutations and a complete absence of the normal copy of ATM in the majority of mutated <category="DiseaseClass">tumours</category> establishes somatic inactivation of this gene in the pathogenesis of <category="SpecificDisease">sporadic T-PLL</category> and suggests that ATM acts as a <category="Modifier">tumour</category> suppressor . As constitutional DNA was not available , a putative hereditary predisposition to <category="SpecificDisease">T-PLL</category> will require further investigation . . 
```


```
import re

# use regular expression to find labels
categories=re.findall('<category.*?<\/category>', sample_text)
for sample in categories: 
    print(sample)
```

```
<category="Modifier">ataxia-telangiectasia</category>
<category="SpecificDisease">sporadic T-cell leukaemia</category>
<category="SpecificDisease">Ataxia-telangiectasia</category>
<category="SpecificDisease">A-T</category>
<category="DiseaseClass">recessive multi-system disorder</category>
<category="DiseaseClass">cancer</category>
<category="DiseaseClass">lymphoid neoplasias</category>
<category="Modifier">A-T</category>
<category="Modifier">tumour</category>
<category="SpecificDisease">sporadic T-cell prolymphocytic leukaemia</category>
<category="SpecificDisease">T-PLL</category>
<category="DiseaseClass">clonal malignancy</category>
<category="SpecificDisease">mature T-cell leukaemia</category>
<category="SpecificDisease">A-T</category>
<category="SpecificDisease">T-PLL</category>
<category="SpecificDisease">A-T</category>
<category="DiseaseClass">leukaemia</category>
<category="SpecificDisease">T-PLL</category>
<category="Modifier">A-T</category>
<category="Modifier">tumour</category>
<category="DiseaseClass">leukaemia</category>
<category="Modifier">tumour</category>
<category="SpecificDisease">B-cell non-Hodgkins lymphomas</category>
<category="SpecificDisease">B-NHL</category>
<category="Modifier">B-NHL</category>
<category="DiseaseClass">tumours</category>
<category="SpecificDisease">sporadic T-PLL</category>
<category="Modifier">tumour</category>
<category="SpecificDisease">T-PLL</category>
```

In the following code, we see that the abstract has been broken into sentences. Each sentence is further parsed into words with labels that correspond to the original HTML-styled tags in the dataset.

```
NER_DATA_DIR = f'{DATA_DIR}/NER'
os.makedirs(os.path.join(DATA_DIR, 'NER'), exist_ok=True)

# show downloaded files
!ls -lh $NER_DATA_DIR

!head $NER_DATA_DIR/train.tsv
```

#### output
```
Identification	O
of	O
APC2	O
,	O
a	O
homologue	O
of	O
the	O
adenomatous	B-Disease
polyposis	I-Disease
```

## Preprocess Data
We need to convert these to a format that is compatible with NeMo token classification module.

Script for conversion can be found [here](https://github.com/NVIDIA-NeMo/NeMo/blob/stable/examples/nlp/token_classification/data/import_from_iob_format.py).

```bash
# invoke the conversion script 
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/train.tsv
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/dev.tsv
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/test.tsv

# preview dataset
!head -n 1 $NER_DATA_DIR/text_train.txt
!head -n 1 $NER_DATA_DIR/labels_train.txt
```

#### output
```
Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor . 
O O O O O O O O B-Disease I-Disease I-Disease I-Disease O O 
```






