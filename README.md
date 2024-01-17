# Awesome-LLMs-Datasets
Summarize existing representative LLMs text datasets across five dimensions: **Pre-training Corpora, Fine-tuning Instruction Datasets, Preference Datasets, Evaluation Datasets, and Traditional NLP Datasets**. (Regular updates)

The paper **"The Roots of the AI Tree: A Survey of Large Language Model Datasets"** will be released soon.（2024/2）

## Dataset Information Module
The following is a summary of the dataset information module.

- Corpus/Dataset name
- Publisher
- Release Time
  - “X” indicates unknown month. 
- Size
- Public or Not
  - “All” indicates full open source; 
  - “Partial” indicates partially open source; 
  - “Not” indicates not open source. 
- License
- Language
  - “EN” indicates English;
  - “ZH” indicates Chinese;
  - “AR” indicates Arabic;
  - “ES” indicates Spanish;
  - “RU” indicates Russian;
  - “DE” indicates German;
  - “PL” indicates Programming Language;
  - “Multi” indicates Multilingual, and the number in parentheses indicates the number of languages included. 
- Construction Method
  - “HG” indicates Human Generated Corpus/Dataset;
  - “MC” indicates Model Constructed Corpus/Dataset;
  - “CI” indicates Collection and Improvement of Existing Corpus/Dataset.
- Category
- Source
- Domain
- Instruction Category
- Preference Evaluation Method
  - “VO” indicates Vote;
  - “SO” indicates Sort;
  - “SC” indicates Score;
  - “-H” indicates Conducted by Humans;
  - “-M” indicates Conducted by Models.
- Question Type
  - “SQ” indicates Subjective Questions;
  - “OQ” indicates Objective Questions;
  - “Multi” indicates Multiple Question Types.
- Evaluation Method
  - “CE” indicates Code Evaluation;
  - “HE” indicates Human Evaluation;
  - “ME” indicates Model Evaluation.
- Focus
- Numbers of Evaluation Categories/Subcategories
- Evaluation Category
- Number of Entity Categories (NER Task)
- Number of Relationship Categories (RE Task)

## Changelog
- （2024/1/17）Create the **Awesome-LLMs-Datasets** dataset repository.

## Table of Contents
- **[Pre-training Corpora](#pre-training-corpora)**
  - [General Pre-training Corpora](#general-pre-training-corpora)
    - [Webpages](#webpages)
    - [Language Texts](#language-texts)
    - [Books](#books)
    - [Academic Materials](#academic-materials)
    - [Code](#code01)
    - [Parallel Corpus](#parallel-corpus)
    - [Social Media](#social-media)
    - [Encyclopedia](#encyclopedia)
    - [Multi-category](#multi-category)
  - [Domain-specific Pre-training Corpora](#domain-specific-pre-training-corpora)
    - [Financial](#financial01)
    - [Medical](#medical01)
    - [Other](#other01)
- **[Instruction Fine-tuning Datasets](#instruction-fine-tuning-datasets)**
  - [General Instruction Fine-tuning Datasets](#general-instruction-fine-tuning-datasets)
    - [Human Generated Datasets (HG)](#human-generated-datasets-hg)
    - [Model Constructed Datasets (MC)](#model-constructed-datasets-mc)
    - [Collection and Improvement of Existing Datasets (CI)](#collection-and-improvement-of-existing-datasets-ci)
    - [HG & CI](#hg--ci)
    - [HG & MC](#hg--mc)
    - [CI & MC](#ci--mc)
    - [HG & CI & MC](#hg--ci--mc)
  - [Domain-specific Instruction Fine-tuning Datasets](#domain-specific-instruction-fine-tuning-datasets)
    - [Medical](#medical02)
    - [Code](#code02)
    - [Legal](#legal)
    - [Math](#math01)
    - [Education](#education)
    - [Other](#other02)
- **[Preference Datasets](#preference-datasets)**
  - [Preference Evaluation Methods](#preference-evaluation-methods)
    - [Vote](#vote)
    - [Sort](#sort)
    - [Score](#score)
    - [Other](#other03)
- **[Evaluation Datasets](#evaluation-datasets)**
  - [General](#general)
  - [Exam](#exam)
  - [Subject](#subject)
  - [NLU](#nlu)
  - [Reasoning](#reasoning)
  - [Knowledge](#knowledge)
  - [Long Text](#long-text)
  - [Tool](#tool)
  - [Agent](#agent)
  - [Code](#code03)
  - [OOD](#ood)
  - [Law](#law)
  - [Medical](#medical03)
  - [Financial](#financial02)
  - [Social Norms](#social-norms)
  - [Factuality](#factuality)
  - [Evaluation](#evaluation)
  - [Multitask](#multitask01)
  - [Multilingual](#multilingual)
  - [Other](#other04)
- **[Traditional NLP Datasets](#traditional-nlp-datasets)**
  - [Question Answering](#question-answering)
    - [Reading Comprehension](#reading-comprehension)
      - [Selection & Judgment](#selection--judgment)
      - [Cloze Test](#cloze-test)
      - [Answer Extraction](#answer-extraction)
      - [Unrestricted QA](#unrestricted-qa)
    - [Knowledge QA](#knowledge-qa)
    - [Reasoning QA](#reasoning-qa)
  - [Recognizing Textual Entailment](#recognizing-textual-entailment)
  - [Math](#math02)
  - [Coreference Resolution](#coreference-resolution)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Semantic Matching](#semantic-matching)
  - [Text Generation](#text-generation)
  - [Text Translation](#text-translation)
  - [Text Summarization](#text-summarization)
  - [Text Classification](#text-classification)
  - [Text Quality Evaluation](#text-quality-evaluation)
  - [Text-to-Code](#text-to-code)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Relation Extraction](#relation-extraction)
  - [Multitask](#multitask02)

## Pre-training Corpora
The pre-training corpora are large collections of text data used during the pre-training process of LLMs.

### General Pre-training Corpora
The general pre-training corpora are large-scale datasets composed of extensive text from diverse domains and sources. Their primary characteristic is that the text content is not confined to a single domain, making them more suitable for training general foundational models. **Corpora are classified based on data categories.**

**Dataset information format：**

```
Dataset name  Release Time | Public or Not | Language | Construction Method
Paper | Github | Dataset | Website
(1) Publisher:
(2) Size:
(3) License:
(4) Source:
```

#### Webpages

- CC-Stories  2018-6 | Not | EN | CI
  - [Paper](https://arxiv.org/pdf/1806.02847.pdf) | [Github](https://github.com/tensorflow/models/tree/archive/research/lm_commonsense) | [Dataset](https://huggingface.co/datasets/spacemanidol/cc-stories)
  - (1) Publisher: Google Brain
  - (2) Size: 31 GB
  - (3) License: -
  - (4) Source: Common Crawl

-
#### Language Texts

#### Books

#### Academic Materials

#### Code <a id="code01"></a>

#### Parallel Corpus

#### Social Media

#### Encyclopedia

#### Multi-category

### Domain-specific Pre-training Corpora

#### Financial <a id="financial01"></a>

#### Medical <a id="medical01"></a>

#### Other <a id="other01"></a>

## Instruction Fine-tuning Datasets

### General Instruction Fine-tuning Datasets

#### Human Generated Datasets (HG)

#### Model Constructed Datasets (MC)

#### Collection and Improvement of Existing Datasets (CI)

#### HG & CI

#### HG & MC

#### CI & MC

#### HG & CI & MC

### Domain-specific Instruction Fine-tuning Datasets

#### Medical <a id="medical02"></a>

#### Code <a id="code02"></a>

#### Legal

#### Math <a id="math01"></a>

#### Education

#### Other <a id="other02"></a>

## Preference Datasets

### Preference Evaluation Methods

#### Vote

#### Sort

#### Score

#### Other <a id="other03"></a>

## Evaluation Datasets

### General

### Exam

### Subject

### NLU

### Reasoning

### Knowledge

### Long Text

### Tool

### Agent

### Code <a id="code03"></a>

### OOD

### Law

### Medical <a id="medical03"></a>

### Financial <a id="financial02"></a>

### Social Norms

### Factuality

### Evaluation

### Multitask <a id="multitask01"></a>

### Multilingual

### Other <a id="other04"></a>

## Traditional NLP Datasets

### Question Answering

#### Reading Comprehension

##### Selection & Judgment

##### Cloze Test

##### Answer Extraction

##### Unrestricted QA

#### Knowledge QA

#### Reasoning QA

### Recognizing Textual Entailment

### Math <a id="math02"></a>

### Coreference Resolution

### Sentiment Analysis

### Semantic Matching

### Text Generation

### Text Translation

### Text Summarization

### Text Classification

### Text Quality Evaluation

### Text-to-Code

### Named Entity Recognition

### Relation Extraction

### Multitask <a id="multitask02"></a>
