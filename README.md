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
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Source:
```

#### Webpages

- **CC-Stories  2018-6 | Not | EN | CI | [Paper](https://arxiv.org/pdf/1806.02847.pdf) | [Github](https://github.com/tensorflow/models/tree/archive/research/lm_commonsense) | [Dataset](https://huggingface.co/datasets/spacemanidol/cc-stories)**
  - Publisher: Google Brain
  - Size: 31 GB
  - License: -
  - Source: Common Crawl

- **CC100  2020-7 | All | Multi (100) | CI | [Paper](https://aclanthology.org/2020.acl-main.747.pdf) | [Dataset](https://huggingface.co/datasets/cc100)**
  - Publisher: Facebook AI
  - Size: 2.5 TB
  - License: Common Crawl Terms of Use
  - Source: Common Crawl

#### Language Texts

- **ANC  2003-X | All | EN | HG | [Website](https://anc.org/)**
  - Publisher: The US National Science Foundation et al.
  - Size: -
  - License: -
  - Source: American English texts

#### Books

#### Academic Materials

#### Code <a id="code01"></a>

#### Parallel Corpus

#### Social Media

#### Encyclopedia

#### Multi-category

### Domain-specific Pre-training Corpora
Domain-specific pre-training corpora are LLM datasets customized for specific fields or topics. The type of corpus is typically employed in the incremental pre-training phase of LLMs. **Corpora are classified based on data domains.**

**Dataset information format：**

```
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Source:
  - Category:
  - Domain:
```

#### Financial <a id="financial01"></a>

#### Medical <a id="medical01"></a>

#### Other <a id="other01"></a>

## Instruction Fine-tuning Datasets
The instruction fine-tuning datasets consists of a series of text pairs comprising “instruction inputs” and “answer outputs.” “Instruction inputs” represent requests made by humans to the model. There are various types of instructions, such as classification, summarization, paraphrasing, etc. “Answer outputs” are the responses generated by the model following the instruction and aligning with human expectations.

### General Instruction Fine-tuning Datasets
General instruction fine-tuning datasets contain one or more instruction categories with no domain restrictions, primarily aiming to enhance the instruction-following capability of LLMs in general tasks. **Datasets are classified based on construction methods.**

**Dataset information format：**

```
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Source:
  - Instruction Category:
```

#### Human Generated Datasets (HG)

#### Model Constructed Datasets (MC)

#### Collection and Improvement of Existing Datasets (CI)

#### HG & CI

#### HG & MC

#### CI & MC

#### HG & CI & MC

### Domain-specific Instruction Fine-tuning Datasets
The domain-specific instruction fine-tuning datasets are constructed for a particular domain by formulating instructions that encapsulate knowledge and task types closely related to that domain. 

**Dataset information format：**

```
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Source:
  - Instruction Category:
  - Domain:
```

#### Medical <a id="medical02"></a>

#### Code <a id="code02"></a>

#### Legal

#### Math <a id="math01"></a>

#### Education

#### Other <a id="other02"></a>

## Preference Datasets
Preference datasets are collections of instructions that provide preference evaluations for multiple responses to the same instruction input.

### Preference Evaluation Methods
The preference evaluation methods for preference datasets can be categorized into voting, sorting, scoring, and other methods. **Datasets are classified based on preference evaluation methods.**

**Dataset information format：**

```
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Domain:
  - Instruction Category: 
  - Preference Evaluation Method: 
  - Source: 
```

#### Vote

#### Sort

#### Score

#### Other <a id="other03"></a>

## Evaluation Datasets
Evaluation datasets are a carefully curated and annotated set of data samples used to assess the performance of LLMs across various tasks.**Datasets are classified based on evaluation domains.**

**Dataset information format：**

```
- Dataset name  Release Time | Public or Not | Language | Construction Method | Paper | Github | Dataset | Website
  - Publisher:
  - Size:
  - License:
  - Question Type: 
  - Evaluation Method: 
  - Focus: 
  - Numbers of Evaluation Categories/Subcategories: 
  - Evaluation Category: 
```

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
Diverging from instruction fine-tuning datasets, we categorize text datasets dedicated to natural language tasks before the widespread adoption of LLMs as traditional NLP datasets.

**Dataset information format：**

```
- Dataset name  Release Time | Language | Paper | Github | Dataset | Website
  - Publisher:
  - Train/Dev/Test/All Size: 
  - License:
  - Number of Entity Categories: (NER Task)
  - Number of Relationship Categories: (RE Task)
```

### Question Answering
The task of question-answering requires the model to utilize its knowledge and reasoning capabilities to respond to queries based on provided text (which may be optional) and questions. 

#### Reading Comprehension
The task of reading comprehension entails presenting a model with a designated text passage and associated questions, prompting the model to understand the text for the purpose of answering the questions.

##### Selection & Judgment

##### Cloze Test

##### Answer Extraction

##### Unrestricted QA

#### Knowledge QA
In the knowledge QA task, models respond to questions by leveraging world knowledge, common sense, scientific insights, domain-specific information, and more.

#### Reasoning QA
The focal point of reasoning QA tasks is the requirement for models to apply abilities such as logical reasoning, multi-step inference, and causal reasoning in answering questions. 

### Recognizing Textual Entailment
The primary objective of tasks related to Recognizing Textual Entailment (RTE) is to assess whether information in one textual segment can be logically inferred from another. 

### Math <a id="math02"></a>
Mathematical assignments commonly involve standard mathematical calculations, theorem validations, and mathematical reasoning tasks, among others.

### Coreference Resolution
The core objective of tasks related to coreference resolution is the identification of referential relationships within texts.

### Sentiment Analysis
The sentiment analysis task, commonly known as emotion classification, seeks to analyze and deduce the emotional inclination of provided texts, commonly categorized as positive, negative, or neutral sentiments.

### Semantic Matching
The task of semantic matching entails evaluating the semantic similarity or degree of correspondence between two sequences of text. 

### Text Generation
The narrow definition of text generation tasks is bound by provided content and specific requirements. It involves utilizing benchmark data, such as descriptive terms and triplets, to generate corresponding textual descriptions.

### Text Translation
Text translation involves transforming text from one language to another.

### Text Summarization
The task of text summarization pertains to the extraction or generation of a brief summary or headline from an extended text to encapsulate its primary content. 

### Text Classification
Text classification tasks aim to assign various text instances to predefined categories, comprising text data and category labels as pivotal components.

### Text Quality Evaluation
The task of text quality evaluation, also referred to as text correction, involves the identification and correction of grammatical, spelling, or language usage errors in text.

### Text-to-Code
The Text-to-Code task involves models converting user-provided natural language descriptions into computer-executable code, thereby achieving the desired functionality or operation.

### Named Entity Recognition
The Named Entity Recognition (NER) task aims to discern and categorize named entities within a given text.

### Relation Extraction
The endeavor of Relation Extraction (RE) necessitates the identification of connections between entities within textual content.  This process typically includes recognizing and labeling pertinent entities, followed by the determination of the specific types of relationships that exist among them.

### Multitask <a id="multitask02"></a>
Multitask datasets hold significance as they can be concurrently utilized for different categories of NLP tasks.
