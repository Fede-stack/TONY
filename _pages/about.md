---
permalink: /
title: "Home"
excerpt: "About TONY"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
sidebar:
  nav: "docs"
---

{% include toc %}

# About TONY

TONY (**TO**lkit for **N**LP in Ps**Y**cology) is a Python package for Natural Language Processing (NLP) applied to mental health-related text data.

The package combines two complementary approaches. First, it employs traditional Linguistic-based analyses to extract linguistic markers and compute standard metrics that identify patterns in text. Second, it leverages transformer-based analyses using deep learning models to provide advanced predictions on emotions, psychological states, and clinical traits.

This combination allows researchers and practitioners to analyze mental health-related texts using both interpretable methods and state-of-the-art techniques, offering flexibility for research and clinical applications.

 

<img src="https://raw.githubusercontent.com/Fede-stack/TONY_py/main/images/overview.png" alt="" width="2000">


## Getting Started

Overview of how to begin using the package.

### Installation

```python
!pip install git+https://github.com/Fede-stack/TONYpy.git
```

### Import the package (and have fun playing with it :) )

```python
import TONY
```
## How to cite the package

```bibtex
@inproceedings{ravenda-etal-2026-tony,
    title = "{TONY}: an open-source {TO}olkit for Nlp in ps{Y}chology",
    author = "Ravenda, Federico  and
      Ravenda, Sofia Irene  and
      Karpenko, Volodymyr  and
      Montagnani, Daniele  and
      Raballo, Andrea  and
      Mira, Antonietta",
    editor = "Durrett, Greg  and
      Jian, Ping",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-demo.65/",
    doi = "10.18653/v1/2026.acl-demo.65",
    pages = "660--671",
    ISBN = "979-8-89176-392-0",
    abstract = "The growing demand for Mental Health (MH) services highlights the need for scalable computational tools, yet progress in computational psychology is hindered by scarce sensitive data, complex assessment procedures, and high technical barriers. While language is a well-established marker of different MH conditions, existing NLP solutions are often fragmented, closed-source, or difficult to use, limiting their adoption in interdisciplinary research.We present TONY, an open-source, python TOolkit for NLP in clinical psYchology. TONY bridges traditional psycholinguistic analysis and modern NLP by combining interpretable lexical features with state-of-the-art lightweight transformer models within a unified and easy-to-use framework. This hybrid approach enables robust and transparent text analysis without relying on large-scale models or closed-source software.TONY is designed for researchers and practitioners working at the intersection of NLP and MH, facilitating collaboration across disciplines. Compared to the few existing systems, TONY offers a more comprehensive and exhaustive solution, reducing the barrier to entry through a unified, modular, and reproducible pipeline that integrates classical and neural approaches in a single open framework. The toolkit is released under an open-source license and is evaluated through multiple MH{--}related datasets, demonstrating its flexibility and effectiveness in low-resource settings"
}
```
