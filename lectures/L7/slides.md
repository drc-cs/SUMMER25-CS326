---
title: CS 326
separator: <!--s-->
verticalSeparator: <!--v-->

theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.7em; left: 0; width: 60%; position: absolute;">

  # Introduction to Data Science Pipelines
  ## L7

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%;">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Welcome to CS 326
  ## Please check in and enter the provided code.

  </div>
  </div>
  <div class="c2" style="width: 50%; height: auto;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width="100%" height="100%" style="border-radius: 10px;"></iframe>
  </div>
</div>

<!--s-->

# Verbal Clarification from L6

- The Wilcoxon Signed-Rank Test is a non-parametric statistical test used to compare two related samples or matched pairs. It is used when the data does not meet the assumptions of the paired t-test, such as normality.

- The Wilcoxon Rank-Sum Test (also known as the Mann-Whitney U Test) is a non-parametric statistical test used to compare two independent samples. It is used when the data does not meet the assumptions of the independent t-test, such as normality.

<!--s-->

<div class="header-slide">

# Natural Language Processing

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

# Intro Poll
## On a scale of 1-5, how confident are you with **text** methods such as:

- Regular Expressions
- Tokenization
- Bag of Words (BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Markov Chains
- Evaluation of Text Generation Models
- R.A.G.

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Introduction to NLP | Definition and Importance

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.

NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data.

<!--s-->

## NLP Applications

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Applications of NLP

- **Chatbots**: Automate customer service interactions.

- **Sentiment Analysis**: Determine sentiment from text data.

- **Machine Translation**: Translate text from one language to another.

</div>

<div class="c2 col-centered" style="width: 50%">

<div>

<img src="https://amazinum.com/wp-content/uploads/2022/10/blog-NLP-pic1.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Amazinum 2024</span>

</div>

</div>

</div>

<!--s-->

## NLP Impact on Society

NLP is transforming industries by automating tasks, improving data analysis, and enhancing communication. It impacts our daily lives through digital assistants, real-time translation, and personalized content delivery.

- Initial attempts involved rule-based systems (if-then statements).

- Classical methods use statistics (Naive Bayes).

- Deep learning techniques (combined with statistics) have emerged in recent years (LSTM/transformers).

<!--s-->

<div class="header-slide">

# Text Processing

</div>

<!--s-->

<div class="header-slide">

# Regular Expressions

</div>

<!--s-->

## Regular Expressions

Regular expressions (regex) are a powerful tool for working with text data. They allow us to search, match, and manipulate text using a concise and expressive syntax.

You may feel compelled to do basic sting manipulation with Python's built-in string methods. However, regular expressions are much more powerful and flexible. Consider the following example:

> "My phone number is (810) 555-1234."

<!--s-->

## Regular Expressions | Example

> "My phone number is (810)555-1234"

### String Methods

<span class="code-span">phone_number = text.split(" ")[-1]</span>

This method would work for the given example but **not** for "My phone number is (810) 555-1234. Call me!"
or "My phone number is (810) 555-1234. Call me! It's urgent!"

### Regular Expression

<span class="code-span">phone_number = re.search(r'\(\d{3}\)\d{3}-\d{4}', text).group()</span>

This regular expression will match any phone number in the format (810)555-1234, including the additional text above.

<!--s-->

## Regular Expressions | Syntax

Regular expressions are a sequence of characters that define a search pattern. They are used to search, match, and manipulate text strings.

<div style="font-size: 0.8em; overflow-y: scroll; height: 80%;">
  
| Pattern | Description |
|---------|-------------|
| <span class='code-span'>.</span>     | Matches any character except newline |
| <span class='code-span'>^</span>     | Matches the start of a string |
| <span class='code-span'>$</span>     | Matches the end of a string |
| <span class='code-span'>*</span>     | Matches 0 or more repetitions of the preceding element |
| <span class='code-span'>+</span>     | Matches 1 or more repetitions of the preceding element |
| <span class='code-span'>?</span>     | Matches 0 or 1 repetition of the preceding element |
| <span class='code-span'>{n}</span>   | Matches exactly n repetitions of the preceding element |
| <span class='code-span'>{n,}</span>  | Matches n or more repetitions of the preceding element |
| <span class='code-span'>{n,m}</span> | Matches between n and m repetitions of the preceding element |
| <span class='code-span'>[]</span>    | Matches any one of the characters inside the brackets |
| <span class='code-span'> \| </span>     | Matches either the expression before or the expression after the operator |
| <span class='code-span'>()</span>    | Groups expressions and remembers the matched text |
| <span class='code-span'>\d</span>    | Matches any digit (equivalent to <span class='code-span'>[0-9]</span>) |
| <span class='code-span'>\D</span>    | Matches any non-digit character |
| <span class='code-span'>\w</span>    | Matches any word character (equivalent to <span class='code-span'>[a-zA-Z0-9_]</span>) |
| <span class='code-span'>\W</span>    | Matches any non-word character |
| <span class='code-span'>\s</span>    | Matches any whitespace character (spaces, tabs, line breaks) |
| <span class='code-span'>\S</span>    | Matches any non-whitespace character |
| <span class='code-span'>\b</span>    | Matches a word boundary |
| <span class='code-span'>\B</span>    | Matches a non-word boundary |
| <span class='code-span'>\\</span>    | Escapes a special character |

</div>

<!--s-->

## Regular Expressions

Want to practice or make sure your expression works? 

Live regular expression practice: https://regex101.com/

<!--s-->

<div class="header-slide">

# Tokenization

</div>

<!--s-->

## Tokenization

Tokenization is the process of breaking text into smaller units, such as sub-words or words. It is a fundamental step in NLP tasks.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

- **Character Tokenization**: Splitting text into individual characters.

- **Word Tokenization**: Splitting text into individual words or n-grams.

- **Tokenization Libraries**: NLTK, spaCy, scikit-learn.

</div>

<div class="c2" style="width: 50%">

<div>

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*VymfbxfPtuH3MpJl.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Khandelwal 2023</span>

</div>

</div>

</div>

🔥 *NLTK is awesome. It's a great library for beginners!*

<!--s-->

<div class="header-slide">

# Stemming and Lemmatization

</div>

<!--s-->

## Stemming and Lemmatization

Stemming and lemmatization are text normalization techniques that reduce words to their base or root form. They are used to improve text processing and analysis by reducing the vocabulary size.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Stemming

*Reduces words to their base or root form.*

Stemming examples include "running" to "run" and "requested" to "request."

Some disadvantages include over-stemming and under-stemming -- for example, "better" to "bett" instead of "good."

</div>

<div class="c2" style="width: 50%">

### Lemmatization

*Reduces words to their base form using a dictionary.*

Lemmatization examples include "better" to "good" and "requested" to "request."

Some disadvantages include increased computational complexity and slower processing times, and often requires a dictionary or corpus.

</div>

</div>

<!--s-->

<div class="header-slide">

# Stop Words

</div>

<!--s-->

## Stop Words

Stop words are common words that are filtered out during text processing to reduce noise and improve performance. They are typically removed before tokenization.

**Examples** include <span class="code-span">and</span> / <span class="code-span">the</span> / <span class="code-span">or</span>

<!--s-->

## Example Sentence Preprocessing

<div style = "font-size: 0.8em;">

Consider the following sentence:

> "The quick brown fox jumps over the lazy dog."

Tokenized: 

> ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

Lemmatized:

> ["The", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog"]

Stop Words Removed:

> ["quick", "brown", "fox", "jump", "lazy", "dog"]

</div>

<!--s-->

<div class="header-slide">

# Classical NLP Methods

</div>

<!--s-->

## Bag of Words (BoW)

BoW represents text by counting the occurrences of each word, ignoring grammar and order. It's easy to use but lacks context and semantic understanding.

<img src="https://storage.googleapis.com/slide_assets/bow.png" width="100%" style="margin: 0; padding: 0; display: block; border-radius: 10px;">

<!--s-->

## Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a corpus. It is used to represent text data as a vector.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0; font-size: 0.8em;">

### TF-IDF

- **Term Frequency (TF)**: Frequency of a term in a document.

- **Inverse Document Frequency (IDF)**: Importance of a term across a corpus.

</div>

<div class="c2" style="width: 40%; font-size: 0.8em;">

### Calculation

$$ \text{TF-IDF} = \text{TF} \times \text{IDF} $$
$$ \text{IDF} = \log \left( \frac{N}{n} \right) $$
$$ \text{TF} = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}} $$

Where:
  - $f_{t,d}$ is the frequency of term $t$ in document $d$.
  - $N$ is the total number of documents.
  - $n$ is the number of documents containing term $t$.
  - $t'$ is a term in document $d$.
  - $d$ is a document.

</div>

</div>

<!--s-->

<div class="header-slide">

# Text Classification

</div>

<!--s-->

## K-Nearest Neighbors

Once you have a fixed-length vector representation of text, you can use K-Nearest Neighbors (KNN) to classify text by comparing document vectors.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Concept

Classifies text by comparing document vectors.

### Text Representation

Uses BoW or TF-IDF vectors (or more recently, word embeddings).

</div>

<div class="c2" style="width: 50%">

<div>

<img src="https://storage.googleapis.com/slide_assets/bow.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">CITATION</span>

</div>

</div>

</div>

<!--s-->

<div class="header-slide">

# Text Generation

</div>

<!--s-->

## Markov Chains

Markov Chains are probabilistic models used for generating text. By modeling the context of words with historical patterns, Markov chains can simulate text generation processes.

<div class="col-centered">
<img src="https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2F2.bp.blogspot.com%2F-U2fyhOJ7bN8%2FUJsL23oh3zI%2FAAAAAAAADRs%2FwZNWvVR-Jco%2Fs1600%2Ftext-markov.png" width="500" style="margin: 0; padding: 0; display: block; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">awalsh128.blogspot.com</span>
</div>

<!--s-->

## Markov Chains

In the context of text generation, a Markov chain uses a finite set of states (words) and transitions between these states based on probabilities.

### Key Elements

- **States**: Words or tokens in a text.

- **Transition Probabilities**: The probability of moving from one word to another. 

- **Order**: Refers to how many previous states (words) influence the next state.

<!--s-->

## Markov Chains

Consider a simple first-order Markov Chain, which uses the current word to predict the next word.


### Transition Matrix

A transition matrix represents the probabilities of transitioning from each word to possible subsequent words.

### Markov Process (First Order)

$$ P(w_{t+1} | w_t) = P(w_{t+1} | w_t, w_{t-1}, \ldots, w_1) $$

tldr; the probability of the next word depends only on the current word.


<!--s-->

## Markov Chains

Let's say we have the following text:

> "the quick brown fox jumps over the lazy dog."

|  | the | quick | brown | fox | jumps | over | lazy | dog |
|--------------|-------|---------|---------|-------|---------|--------|--------|-------|
| the        | 0.0   | 0.5     | 0.0     | 0.0   | 0.0     | 0.0    | 0.5    | 0.0   |
| quick      | 0.0   | 0.0     | 1.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| brown      | 0.0   | 0.0     | 0.0     | 1.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| fox        | 0.0   | 0.0     | 0.0     | 0.0   | 1.0     | 0.0    | 0.0    | 0.0   |
| jumps      | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 1.0    | 0.0    | 0.0   |
| over       | 1.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| lazy       | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 1.0   |
| dog        | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |

Using these probabilities, you can generate new text by predicting each subsequent word based on the current word.

<!--s-->

## Markov Chains

Increasing the order allows the model to depend on more than one preceding word, creating more coherent and meaningful sentences.

### Second-Order Markov Chain Example

Given bi-grams:

> "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"

The transition probability now depends on pairs of words:

$$ P(w_3 | w_1, w_2) $$

This provides better context for the generated text, but can also reduce flexibility and introduction of new combinations.

<!--s-->

<div class="header-slide">

# Evaluation of Text Generation Models

</div>

<!--s-->

## Evaluating Text Generation Models

Text generation models are evaluated based on their ability to predict the next word in a sentence. With natural language processing, especially with more advanced models, there are several metrics used to evaluate the quality of generated text:

- **Perplexity**: Measures how well a probability model predicts a sample.
- **BLEU Score**: Measures the quality of machine-translated text.
- **ROUGE Score**: Measures the quality of text summarization.

<!--s-->

## Evaluation | Perplexity

Perplexity measures how well a probability model predicts a sample. A lower perplexity indicates better predictive performance. Perplexity is defined as: 

$$ \text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_{i-1})} $$

Where:
- $ N $ is the number of words in the sample.
- $ P(w_i | w_{i-1}) $ is the probability of word $w_i $ given the previous word $w_{i-1}$.

Intuitively, perplexity measures how surprised a model is by the next word in a sentence.

<!--s-->

## Evaluation | BLEU Score

BLEU is a metric used to evaluate the quality of machine-translated text. It compares the machine-translated text to a reference translation. The BLEU score ranges from 0 to 1.

$$ \text{BLEU} = \text{BP} \times \exp \left( \sum_{n=1}^{N} w_n \log p_n \right) $$

Where:
- $ \text{BP} $ is the brevity penalty.
- $ w_n $ is the weight for n-grams.
- $ p_n $ is the precision for n-grams.

BLEU is more precise and conservative, focusing on word accuracy in translation with a focus on precision.

<!--s-->

## Evaluation | ROUGE Score

ROUGE is a metric used to evaluate the quality of text summarization. It compares the generated summary to a reference summary. The ROUGE score ranges from 0 to 1.

$$ \text{ROUGE} = \frac{\text{Number of Overlapping Words}}{\text{Total Number of Words in Reference Summary}} $$

ROUGE focuses on recall (or the number of overlapping words) in summarization.

<!--s-->

## Summary

- Regular expressions are a powerful tool for working with text data.
- Tokenization is the process of breaking text into smaller units.
- Stemming and lemmatization are text normalization techniques.
- Bag of Words (BoW) represents text by counting word occurrences.
- Term Frequency-Inverse Document Frequency (TF-IDF) evaluates word importance in a document.
- Markov Chains are probabilistic models for generating text.
- Text generation models are evaluated based on perplexity, BLEU, and ROUGE scores.

<!--s-->

<div class="header-slide">

# R.A.G.

</div>

<!--s-->

## Background

In this course, we have explored traditional machine learning models and not discussed many state-of-the-art approaches. For many tasks, classical models are usually very effective and serve as steadfast baselines for nearly any data science project. However, for data science tasks involving text data (e.g., information retrieval, text summarization, question-answering, etc.) recent advances far surpass the capabilities of traditional models.

So now we will discuss a modern NLP approach called Retrieval-Augmented Generation (RAG), which combines the strengths of information retrieval (IR) systems with large language models (LLMs) to create a powerful tool for working with text data in a variety of applications. 

Learning RAG will also introduce longstanding concepts that are critical for modern NLP applications -- including **tokenization**, **chunking**, **embedding**, **vector storage**, **vector retrieval**, and **LLM text generation**.

<!--s-->

## Motivation | RAG

Large language models (LLMs) have revolutionized natural language processing (NLP) by achieving state-of-the-art performance on a wide range of tasks. We will discuss LLMs in more detail later in this lecture. However, for now it's important to note that modern LLMs have some severe limitations, including:

- **Inability to (natively) access external knowledge**
- **Hallucinations** (generating text that is not grounded in reality)

Retrieval-Augmented Generation (RAG) is an approach that addresses these limitations by combining the strengths of information retrieval systems with LLMs.

<!--s-->

## Motivation | RAG

So what is Retrieval-Augmented Generation (RAG)?

1. **Retrieval**: A storage & retrieval system that obtains context-relevant documents from a database.
2. **Generation**: A large language model that generates text based on the obtained documents.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Motivation | Creating an Expert Chatbot 🤖

<div style = "font-size: 0.8em;">

Our goal today is to build a RAG system that will answer questions about Northwestern's policy on academic integrity. To do this, we will:

1. **Tokenize** a document.<br>
Tokenization is the process of breaking text into smaller units called tokens. We'll learn how GPT-4 tokenizes text, and do the same.

2. **Chunk** the document into smaller, searchable units.<br>
Chunking is the process of creating windows of text that can be indexed and searched. We'll learn how to chunk text to make it compatible with a vector database.

3. **Embed** the text chunks.<br>
Word embeddings are dense vector representations of words that capture semantic information. We'll learn how to embed chunks using OpenAI's embedding model (and others!).

4. **Store and Retrieve** the embeddings from a vector database.<br>
We'll store the embeddings in a vector database and retrieve relevant documents based on the current context of a conversation. We'll demo with chromadb.

5. **Generate** text using the retrieved chunks and conversation context.<br>
We'll generate text with GPT-4 based on the retrieved chunks and a provided query, using OpenAI's API.

</div>

<!--s-->

<div class="header-slide">

## Retrieval-Augmented Generation (RAG)

1. <span style="color: #6f40b5;">**Tokenize** a document.</span><br>

2. **Chunk** the document into smaller, searchable units.<br>

3. **Embed** the chunks.<br>

4. **Store and Retrieve** the embeddings from a vector database.<br>

5. **Generate** text using the retrieved chunks and conversation context.<br>

</div>

<!--s-->

## Tokenize

Tokenization is the process of breaking text into smaller units called tokens. Tokens can be words, subwords, or characters. Tokenization is a crucial step in NLP because it allows us to work with text data in a structured way.

Some traditional tokenization strategies include:

- **Word Tokenization**. E.g. <span class="code-span">"Hello, world!" -> ["Hello", ",", "world", "!"] -> [12, 4, 56, 3]</span>
- **Subword Tokenization**. E.g. <span class="code-span">"unbelievable" -> ["un", "believable"] -> [34, 56]</span>
- **Character Tokenization**. E.g. <span class="code-span">"Hello!" -> ["H", "e", "l", "l", "o", "!"] -> [92, 34, 56, 56, 12, 4]</span>

One common, modern tokenization strategy is **Byte Pair Encoding(BPE)**, which is used by many large language models.

<!--s-->

## Tokenize | Byte Pair Encoding (BPE)

BPE is a subword tokenization algorithm that builds a vocabulary of subwords by iteratively merging the most frequent pairs of characters.

BPE is a powerful tokenization algorithm because it can handle rare words and out-of-vocabulary words. It is used by many large language models, including GPT-4. The algorithm is as follows:

```text
1. Initialize the vocabulary with all characters in the text.
2. While the vocabulary size is less than the desired size:
    a. Compute the frequency of all character pairs.
    b. Merge the most frequent pair.
    c. Update the vocabulary with the merged pair.
```

<!--s-->

## Tokenize | Byte Pair Encoding (BPE) with TikToken

One BPE implementation can be found in the `tiktoken` library, which is an open-source library from OpenAI.

```python

import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # Get specific encoding used by GPT-4.
enc.encode("Hello, world!") # Returns the tokenized text.

>> [9906, 11, 1917, 0]

```

<!--s-->

<div class="header-slide">

## Retrieval-Augmented Generation (RAG)

1. **Tokenize** a document.<br>

2. <span style="color: #6f40b5; font-size: 1em;">**Chunk** the document into smaller, searchable units.</span><br>

3. **Embed** the chunks.<br>

4. **Store and Retrieve** the embeddings from a vector database.<br>

5. **Generate** text using the retrieved chunks and conversation context.<br>

</div>

<!--s-->

## Chunk

<div style = "font-size: 0.8em;">

Chunking is the process of creating windows of text that can be indexed and searched. Chunking is essential for information retrieval systems because it allows us to break down large documents into smaller, searchable units.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; height: 100%;">


### Sentence Chunking

Sentence chunking is the process of breaking text into sentences.

E.g. <span class="code-span">"Hello, world! How are you?" -> ["Hello, world!", "How are you?"]</span>

### Paragraph Chunking

Paragraph chunking is the process of breaking text into paragraphs.

E.g. <span class="code-span">"Hello, world! \n Nice to meet you." -> ["Hello, world!", "Nice to meet you."]</span>

### Agent Chunking

Agent chunking is the process of breaking text down using an LLM.

</div>

<div class="c2" style = "width: 50%; height: 100%;">

### Sliding Word / Token Window Chunking

Sliding window chunking is a simple chunking strategy that creates windows of text by sliding a window of a fixed size over the text.

E.g. <span class="code-span">"The cat in the hat" -> ["The cat in", "cat in the", "in the hat"]</span>

### Semantic Chunking

Semantic chunking is the process of breaking text into semantically meaningful units.

E.g. <span class="code-span">"The cat in the hat. One of my favorite books." -> ["The cat in the hat.", "One of my favorite books."]</span>

</div>
</div>

<!--s-->

## Chunk | NLTK Sentence Chunking

NLTK is a powerful library for natural language processing that provides many tools for text processing. NLTK provides a sentence tokenizer that can be used to chunk text into sentences.

### Chunking with NLTK

```python
from nltk import sent_tokenize

# Load Academic Integrity document.
doc = open('/Users/joshua/Desktop/academic_integrity.md').read()

# Split the document into sentences.
chunked_data = sent_tokenize(doc)
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%;">

### Input: Original Text

```text
The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and
throughout the larger intellectual community of which we are members. Please read this booklet carefully,
as you will be held responsible for its contents. It describes the ways in which common sense and decency apply
to academic conduct. When you applied to Northwestern, you agreed to abide by our principles of academic integrity;
these are spelled out on the first three pages. The balance of the booklet provides information that will help you avoid
violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who 
can give you further information and counseling within the undergraduate schools.
```

</div>
<div class="c2" style = "width: 50%; height: 100%;">

### Output: Chunked Text (by Sentence)
```text
[
    "The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and throughout the larger intellectual community of which we are members."
    "Please read this booklet carefully, as you will be held responsible for its contents."
    "It describes the ways in which common sense and decency apply to academic conduct."
    "When you applied to Northwestern, you agreed to abide by our principles of academic integrity; these are spelled out on the first three pages."
    "The balance of the booklet provides information that will help you avoid violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who can give you further information and counseling within the undergraduate schools."
]

```
</div>
</div>

<!--s-->

<div class="header-slide">

## Retrieval-Augmented Generation (RAG)

1. **Tokenize** a document.<br>

2. **Chunk** the document into smaller, searchable units.<br>

3. <span style="color: #6f40b5;">**Embed** the chunks.</span><br>

4. **Store and Retrieve** the embeddings from a vector database.<br>

5. **Generate** text using the retrieved chunks and conversation context.<br>

</div>

<!--s-->

## Embed

Word embeddings are dense vector representations of words that capture semantic information. Word embeddings are essential for many NLP tasks because they allow us to work with words in a continuous and meaningful vector space.

**Traditional embeddings** such as Word2Vec are static and pre-trained on large text corpora.

**Contextual embeddings** such as those used by BERT and GPT are dynamic and trained on large language modeling tasks.

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Google</span>

<!--s-->

## Embed | Traditional Word Embeddings

Word2Vec is a traditional word embedding model that learns word vectors by predicting the context of a word. Word2Vec has two standard architectures:

- **Continuous Bag of Words (CBOW)**. Predicts a word given its context.
- **Skip-gram**. Predicts the context given a word.

Word2Vec is trained on large text corpora and produces dense word vectors that capture semantic information. The result of Word2Vec is a mapping from words to vectors, where similar words are close together in the vector space.

<img src="https://storage.googleapis.com/slide_assets/word2vec.png" style="margin: 0 auto; display: block; width: 50%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Braun 2017</span>

<!--s-->

## Embed | Contextual Word Embeddings

Contextual word embeddings are word embeddings that are dependent on the context in which the word appears. Contextual word embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

For example, the word "bank" can have different meanings depending on the context:

- **"I went to the bank to deposit my paycheck."**
- **"The river bank was covered in mud."**

[HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) contains a [MTEB](https://arxiv.org/abs/2210.07316) leaderboard for some of the most popular contextual word embeddings:

<img src="https://storage.googleapis.com/cs326-bucket/lecture_14/leaderboard.png" style="margin: 0 auto; display: block; width: 50%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">HuggingFace, 2024</span>

<!--s-->

## Embed | OpenAI's Embedding Model

OpenAI provides an embedding model via API that can embed text into a dense vector space. The model is trained on a large text corpus and can embed text into a n-dimensional vector space.

```python
import openai

openai_client = openai.Client(api_key = os.environ['API_KEY'])
embeddings = openai_client.embeddings.create(model="text-embedding-3-large", documents=chunked_data)
```


🔥 Although they do not top the MTEB leaderboard, OpenAI's embeddings work well and the convenience of the API makes them a popular choice for many applications.

<!--s-->

<div class="header-slide">

## Retrieval-Augmented Generation (RAG)

1. **Tokenize** a document.<br>

2. **Chunk** the document into smaller, searchable units.<br>

3. **Embed** the chunks.<br>

4.  <span style="color: #6f40b5;">**Store and Retrieve** the embeddings from a vector database.</span><br>

5. **Generate** text using the retrieved chunks and conversation context.<br>

</div>

<!--s-->

## Store & Retrieve

A vector database is a database that stores embeddings and allows for fast similarity search. Vector databases are essential for information retrieval systems because they enable us to *quickly* retrieve relevant documents based on their similarity to a query. 

This retrieval process is very similar to your KNN search! However, vector databases will implement Approximate Nearest Neighbors (ANN) algorithms to speed up the search process -- ANN differs from KNN in that it does not guarantee the exact nearest neighbors, but rather a set of approximate nearest neighbors.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many vector databases options available, such as:

- [ChromaDB](https://www.trychroma.com/)
- [Pinecone](https://www.pinecone.io/product/)
- [Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Postgres with PGVector](https://github.com/pgvector/pgvector)
- [FAISS](https://ai.meta.com/tools/faiss/)
- ...

</div>
<div class="c2" style = "width: 50%">

<img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bg8JUIjbKncnqC5Vf3AkxA.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Belagotti, 2023</span>

</div>
</div>

<!--s-->

## Store & Retrieve | ChromaDB

<div style="font-size: 0.9em">

ChromaDB is a vector database that stores embeddings and allows for fast text similarity search. ChromaDB is built on top of SQLite and provides a simple API for storing and retrieving embeddings.

### 1. Initializing

Before using ChromaDB, you need to initialize a client and create a collection.

```python
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection('academic_integrity_nw')
```


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### 2. Storing Embeddings

Storing embeddings in ChromaDB is simple. You can store embeddings along with the original documents and ids.

```python
# Store embeddings in chromadb.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])
```

</div>
<div class="c2" style = "width: 50%">

### 3. Retrieving Embeddings

You can retrieve embeddings from ChromaDB based on a query. ChromaDB will return the most similar embeddings (and the original text) to the query.

```python
# Get relevant documents from chromadb, based on a query.
query = "Can a student appeal?"
relevant_chunks = collection.query(query_embeddings = embedding_function([query]), n_results = 2)['documents'][0]

>>> ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.',
     '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']

```

</div>
</div>
</div>

<!--s-->

## 3. Retrieving Embeddings | 🔥 Tips & Re-Ranking

In practice, the retrieved documents may not be in the order you want. While a vector db will often return documents in order of similarity to the query, you can re-rank documents based on a number of factors. Remember, your chatbot is paying per-token on calls to LLMs. You can cut costs by re-ranking the most relevant documents first and only sending those to the LLM.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

### Multi-criteria Optimization

Consideration of additional factors beyond similarity, such as document quality, recency, and 'authoritativeness'.

### User Feedback

Incorporate user feedback into the retrieval process. For example, if a user clicks on a document, it can be re-ranked higher in future searches.

</div>

<div class="c2" style = "width: 50%">

### Diversification

Diversify the search results by ensuring that the retrieved documents cover a wide range of topics.

### Query Expansion & Rephrasing

For example, if a user asks about "academic integrity", the system could expand the query to include related terms like "plagiarism" and "cheating". This will help retrieve more relevant documents.


<!--s-->

<div class="header-slide">

## Retrieval-Augmented Generation (RAG)

1. **Tokenize** a document.<br>

2. **Chunk** the document into smaller, searchable units.<br>

3. **Embed** the chunks.<br>

4. **Store and Retrieve** the embeddings from a vector database.<br>

5.  <span style="color: #6f40b5;">**Generate** text using the retrieved chunks and conversation context.</span><br>

</div>

<!--s-->

## Generate

Once we have retrieved the relevant chunks based on a query, we can generate text using a large language model. Large language models can be used for many tasks -- including text classification, text summarization, question-answering, multi-modal tasks, and more.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many large language models available at platforms like:

- [OpenAI GPT-4o](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAiAudG5BhAREiwAWMlSjKXwuvq9JRRX0xxXaS7yCSn-NWo3e4rso3D-enl2IblIH09phtCvSxoCJhoQAvD_BwE)
- [Anthropic Claude](https://claude.ai/)
- [HuggingFace (Many)](https://huggingface.co/)
- ...


</div>
<div class="c2" style = "width: 50%">

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcce3c437-4b9c-4d15-947d-7c177c9518e5_4258x5745.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Raschka, 2023</span>

</div>
</div>

<!--s-->

## Generate | Transformers

Large language models (LLMs) are powerful tools for text generation because they can generate coherent and contextually relevant text.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

**Encoder-Decoder Models**: T5, BART.

Encoder-decoder models generate text by encoding the input text into a fixed-size vector and then decoding the vector into text. Used in machine translation and text summarization.

**Encoder-Only**: BERT

Encoder-only models encode the input text into a fixed-size vector. These models are powerful for text classification tasks but are not typically used for text generation.

**Decoder-Only**: GPT-4, GPT-3, Gemini 

Autoregressive models generate text one token at a time by conditioning on the previous tokens. Used in text generation, language modeling, and summarization.

</div>
<div class="c2 col-centered" style = "width: 50%">

<div>
<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F81c2aa73-dd8c-46bf-85b0-90e01145b0ed_1422x1460.png" style="margin: 0; padding: 0; ">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Vaswani, 2017</span>
</div>
</div>
</div>

<!--s-->

## Transformers | Self-Attention

Self-attention is a mechanism that allows the model to weigh the importance of different words in a sequence when generating text. It computes attention scores for each word based on its relationship with other words in the sequence. It does this by computing three vectors for each word: the query (q), key (k), and value (v) vectors.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

**Query (q)**: Represents the word for which we are computing attention.

**Key (k)**: Represents the words we are attending to.

**Value (v)**: Represents the information we want to extract from the attended words.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://www.mdpi.com/applsci/applsci-11-01548/article_deploy/html/images/applsci-11-01548-g001.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Jung 2021</p>
</div>

</div>
</div>

<!--s-->

## Transformers | Self-Attention

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The attention score is computed as the dot product of the query and key vectors, followed by a softmax operation to normalize the scores. Value vectors are then weighted by these scores to produce the final output.

$$ \text{Attention}(q, k, v) = \text{softmax}\left(\frac{q \cdot k^T}{\sqrt{d_k}}\right) v $$

where: 
- $q$ is the query vector
- $k$ is the key vector
- $v$ is the value vector
- $d_k$ is the dimension of the key vector

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/attention.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>

</div>
</div>
</div>

<!--s-->

## Transformers | Self-Attention

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/attention.png' style='border-radius: 10px; width: 60%; height: 80%'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>

</div>

<!--s-->

## Transformers | Multi-Head Attention

Multi-head attention is an extension of the self-attention mechanism that allows the model to focus on different parts of the input sequence simultaneously. It does this by using multiple sets of query, key, and value vectors, each with its own learned parameters.

<div style='text-align: center;'>
   <img src='https://sanjayasubedi.com.np/assets/images/deep-learning/mha-scratch/mha_dp_fig.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

<!--s-->

## Transformers | Architecture

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/transformer.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>Vashwani 2017</p>
</div>

<!--s-->

## Generate | GPT-4 & OpenAI API

What really sets OpenAI apart is their extremely useful and cost-effective API. This puts their LLM in the hands of users with minimal effort.

```python

import openai

openai_client = openai.Client(api_key = os.environ['API_KEY'])
response = openai_client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, GPT-4!"}
    ]
)

```

<!--s-->

<div class="header-slide">

# Putting it All Together

</div>

<!--s--> 

## Putting it All Together

Now that we have discussed the components of Retrieval-Augmented Generation (RAG), let's use what we have learned to build an expert chatbot that can answer questions about Northwestern's policy on academic integrity.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Putting it All Together | Demo Copied Here

```python[1-10 | 12-13 | 15-16 | 18-19 | 21-23 | 25-26 | 28 - 39 | 40 - 42 | 44-46 | 48 - 51]
import os

import chromadb
import openai
from chromadb.utils import embedding_functions
from nltk import sent_tokenize

# Initialize clients.
chroma_client = chromadb.Client()
openai_client = openai.Client(api_key = os.environ['API_KEY'])

# Create a new collection.
collection = chroma_client.get_or_create_collection('academic_integrity_nw')

# Load Academic Integrity document.
doc = open('/Users/joshua/Desktop/academic_integrity.md').read()

# Chunk the document into sentences.
chunked_data = sent_tokenize(doc)

# Embed the chunks.
embedding_function = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002", api_key=os.environ['API_KEY'])
embeddings = embedding_function(chunked_data)

# Store embeddings in ChromaDB.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])

# Create a system prompt template.
SYSTEM_PROMPT = """

You are an expert in academic integrity at Northwestern University. You will provide a response 
to a student query using exact language from the provided relevant chunks of text.

RELEVANT CHUNKS:

{relevant_chunks}

"""

# Get user query.
user_message = "Can I appeal?"
print("User: " + user_message)

# Get relevant documents from chromadb.
relevant_chunks = collection.query(query_embeddings = embedding_function([user_message]), n_results = 2)['documents'][0]
print("Retrieved Chunks: " + str(relevant_chunks))

# Send query and relevant documents to GPT-4.
system_prompt = SYSTEM_PROMPT.format(relevant_chunks = "\n".join(relevant_chunks))
response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
print("RAG-GPT Response: " + response.choices[0].message.content)

```

```text
User: Can a student appeal?
Retrieved Chunks: ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.', '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']
RAG-GPT Response: Yes, a student may appeal any finding or sanction as specified by the school holding jurisdiction.
```
<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

Today we discussed Retrieval-Augmented Generation (RAG), a modern NLP approach that combines the strengths of information retrieval systems with large language models. Building a RAG system exposed us to many critical concepts in NLP, including: 

1. **Tokenization**: Breaking text into smaller units called tokens.
2. **Chunking**: Creating windows of text that can be indexed and searched.
3. **Embedding**: Representing text as dense vectors in a continuous vector space.
4. **Storage & Retrieval**: Storing embeddings in a vector database and retrieving relevant documents based on their similarity to a query.
5. **Generation**: Generating text using a large language model.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

# Exit Poll
## On a scale of 1-5, how confident are you with **text** methods such as:

- Regular Expressions
- Tokenization
- Bag of Words (BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Markov Chains
- Evaluation of Text Generation Models
- R.A.G.

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->