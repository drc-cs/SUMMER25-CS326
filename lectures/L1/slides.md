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
  ## L1

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
  ## Please create an account and check in by entering the provided code.

  </div>
  </div>
  <div class="c2" style="width: 50%; height: auto;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width="100%" height="100%" style="border-radius: 10px;"></iframe>
  </div>
</div>

<!--s-->

## Attendance

Attendance at lectures is **mandatory** and in your best interest!

Attendance is mandatory for CS 326 and is worth 10% of your final grade. Most lectures will have interactive quizzes throughout the lecture that will be graded for completion. We only have 9 sessions together this summer, so your effort is appreciated!

<!--s-->

## Grading

There is a high emphasis on the practical application of the concepts covered in this course. 
<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

| Component | Weight |
| --- | --- |
| Homework | 50% |
| Exam Part I | 20% |
| Exam Part II | 20% |
| Attendance | 10% |

</div>
<div class="c2" style = "width: 50%">

| Grade | Percentage |
| --- | --- |
| A | 94-100 |
| A- | 90-93 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |

</div>
</div>

<!--s-->

## Homework 

Homeworks are designed to reinforce the concepts covered in lecture. They will be a mix of theoretical and practical problems, and each will include a programmatic and free response portion. If there is time, we will work on homework at the end of every lecture as a group. For the automated grading server, you get 2 attempts per homework assignment. The highest score will be recorded.

- **Due**: Homework due dates are posted in the syllabus.

- **Late Policy**: Late homeworks will lose 10% per day (1% of your final grade).

- **Platform**: A submission script has been provided to submit your homeworks.

- **Collaboration**: You are encouraged to work with your peers on homeworks. However, you must submit your own work. Copying and pasting code from other sources will be detected and penalized.

<!--s-->

## LLMs (The Talk)

<iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>

<!--s-->

## Exams

There are two exams in this class. They will cover the theoretical and practical concepts covered in the lectures and homeworks. If you follow along with the lectures and homeworks, you will be well-prepared for the exams.

<!--s-->

## Academic Integrity [&#x1F517;](https://www.northwestern.edu/provost/policies-procedures/academic-integrity/index.html)

### Homeworks

- Do not exchange code fragments on any assignments.
- Do not copy solutions from any source, including the web or previous CS 326 students.
- You cannot upload / sell your assignments to code sharing websites.

<!--s-->

## Accommodations

Any student requesting accommodations related to a disability or other condition is required to register with AccessibleNU and provide professors with an accommodation notification from AccessibleNU, preferably within the first two days of class.

All information will remain confidential.

<!--s-->

## Mental Health

If you are feeling distressed or overwhelmed, please reach out for help. Students can access confidential resources through the Counseling and Psychological Services (CAPS), Religious and Spiritual Life (RSL) and the Center for Awareness, Response and Education (CARE).

<!--s-->

## Stuck on Something?

### **Email**

I'm here to help you! Please try contacting me through email first (josh@northwestern.edu), and I will do my best to respond within a few hours.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 40%; position: absolute;">
    
  ## L1 | Q1

  What are you most looking forward to learning in CS 326?

  </div>
  </div>

  <div class="c2 col-centered" style = "width: 80%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q1" width="100%" height="100%" style="border-radius: 10px;"></iframe>
  </div>
</div>

</div>

<!--s-->

<div class="header-slide">

# Homework Assignment
## H1 "Hello World!"

Due: 06.23.2025

</div>

<!--s-->

## H1 | Installing Conda

We will be using Python for this course. Conda is a package manager that will help us install Python and other packages. 

Don't have <span class="code-span">conda</span> installed? [[click-here]](https://docs.conda.io/en/latest/miniconda.html)

<!--s-->

## H1 | Cloning Repo & Installing Environment

We will be using a public GitHub repository for this course. Enter the following commands in your terminal to clone the repository and install the class environment.


```bash
git clone https://github.com/drc-cs/SUMMER25-CS326.git
cd SUMMER25-CS326
```

We will be using a conda environment (cs326) for this course. 

```bash
conda env create -f environment.yml
conda activate cs326
```

Don't have <span class="code-span">git</span> installed? [[click-here]](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

<!--s-->

## H1 | VSCode

All demonstrations will be in VSCode, which is a popular IDE. You're welcome to use any IDE you want, but I will be best equipped to help you if you use VSCode.

Don't have <span class="code-span">visual studio code</span> installed? [[click-here]](https://code.visualstudio.com/)

After installing VSCode, you will need to install the Python extension: 

1. Open VSCode
2. Click on the "Extensions" icon in the "Activity Bar" on the side of the window
3. Search for "Python" and click "Install"

Add the <span class="code-span">code</span> command to your PATH so you can open VSCode from the terminal.

1. Open the "Command Palette" (Ctrl+Shift+P on PC or Cmd+Shift+P on Mac)
2. Search for "Shell Command: Install 'code' command in PATH" and click it

<!--s-->

## H1 | Opening Repo in VSCode

Restart your terminal and open the cloned repository in VSCode using the following command:

```bash
code SUMMER25-CS326
```

You should see the following folders:

- <span class="code-span">README.md</span>: Contains the course syllabus.
- <span class="code-span">lectures/</span>: Contains the lecture slides in markdown format.
- <span class="code-span">homeworks/</span>: Contains the homework assignments.

<!--s-->

## H1 | Pulling

Before you start working on any homework, make sure you have the latest version of the repository.

The following command (when used inside the class folder) will pull the latest version of the repository and give you access to the most up-to-date homework:

```bash
git pull
```

If you have any issues with using this git-based system, please reach out.

<!--s-->

## H1 | Opening Homework

Open the <span class="code-span">homeworks/</span> folder in VSCode. You should see a folder called <span class="code-span">H1/</span>. Open the folder and you will see three files: 

- <span class="code-span">hello_world.py</span>: This file contains placeholders for the methods you will write.

- <span class="code-span">hello_world.ipynb</span>: This is a Jupyter notebook that provides a useful narrative for the homework and methods found in the <span class="code-span">hello_world.py</span> file.

- <span class="code-span">hello_world_test.py</span>: This is the file that will be used to test your code. Future homeworks will not include this file, and this is for demonstration purposes only.

We'll do the first homework together.

<!--s-->

<div class = "header-slide">

 ## Homework Demonstration

</div>

<!--s-->

## H1 | Submitting Homework

You will submit your homework using the provided submission script. 

But first, you need a username (your **northwestern** email e.g. JaneDoe2024@u.northwestern.edu) and password!

```bash
python account.py --create-account
```

Once you have a username and password, you can submit your completed homework. You should receive your score or feedback within a few seconds, but this may take longer as the homeworks get more involved.

```bash
python submit.py --homework H1/hello_world.py --username your_username --password your_password 
```

You can save your username and password as environment variables so you don't have to enter them every time (or expose them in your notebooks)!

```bash
export AG_USERNAME="your_username"
export AG_PASSWORD="your_password"
```
<!--s-->

## H1 | Homework Grading

The highest score will be recorded, so long as it is submitted before the deadline! You have 2 attempts for every homework. 

Late homeworks will be penalized 10% per day.

<!--s-->

## H1 | Expected Learnings / TLDR;

With this <span class="code-span">hello_world</span> assignment, we worked on the following:

1. Environment installation practice (<span class="code-span">conda</span>)
2. Exposure to git management (<span class="code-span">GitHub</span>)
3. Local development IDE practice (<span class="code-span">vscode</span>)
4. Familiarity with unit testing (<span class="code-span">pytest</span>)

These tools are all critical for any industry position and are often expected for entry level positions. Please continue to familiarize yourself with them over the course of the quarter.

<!--s-->

<div class = "header-slide">

# Data Sources

</div>

<!--s-->

## Data Sources

Data is at the heart of all data science! It's the raw material that we use to build models, make predictions, and draw conclusions. Data can be gathered from a variety of sources, and it comes in many different forms. 

<div class = "col-wrapper">
<div class = "c1" style = "width: 50%;">

Common Data Sources:

- Bulk Downloads
- APIs
- Scraping, Web Crawling
- BYOD (Bring Your Own Data)

</div>

<div class = "c2" style = "width: 50%">

<img src="https://imageio.forbes.com/blogs-images/gilpress/files/2016/03/Time-1200x511.jpg?format=jpg&width=1440" width="100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Forbes 2022</p>

</div>

<!--s-->

## Data Sources | Bulk Downloads

Large datasets that are available for download from the internet.


| Source | Examples | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Government | data.gov | often free, very large datasets | Non-specific |
| Academic | UCI Machine Learning Repository | usually free, large datasets | Non-specific |
| Industry | Kaggle, HuggingFace | sometimes free, large datasets | Non-specific, sometimes expensive |


```python
# Demonstration of pulling from the UCI machine learning repository.
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
```

```python
# Demonstration of pulling from the HuggingFace datasets library.
from datasets import load_dataset
dataset = load_dataset('imdb')
```

<!--s-->

## Data Sources | APIs

Data APIs are often readily available for sign up (usually for a fee).

While offering a lot of data, APIs can be restrictive in terms of the data they provide (and the rate at which you can pull it!).

API's often have better structure (usually backed by a db), and they often have the additional benefit of streaming or real-time data that you can poll for updates.

```python
import requests

API_KEY = 'your_api_key'
url = f'https://api.yourapi.com/data?api_key={API_KEY}'
r = requests.get(url)
data = r.json()
```

<!--s-->

## Data Sources | Scraping, Web Crawling

Web crawling is a free way to collect data from the internet. But be **cautious**. Many websites have terms of service that prohibit scraping, and you can easily overstep those and find yourself in legal trouble.

Examples of packages built for webscraping in Python include <span class="code-span">beautifulsoup</span> and <span class="code-span">scrapy</span>.

```python
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
soup.find_all('p')
```

<!--s-->

## Data Sources | BYOD (Bring Your Own Data)

Data that you collect yourself. In this case, you determine the format.

Examples for collecting survey data include Qualtrics and SurveyMonkey.

```python
import pandas as pd
df = pd.read_csv('survey_data.csv')
```
<!--s-->

## Data Sources | BYOD (Bring Your Own Data)
Collecting data yourself is common in academia and industry. But you need to be careful.

<div class = "col-wrapper">
<div class = "c1" style = "width: 70%;">

**Policies Exist**
- *GDPR*: Controllers of personal data must put in place appropriate technical and organizational measures to implement the data protection principles.
- *HIPAA*: Covered entities must have in place appropriate administrative, technical, and physical safeguards to protect the privacy of protected health information.

**Bias *Always* Exists** Bias is everywhere. It's in the data you collect, the data you don't collect, and the data you use to train your models. In almost all cases, we perform a *sampling*. It's your job to ensure it is a reasonable sample.

**Ethical Concerns**
Don't collect data that you don't need. Definitely don't collect data that you don't have permission to collect.

</div>
<div class = "c2 col-centered"  style = "width: 30%">
<iframe width="256" height="256" src="https://www.youtube.com/embed/R2vhjC5qCQk?si=m3ImvZqhDawRoMTo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
</div>

<!--s-->

## L1 | Q2

You're building a model to predict the price of a house based on its location, size, and number of bedrooms. Which of the following data sources would be a great first place to look?

<div class="col-wrapper">
<div class="c1" style = "width: 60%;">

<div style = "line-height: 2em;">
&emsp; A. Bulk Downloads <br>
&emsp; B. APIs <br>
&emsp; C. Scraping, web crawling <br>
&emsp; D. BYOD (Bring Your Own Data) <br>
</div>
</div>

<div class="c2 col-centered" style = "width: 40%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q2" width="100%" height="100%"></iframe>
</div>
</div>

<!--s-->

## L1 | Q3

You just built an amazing stock market forecasting model. Congrats! Now, you want to test it on real-time data. Which of the following data sources would be a great first place to look?

<div class="col-wrapper">
<div class="c1" style = "width: 60%;">

<div style = "line-height: 2em;">
&emsp; A. Bulk Downloads <br>
&emsp; B. APIs <br>
&emsp; C. Scraping, web crawling <br>
&emsp; D. BYOD (Bring Your Own Data) <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 40%;">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q3" width="100%" height="100%"></iframe>

</div>
</div>

<!--s-->

<div class = "header-slide">

# Data Structures

</div>

<!--s-->

## Data Structures

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

<!--s-->

## Data Structures | Structured Example (Relational Database)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```sql
CREATE TABLE employees (
  employee_id INT PRIMARY KEY,
  name VARCHAR(100),
  hire_date DATE
);
```
<!--s-->

## Data Structures | Semi-Structured Example (JSON)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```json
employeeData = {
  "employee_id": 1234567,
  "name": "Jeff Fox",
  "hire_date": "1/1/2013"
};
```

<!--s-->

## Data Structures | Unstructured Example (Plain Text)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```text
Dear Sir or Madam,

My name is Jeff Fox (employee id 1234567). I'm excited to start on 1/1/2013.

Sincerely,
Jeff Fox
```

<!--s-->

<div class = "header-slide">

# Relational Databases

</div>

<!--s-->

## Relational Databases

<img src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/a2906fd68b050d7f9e0714c7d566990efd645005-1953x1576.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

<!--s-->

## Relational Databases

<img src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/db72cc3ac506bec544588454972113c4dc3abe50-1953x1576.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

<!--s-->


## Relational Databases

Relational databases are a type of database management system (DBMS) that store and manage data in a structured format using tables. Each table, or relation, consists of rows and columns, where rows represent individual records and columns represent the attributes of the data. Widely used systems include MySQL and PostgreSQL. 

### Key Vocabulary:

- **Tables:** Organized into rows and columns, with each row being a unique data entry and each column representing a data attribute.
- **Relationships:** Tables are connected through keys, with primary keys uniquely identifying each record and foreign keys linking related records across tables.
- **SQL (Structured Query Language):** The standard language used to query and manipulate data within a relational database.

<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 1)

### `CREATE TABLE`

  ```sql
  /* Create a table called table_name with column1, column2, and column3. */
  CREATE TABLE table_name (
    column1 INT PRIMARY KEY, /* Primary key is a unique identifier for each row. */
    column2 VARCHAR(100), /* VARCHAR is a variable-length string up to 100 characters. */
    column3 DATE /* DATE is a date type. */
  );
  ```

### `INSERT INTO`

  ```sql
  /* Insert values into column1, column2, and column3 in table_name. */
  INSERT INTO table_name (column1, column2, column3) VALUES (value1, value2, value3);
  ```

### `UPDATE`

  ```sql
  /* Update column1 in table_name to 'value' where column2 is equal to 'value'. */
  UPDATE table_name SET column1 = 'value' WHERE column2 = 'value';
  ```

### `DELETE`

  ```sql
  /* Delete from table_name where column1 is equal to 'value'. */
  DELETE FROM table_name WHERE column1 = 'value';
  ```

</div>

<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 2)

### `SELECT`
  
  ```sql
  /* Select column1 and column2 from table_name.*/
  SELECT column1, column2 FROM table_name;
  ```

### `WHERE`

  ```sql
  /* Select column1 and column2 from table_name where column1 is equal to 'value' and column2 is equal to 'value'. */
  SELECT column1, column2 FROM table_name WHERE column1 = 'value' AND column2 = 'value';
  ```

### `ORDER BY`

  ```sql
  /* Select column1 and column2 from table_name and order by column1 in descending order. */
  SELECT column1, column2 FROM table_name ORDER BY column1 DESC;
  ```

### `LIMIT`

  ```sql
  /* Select column1 and column2 from table_name and limit the results to 10. */
  SELECT column1, column2 FROM table_name LIMIT 10;
  ```

</div>
<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 3)

### `JOIN`

  ```sql
  /* Select column1 and column2 from table1 and table2 where column1 is equal to column2. */
  SELECT column1, column2 FROM table1 JOIN table2 ON table1.column1 = table2.column2;
  ```

### `GROUP BY`

  ```sql
  /* Select column1 and column2 from table_name and group by column1. */
  SELECT column1, column2 FROM table_name GROUP BY column1;
  ```

### `COUNT`

  ```sql
  /* Select the count of column1 from table_name. */
  SELECT COUNT(column1) FROM table_name;

  /* Group by column2 and select the count of column1 from table_name. */
  SELECT column2, COUNT(column1) FROM table_name GROUP BY column2;
  ```

### `SUM`

  ```sql
  /* Select the sum of column1 from table_name. */
  SELECT SUM(column1) FROM table_name;

  /* Group by column2 and select the sum of column1 from table_name. */
  SELECT column2, SUM(column1) FROM table_name GROUP BY column2;
  ```

</div>

<!--s-->

### SQL and Pandas (🔥)

```python
import pandas as pd
import sqlite3

# Create a connection to a SQLite database.
conn = sqlite3.connect('example.db')

# Load a DataFrame into the database.
df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
df.to_sql('table_name', conn, if_exists='replace')

# Query the database.
query = 'SELECT * FROM table_name'
df = pd.read_sql(query, conn)
```

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  ## L1 | Q4

  
  Write a SQL query that selects the <span class = "code-span">name</span> and <span class = "code-span">year</span> columns from a <span class = "code-span">movies</span> table where the <span class = "code-span">year</span> is greater than 2000.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 5%">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q4" width="100%" height="100%"></iframe>
  </div>
</div>

<!--s-->

## Practice Your SQL!

The majority of data science interviews will have a SQL component. It's a good idea to practice your SQL skills. Here are a few resources to get you started:

- ### [SQLZoo](https://sqlzoo.net/)
- ### [W3Schools](https://www.w3schools.com/sql/)
- ### [LeetCode](https://leetcode.com/problemset/database/)
- ### [HackerRank](https://www.hackerrank.com/domains/sql)
- ### [SQL Practice](https://www.sql-practice.com/)

<!--s-->

<div class = "header-slide">

# Data Cleaning

</div>

<!--s-->

## Data Cleaning

Data is often dirty! Don't ever give your machine learning or statistical model dirty data.

Remember the age-old adage:

> Garbage in, garbage out.

 Data cleaning is the process of converting source data into target data without errors, duplicates, or inconsistencies. You will often need to structure data in a way that is useful for your analysis, so learning some basic data manipulation is **essential**.

<!--s-->

## Data Cleaning | Common Data Issues

1. Incompatible data
2. Missing values
3. Extreme Outliers

<!--s-->

## Data Cleaning | Handling Incompatible Data

<div style = "font-size: 0.85em;">

| Data Issue | Description | Example | Solution |
| --- | --- | --- | --- |
| Unit Conversions | Numerical data conversions can be tricky. | 1 mile != 1.6 km | Measure in a common unit, or convert with caution. |
| Precision Representations | Data can be represented differently in different programs. | 64-bit float to 16-bit integer | Use the precision necessary and hold consistent. |
| Character Representations | Data is in different character encodings. | ASCII, UTF-8, ... | Create using the same encoding, or convert with caution. |
| Text Unification | Data is in different formats. | D'Arcy; Darcy; DArcy; D Arcy; D&-#-3-9-;Arcy | Use a common format, or convert with caution. <span class="code-span">RegEx</span> will be your best friend.| 
| Time / Date Unification | Data is in different formats. | 10/11/2019 vs 11/10/2019 | Use standard libraries & UTC. A personal favorite is seconds since epoch. |

</div>

<!--s-->

## Data Cleaning | Handling Missing Values

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

  Data is often missing from datasets. It's important to identify why it's missing. Once you have established that it is **missing at random**, you can proceed with **substitution**.

  When missing data, we have a few options at our disposal:

  1. Drop the entire row
  2. Drop the entire column
  3. Substitute with a reasonable value

</div>
<div class="c2" style = "width: 50%">

  ``` [1-5|4]
  id    col1    col2    col3    col4
  p1    2da6    0.99    32     43
  p2    cd2d    1.23    55      38
  p3    9999    89.2    NaN     32
  p4    4e7c    0.72    9.7     35
  ```

</div>
</div>

<!--s-->

## Data Cleaning | Handling Missing Values with Substitution

<div class = "col-wrapper"> 
<div class = "c1" style = "width: 70%; font-size: 0.85em;">


| Method | Description | When to Use |
| --- | --- | --- |
| Forward / backward fill | Fill missing value using the last / next valid value. | Time Series |
| Imputation by interpolation | Use interpolation to estimate missing values. | Time Series |
| Mean value imputation | Fill missing value with mean from column. | Random missing values |
| Conditional mean imputation | Estimate mean from other variables in the dataset. | Random missing values |
| Random imputation | Sample random values from a column. | Random missing values | 
| KNN imputation | Use K-nearest neighbors to fill missing values. | Random missing values |
| Multiple Imputation | Uses many regression models and other variables to fill missing values. | Random missing values |

</div>

<div class = "c2 col-centered" style = "width: 40%">

```python
import pandas as pd

# Load data.
df = pd.read_csv('data.csv')

# Forward fill.
df_ffill = df.fillna(method='ffill')

# Backward fill.
df_bfill = df.fillna(method='bfill')

# Mean value imputation.
df_mean = df.fillna(df.mean())

# Random value imputation.
df_random = df.fillna(df.sample())

# Imputation by interpolation.
df_interpolate = df.interpolate()
```

</div>
</div>

<!--s-->

## L1 | Q5

You have streaming data that is occasionally dropping values. Which of the following methods would be appropriate to fill missing values when signal fails to update? 

*Please note, in this scenario, you can't use the future to predict the past.*

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 60%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Forward fill <br>
&emsp;B. Imputation by interpolation <br>
&emsp;C. Mean value imputation <br>
&emsp;D. Backward fill <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 40%; padding-bottom: 20%">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q5" width="100%" height="100%"></iframe>
</div>
</div>

<!--s-->

## Data Cleaning | Handling Outliers

Outliers are extreme values that deviate from other observations on data. They may indicate a variability in a measurement, experimental errors, or a novelty.

Outliers can be detected using:

- **Boxplots**: A visual representation of the five-number summary.
- **Scatterplots**: A visual representation of the relationship between two variables.
- **z-scores**: A measure of how many standard deviations a data point is from the mean.
- **IQR**: A measure of statistical dispersion, being equal to the difference between the upper and lower quartiles.

Handling outliers should be done on a case-by-case basis. Don't throw away data unless you have a very compelling (and **documented**) reason to do so!

<!--s-->

## Data Cleaning | Keep in Mind

- Cleaning your data is an iterative process.
  - **Hot tip**: Focus on making your data preprocessing *fast*. You will be doing it a lot, and you'll want to be able to iterate quickly. Look into libraries like <span class="code-span">dask</span>, <span class="code-span">pyspark</span>, and <span class="code-span">ray</span> for large datasets.
- Data cleaning is often planned with visualization. 
  - Always look at the data. Always. We'll go over plotting approaches soon.
- Data cleaning can fix modeling problems.
  - Your first assumption should always be "something is wrong with the data", not "I should try another model".
- Data cleaning is not a one-size-fits-all process, and often requires domain expertise.

<!--s-->

<div class="header-slide"> 

# EDA

</div>

<!--s-->

## Exploratory Data Analysis (EDA) | Introduction

**Exploratory Data Analysis** (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.

This should be done every single time you are exposed to a new dataset. **ALWAYS** look at the data (x3).

EDA will help you to identify early issues or patterns in the data, and will guide you in the next steps of your analysis. It is an absolutely **critical**, but often overlooked step.

<!--s-->

## Exploratory Data Analysis (EDA) | Methodology

We can break down EDA into two main topics:

- **Descriptive EDA**: Summarizing the main characteristics of the data.
- **Graphical EDA**: Visualizing the data to understand its structure and patterns.

<!--s-->

<div class="header-slide"> 

# Descriptive EDA

</div>

<!--s-->

## Descriptive EDA | Describe $x$

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/boxplot.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

## Descriptive EDA | Describe $x$

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/violin.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

## Descriptive EDA | Examples

- **Central tendency**
    - Mean, Median, Mode
- **Spread**
    - Range, Variance, interquartile range (IQR)
- **Skewness**
    - A measure of the asymmetry of the distribution. Typically close to 0 for a normal distribution.
- **Kurtosis**
    - A measure of the "tailedness" of the distribution. Typically close to 3 for a normal distribution.
- **Modality**
    - The number of peaks in the distribution.

<!--s-->

## Central Tendency

- **Mean**: The average of the data. 

    - $ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $

    - <span class="code-span">np.mean(data)</span>

- **Median**: The middle value of the data, when sorted.

    - [1, 2, **4**, 5, 6]

    - <span class="code-span">np.median(data)</span>

- **Mode**: The most frequent value in the data.

    ```python
    from scipy.stats import mode
    data = np.random.normal(0, 1, 1000)
    mode(data)
    ```

<!--s-->

## Spread

- **Range**: The difference between the maximum and minimum values in the data.
    
    - <span class="code-span">np.max(data) - np.min(data)</span>

- **Variance**: The average of the squared differences from the mean.

    - $ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $

    - <span class="code-span">np.var(data)</span>

- **Standard Deviation**: The square root of the variance.

    - `$ \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2} $`
    - <span class="code-span">np.std(data)</span>

- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles.
    - <span class="code-span">np.percentile(data, 75) - np.percentile(data, 25)</span>

<!--s-->

## Skewness

A measure of the lack of "symmetry" in the data.

**Positive skew (> 0)**: the right tail is longer; the mass of the distribution is concentrated on the left of the figure.

**Negative skew (< 0)**: the left tail is longer; the mass of the distribution is concentrated on the right of the figure.

```python
import numpy as np
from scipy.stats import skew

data = np.random.normal(0, 1, 1000)
print(skew(data))
```

<!--s-->

## Skewness | Plot

<iframe width = "80%" height = "80%" src="https://storage.googleapis.com/slide_assets/skewness.html" title="scatter_plot"></iframe>

<!--s-->

## L1 | Q6

Which of the following would correctly calculate the median of the following list?

```python
data = [1, 2, 4, 3, 5]

```

<div class="col-wrapper">
<div class="c1" style = "width: 60%;">

<div style = "line-height: 2em;">

A. 
```python
median = sorted(data)[len(data) // 2]
```

B.

```python
median = sorted(data)[len(data) // 2 - 1]
```

C.
```python
median = sorted(data)[len(data) // 2 + 1]
```

</div>

</div>

<div class="c2 col-centered" style = "width: 40%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q6" width="100%" height="100%"></iframe>
</div>
</div>

<!--s-->

## L1 | Q7

Is this distribution positively or negatively skewed?

<div class="col-wrapper">
<div class="c1" style = "width: 60%;">

&emsp;A. Positively skewed<br>
&emsp;B. Negatively skewed<br>
&emsp;C. No skew<br>
</div>

<div class="c2 col-centered" style = "width: 40%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L1 | Q7" width="100%" height="100%"></iframe>
</div>

</div>

<!--s-->


## Kurtosis

A measure of the "tailedness" of the distribution.

- **Leptokurtic (> 3)**: the tails are fatter than the normal distribution.

- **Mesokurtic (3)**: the tails are the same as the normal distribution.

- **Platykurtic (< 3)**: the tails are thinner than the normal distribution.

```python
import numpy as np
from scipy.stats import kurtosis

data = np.random.normal(0, 1, 1000)
print(kurtosis(data))
```

<!--s-->

## Kurtosis | Plot

<img width = "100%" height = "100%" src="https://www.scribbr.com/wp-content/uploads/2022/07/The-difference-between-skewness-and-kurtosis.webp" title="scatter_plot"></img>

<!--s--> 

## Modality

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; padding-top: 10%;">

The number of peaks in the distribution.

- **Unimodal**: One peak.
- **Bimodal**: Two peaks.
- **Multimodal**: More than two peaks.

</div>

<div class="c2 col-centered" style = "width: 50%;">

<img src="https://treesinspace.com/wp-content/uploads/2016/01/modes.png?w=584" width="100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Trees In Space 2016</p>

</div>

<!--s-->

## Normal Distribution | Definition

A normal distribution is a continuous probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. Mathematically, it is defined as:

<div style = "text-align: center;">
$ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $
</div>

Where:

- $ \mu $ is the mean.
- $ \sigma^2 $ is the variance.

<!--s-->

## Normal Distribution | Properties

The normal distribution has several key properties:

- **Symmetry**: The distribution is symmetric about the mean.
- **Unimodality**: The distribution has a single peak.
- **68-95-99.7 Rule**: 
    - 68% of the data falls within 1 standard deviation of the mean.
    - 95% within 2 standard deviations.
    - 99.7% within 3 standard deviations.

<!--s-->

## Normal Distribution | Properties

How can we tell if our data is normally distributed?


<div class = "col-wrapper">
<div class="c1" style = "width: 50%; padding-top: 10%;">

- **Skewness**: close to 0.
- **Kurtosis**: close to 3.
- **QQ Plot**: A plot of the quantiles of the data against the quantiles of the normal distribution.
- **Shapiro-Wilk Test**: A statistical test to determine if the data is normally distributed.

</div>
<div class="c2 col-centered" style = "width: 50%;">

<img src="https://miro.medium.com/v2/format:webp/0*lsbu809F5gOZrrKX" width="100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Khandelwal 2023</p>

</div>

<!--s-->

## Descriptive EDA | Not Covered

There are many ways to summarize data, and we have only covered a few of them. Here are some common methods that we did not cover today:

- **Covariance**: A measure of the relationship between two variables.
- **Correlation**: A normalized measure of the relationship between two variables.
- **Outliers**: Data points that are significantly different from the rest of the data.
- **Missing Values**: Data points that are missing from the dataset.
- **Percentiles**: The value below which a given percentage of the data falls.
- **Frequency**: The number of times a value occurs in the data.

<!--s-->

<div class="header-slide">

# Graphical EDA 

</div>

<!--s-->

## Graphical EDA | Data Types

There are three primary types of data -- nominal, ordinal, and numerical.

| Data Type | Definition | Example |
| --- | --- | --- |
| Nominal | Categorical data without an inherent order | <span class="code-span">["red", "green", "orange"]</span> |
| Ordinal | Categorical data with an inherent order | <span class="code-span">["small", "medium", "large"]</span> <p><span class="code-span">["1", "2", "3"]</span> |
| Numerical | Continuous or discrete numerical data | <span class="code-span">[3.1, 2.1, 2.4]</span> |

<!--s-->

## Graphical EDA | Choosing a Visualization

The type of visualization you choose will depend on:

- **Data type**: nominal, ordinal, numerical.
- **Dimensionality**: 1D, 2D, 3D+.
- **Story**: The story you want to tell with the data.

Whatever type of plot you choose, make sure your visualization is information dense **and** easy to interpret. It should always be clear what the plot is trying to convey.

<!--s-->

## Graphical EDA | A Note on Tools

Matplotlib and Plotly are the most popular libraries for data visualization in Python.

| Library | Pros | Cons |
| --- | --- | --- |
| Matplotlib | Excellent for static publication-quality plots, very fast render, old and well supported. | Steeper learning curve, many ways to do the same thing, no interactivity, OOTB color schemes. |
| Plotly | Excellent for interactive plots, easy to use, easy tooling for animations, built-in support for dashboarding and publishing online. | Not as good for static plots, less fine-grained control, high density renders can be non-trivial. |

<!--s-->

## Graphical EDA | Basic Visualization Types

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### 1D Data
- Bar chart
- Pie chart
- Histogram
- Boxplot
- Violin plot
- Line plot

### 2D Data
- Scatter plot
- Heatmap
- Bubble plot
- Line plot
- Boxplot
- Violin plot

</div>
<div class="c2 col-centered" style = "width: 50%;">

### 3D+ Data

- 3D scatter plot
- Bubble plot
- Color scatter plot
- Scatter plot matrix

</div>
</div>

<!--s-->

## 1D Data | Histograms

When you have numerical data, histograms are a great way to visualize the distribution of the data. If there is a clear distribution, it's often useful to fit a probability density function (PDF).

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.histogram(data, nbins = 50)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/histogram.html" title="scatter_plot" padding=2em;></iframe>

</div>
</div>

<!--s-->


## 1D Data | Boxplots

Boxplots are a great way to visualize the distribution of the data, and to identify outliers.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.box(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/boxplot.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Violin Plots

Violin plots are similar to box plots, but they also show the probability density of the data at different values.

<div class = "col-wrapper">

<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.violin(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/violin.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Bar Charts

Bar charts are a great way to visualize the distribution of categorical data.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.choice(["A", "B", "C"], 1000)
fig = px.histogram(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/bar.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Pie Charts

Pie charts are another way to visualize the distribution of categorical data.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.choice(["A", "B", "C"], 100)
fig = px.pie(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/pie.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 2D Data | Scatter Plots

Scatter plots can help visualize the relationship between two numerical variables.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
fig = px.scatter(x = x, y = y)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/scatter.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 2D Data | Heatmaps (2D Histogram)

Heatmaps help to visualize the density of data in 2D.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
fig = px.density_heatmap(x = x, y = y)
fig.show()
```

</div>
<div class="c2" style = "width: 60%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/heatmap.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 3D+ Data | Bubble Plots

Bubble plots are a great way to visualize the relationship between three numerical variables and a categorical variable.


<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)
c = np.random.choice(["A", "B", "C"], 100)
fig = px.scatter(x = x, y = y, size = z, color = c)
fig.show()
```

</div>
<div class="c2" style = "width: 60%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/bubble_plot.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 3D+ Data | Scatter Plots

Instead of using the size of the markers (as in the bubble plot), you can use another axis to represent a third numerical variable. And, you still have the option to color by a categorical variable.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)
c = np.random.choice(["A", "B", "C"], 100)

fig = px.scatter_3d(x = x, y = y, z = z, color = c)
fig.show()
```

</div>
<div class="c2" style = "width: 50%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/scatter_3d.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## Graphical EDA | Advanced Visualization Types

- ML Results
    - Residual Plots
    - Regression in 3D
    - Decision Boundary
- Parallel Coordinates
- Maps / Chloropleth

<!--s-->

### Residual Plots

Residual plots are a great way to visualize the residuals of a model. They can help you identify patterns in the residuals, which can help you identify issues with your model.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/residual.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

### Regression in 3D

Regression in 3D is a great way to visualize the relationship between three numerical variables.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/svr.html" title="scatter_plot" padding=2em;></iframe>
<!--s-->

### Decision Boundary

Decision boundaries are a great way to visualize the decision-making process of a classification model.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/contour.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

### Parallel Coordinates

Parallel coordinates are a great way to visualize the relationship between multiple numerical variables, often used to represent hyperparameter tuning results.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/parcoords.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

### Maps / Chloropleth

Chloropleths are a great way to visualize data that is geographically distributed. They can help you understand the spatial distribution of the data, and can help you identify patterns in the data.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/choropleth.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

## EDA TLDR;

- **Descriptive EDA**: Summarizing the main characteristics of the data.
- **Graphical EDA**: Visualizing the data to understand its structure and patterns.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Wrapping Up

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>


<!--s-->

