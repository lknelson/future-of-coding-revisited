# future-of-coding-revisited

Replication repository for the paper "The Future of Coding Revisited"

In this repository you'll find code to reproduce the analysis and tables in the paper "The Future of Coding Revisited"

## Data:
- All of the original hand-coding of the news articles, plus the generated LLM codes from all of the tests
- The combined LLM-generated definitions

## Scripts:

The scripts folders contain the code to partially reproduce the analysis and metrics, including the fine tuning metrics reported in Appendix F.

Three folders contain the code to reproduce the LLM-generated code for our tests.

*Note these can't be fully reproduced as we can't post the entire news articles due to copyright. But we have posted enough information about each article in the data files for anyone to pull the same articles, which you can then use to replicate the analysis.*

* `scripts-relevant-zeroshot`
* `scripts-inequality-zeroshot`
* `scripts-inequalit-fewshot`

* `fine-tuning` contains the code to replicate the fine tuning approaches, reported in Appendix F. 

Others folders contain code and are fully replicable:

* `scripts-generate-LLM-definitions` contains the scripts used to generate LLM definitions of inequality, for use in our tests

* `scripts-generate-accuracy-measures` reproduces the metrics in Table 3

* `scripts-inter-agreement` reproduces the metrics in Appendix E
