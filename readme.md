## Supplementary materials for the paper "*Moral Stories*: Situated Reasoning about Norms, Intents, Actions, and their Consequences" (Emelin et al., 2020)

<p align="center">
  <img src="images/example.png" />
</p>

**Abstract**: In social settings, much of human behavior is governed by unspoken rules of conduct. For artificial systems to be fully integrated into social environments, adherence to such norms is a central prerequisite. We investigate whether contemporary NLG models can function as behavioral priors for systems deployed in social settings by generating action hypotheses that achieve predefined goals under moral constraints. Moreover, we examine if models can anticipate likely consequences of (im)moral actions, or explain why certain actions are preferable by generating relevant norms. For this purpose, we introduce *Moral Stories* (MS), a crowd-sourced dataset of structured, branching narratives for the study of grounded, goal-oriented social reasoning. Finally, we propose decoding strategies that effectively combine multiple expert models to significantly improve the quality of generated actions, consequences, and norms compared to strong baselines, e.g. though abductive reasoning.

---

## Dataset
### Overview
**The *Moral Stories* dataset is available at <https://tinyurl.com/y99sg2uq>.** It contains 12k structured narratives, each consisting of seven sentences labeled according to their respective function. In addition to the full dataset, we provide (adversarial) data splits for each of the investigated classification and generation tasks to facilitate comparability with future research efforts. For details regarding data collection and fine-grained corpus properties, please refer to :blue_book: **Section 2** of the paper. 

### Story examples

<p align="center">
  <img src="images/stories.png" />
</p>

---

## Codebase
We provide code for the replication of data curation steps as well as experiments discussed in our paper. <code>requirements.txt</code> specifies libraries utilized by the codebase. Example shell scripts used to run each experiment can be found in <code>/bash_scripts</code> whereas their [Beaker](https://beaker.org/) analogues are provided in <code>/beaker_scripts</code>. The following briefly describes individual files included in the codebase:

### Dataset collection
(:blue_book: See **Section 2** of the paper.)
* <code>collect\_sc101\_writing\_prompts.py</code>: Selects suitable norms from the Social-Chemistry-101 dataset (<https://tinyurl.com/y7t7g2rx>) to be used as writing prompts for crowd-workers.
* <code>show\_human\_validation\_stats.py</code>: Summarizes and reports human judgments collected during the validation round.
* <code>remove\_low\_scoring\_stories.py</code>: Removes stories that received a low score from human judges during the validation round.
* <code>show\_dataset\_stats.py</code>: Computes and reports various dataset statistics.
* <code>identify\_latent\_topics.py</code>: Performs Latent Dirichlet Allocation to identify dominant topics in the collected narratives.

### Split creation
(:blue_book: See **Section 3** of the paper.)
* <code>create\_action\_lexical\_bias\_splits.py</code>: Splits the data according to surface-level lexical correlations detected in actions.
* <code>create\_consequence\_lexical\_bias\_splits.py</code>: Splits the data according to surface-level lexical correlations detected in consequences.
* <code>create\_minimal\_action\_pairs\_splits.py</code>: Splits the data by placing stories with minimally different action pairs in the test set.
* <code>create\_minimal\_consequence\_pairs\_splits.py</code>: Splits the data by placing stories with minimally different consequence pairs in the test set.
* <code>create\_norm\_distance\_splits.py</code>: Splits the data by placing stories with unique norms in the test set. 

### Experiments
(:blue_book: See **Sections 3 and 4** of the paper.)
* <code>compute\_generation\_metrics.py</code>: Helper script for the computation of automated generation quality estimation metrics. 
* <code>compute\_norm\_diversity.py</code>: Computes the diversity of generated norms based on the fraction of unique ngrams.
* <code>run\_baseline\_experiment.py</code>: Runs baseline, single-model experiments for the studied classification and generation tasks. 
* <code>run\_coe\_action\_ranking\_experiment.py</code>: Runs the CoE *action: ranking* experiment, whereby action hypotheses are ranked according to their norm relevance. 
* <code>run\_coe\_action\_abductive\_refinement\_experiment.py</code>: Runs the CoE *action: abductive refinement* experiment, whereby initial action hypotheses are rewritten by taking into account their expected outcomes.
* <code>run\_coe\_consequence\_ranking\_experiment.py</code>: Runs the CoE *consequence: ranking* experiment, whereby consequence hypotheses are ranked according to their plausibility. 
* <code>run\_coe\_consequence\_iterative\_refinement\_experiment.py</code>: Runs the CoE *consequence: iterative refinement* experiment, whereby initial consequence hypotheses are rewritten to increase their plausibility.
* <code>run\_coe\_norm\_synthetic\_consequences\_experiment.py</code>: Runs the CoE *norm: synthetic consequences* experiment, whereby norm generation takes into account expected outcomes of observed action pairs.
* <code>utils.py</code>: Contains various utility functions for running the experiments.

### Human evaluation
(:blue_book: See **Section 4** of the paper.)
* <code>get\_action\_stats.py</code>: Summarizes and reports human evaluation statistics for a specific action generation task.
* <code>get\_consequence\_stats.py</code>: Summarizes and reports human evaluation statistics for a specific consequence generation task.
* <code>get\_norm\_stats.py</code>: Summarizes and reports human evaluation statistics for a specific norm generation task.

---

## Citation

```
TODO
```

