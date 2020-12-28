## Supplmentary materials for the paper "*Moral Stories*: Situated Reasoning aboutNorms, Intents, Actions, and their Consequences" (Emelin et al., 2020)

<img align="right" src="images/example.png">
> Abstract: In social settings, much of human behavior is governed by unspoken rules of conduct. For artificial systems to be fully integrated into social environments, adherence to such norms is a central prerequisite. We investigate whether contemporary NLG models can function as behavioral priors for systems deployed in social settings by generating action hypotheses that achieve predefined goals under moral constraints. Moreover, we examine if models can anticipate likely consequences of (im)moral actions, or explain why certain actions are preferable by generating relevant norms. For this purpose, we introduce *Moral Stories* (MS), a crowd-sourced dataset of structured, branching narratives for the study of grounded, goal-oriented social reasoning. Finally, we propose decoding strategies that effectively combine multiple expert models to significantly improve the quality of generated actions, consequences, and norms compared to strong baselines, e.g. though abductive reasoning.

---

## Dataset

**The *Moral Stories* dataset is available at <https://tinyurl.com/y99sg2uq>.** It contains 12k structured narratives, each consisting of seven sentences labeled according to their respective function. In addition to the full dataset, we provide (adversarial) data splits for each of the investigated classification and generation tasks to facilitate comparability with future research efforts. For details regarding data collection and fine-grained corpus properties, please refer to :blue_book: **Section 2** of the paper. 

---

## Codebase

We provide code for the replication of experiments described in :blue_book: **Sections 3 and 4** of the paper. <code>requirements.txt</code> specifies the libraries utilized by our codebase. Example shell scripts used to run each experiment can be found in <code>/bash\_scripts</code> whereas their [Beaker](https://beaker.org/) analogues are given in <code>\beaker\_scripts</code>. The following provides an overview of the individual files included in the codebase:  

### dataset_collection/
*:blue_book: Used in **Section 2** of the paper.*
<code>collect\_sc101\_writing\_prompts.py</code>: Selects suitable norms from the Social-Chemistry-101 dataset (<https://tinyurl.com/y7t7g2rx>) to be used as writing prompts for crowd-workers.
<code>show\_human\_validation\_stats.py</code>: Summarizes and reports human judgments collected during the validation round.
<code>remove\_low\_scoring\_stories.py</code>: Removes stories that have received a low score during the valiadtion round from the dataset.
<code>show\_dataset\_stats.py</code>: Computes and reports various dataset statistics.
<code>identify\_latent\_topics.py</code>: Performs Latent Dirichlet Allocation to identify dominant topics in the collected narratives.

### split_creation/
*:blue_book: Used in **Section 3** of the paper.*
<code>create\_action\_lexical\_bias\_splits.py</code>: Splits the MS dataset according to surface-level lexical correlations detected in actions.
<code>create\_consequence\_lexical\_bias\_splits.py</code>: Splits the MS dataset according to surface-level lexical correlations detected in consequences.
<code>create\_minimal\_action\_pairs\_splits.py</code>: Splits the MS dataset by prioritizing stories with minimally different action pairs for the inclusion in the test set.
<code>create\_minimal\_consequence\_pairs\_splits.py</code>: Splits the MS dataset by prioritizing stories with minimally different consequence pairs for the inclusion in the test set.
<code>create\_norm\_distance\_splits.py</code>: Splits the MS dataset by prioritizing stories with unique norms for the inclusion in the test set. 

### experiments/
*:blue_book: Used in **Sections 3 and 4** of the paper.*
<code>compute\_generation\_metrics.py</code>: Helper script for the computation of automated generation quality metrics. 
<code>compute\_norm\_diversity.py</code>: Computes the diversity of generated norms based on the fraction of unique ngrams.
<code>run\_baseline\_experiment.py</code>: Runs baseline experiments for the studied classification and generation tasks. 
<code>run\_coe\_action\_ranking\_experiment.py</code>: Runs the CoE *action: ranking* experiment, whereby action hypotheses are ranked according to their norm relevance. 
<code>run\_coe\_action\_abductive\_refinement\_experiment.py</code>: Runs the CoE *action: abductive refinement* experiment, whereas action hypotheses are rewritten by taking into account their expected outcomes.
<code>run\_coe\_consequence\_ranking\_experiment.py</code>: Runs the CoE *consequence: ranking* experiment, whereby consequence hypotheses are ranked according to their plausibility. 
<code>run\_coe\_consequence\_iterative\_refinement\_experiment.py</code>: Runs the CoE *consequence: iterative refinement* experiment, whereby inial consequence hypotheses are rewritten to increase their plausibility.
<code>run\_coe\_norm\_synthetic\_consequences\_experiment.py</code>: Runs the CoE *norm: synthetic consequences* experiment, whereby norm generation takes into acount expected outcomes of observed action pairs.
<code>utils.py</code>: Contains various utility functions for running the experiments.

### human_evaluation/
*:blue_book: Used in **Section 4** of the paper.*
<code>get\_action\_stats.py</code>: Summarizes and reports human evaluation statistics (obtained via AMT) for an action generation task.
<code>get\_consequence\_stats.py</code>: Summarizes and reports human evaluation statistics for a consequence generation task.
<code>get\_norm\_stats.py</code>: Summarizes and reports human evaluation statistics for a norm generation task.

---

### Citation

```
TODO
```
