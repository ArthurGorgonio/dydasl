# DyDaSL

Dynamic Data Stream Learner is a Semi-supervised framework to classify data streams.

This approach uses a chunk-based strategy to perform the classification process.
An ensemble is trained with _n_ classifiers (default=10) in the _n_-first iterations.
Then, the detection module starts its processes using the chunk odf current
iteration to detect a drift, based in one of the following strategies:
- **Fixed Threshold**: This use a fixed threshold parameter to identify the drift
occurence based on ensemble performance in the labelled data of the current
chunk (default 80%).
- **Non-Weighted Threshold**: This use a flexible threshold parameter based on
ensemble's performance in the labelled data of the current chunk (default 80%).
The new threshold is calculated using the performance of the ensemble on labelled
data of the actual chunk.
- **Weighted Threshold**: Similar to the **Non-Weighted Threshold**. The major
difference is related to use the weighted strategy to benefit the newest classifiers
of the ensemble.
- **Statistical Threshold**: This detection module uses statistical tests between
two chunks (_t_ and _t-1_ iterations) to indentify the drift, if the distribution
changes. Default uses `Kolmogorov–Smirnov` statistical test.
Lastely, the reaction modules are responsable to update the ensemble when a dritf
is detected in the stream data. This module uses the train and/or remove a classifier
approch to update the ensemble, following one of these strategies:
- **Exchage**: Exchange the worst classifier (always one) of the ensemble based
on ensemble prediction.
- **Volatile Exchange**: Exchange the worst classifier(s) (can be one or more
depending if these classifiers has the same classification effectiveness) of the
ensemble based on ensemble prediction.
- **Pareto's Frontier**: Calculate the Pareto's frontier based on two or more
metrics then select all classifers of the first Pareto frontier (non-domined
classifiers).

## Cite this Study
```latex
@article{gorgonio2023icmla,
  author    = {Arhtur Gorg{\^o}nio and Anne Canuto and Arthur Medeiros and Karliane Vale and Flavius Gorg{\^o}nio},
  title     = {Two Efficient Training Strategies for DyDaSL: A Dynamic Data Stream Learner Framework with Semi-Supervised Learning},
  year      = {2023},
  journal   = {International Conference on Machine Learning and Applications},
  pages     = {2000--2005},
  doi       = {10.1109/ICMLA58977.2023.00302}
}
```
