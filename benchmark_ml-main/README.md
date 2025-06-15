# Description
Background:  Non-invasive Brain-Computer Interface (BCI) studies mostly center on the motor imagery (MI) concept, where multi-channel Electroencephalogram (EEG) signals are collected and characterized by patterns for different imagined tasks. Previous studies put extensive efforts into data-driven techniques to improve classification performance on benchmark datasets; however, other aspects, such as experimental factors, still lack thorough investigation. This pilot study aims to evaluate the effect of different cue-based protocols on within-subject MI-BCI baseline performance to better guide the experimental instructions on a specific group of users.  
 
Materials and Method:  An Emotiv EEG headset kit integrated into the Lab-Streaming-Layer (LSL) was used for data acquisition. Three PsychoPy-based protocols were designed, namely, G1, G2, and G3, incorporating different visual instructions of image-cue, arrow-cue, and arrow-cue-feedback utilizing Event-Related (de)Synchronization (ERD/ERS) demonstration, respectively. Imagery data (left/right hand/foot) from 12 healthy college participants (age 20~22, five females) were collected (15 trials/task/run) and randomly allocated for each designated protocol. A processing framework was implemented using a conventional Lasso-based sparse Filter Bank Common Spatial Pattern (SFBCSP) for feature extraction/selection and Linear Discriminant Analysis (LDA) for classification to assess the baseline performance. Average ROC (5-fold cross-validation) was calculated for the upper-limb binary model of each run with different non-overlapping time segments. Statistical non-parametric tests were used for within-group and cross-group comparative analysis.

# ml_benchmark
python benchmark_ml-main/run.py

# analysis_main 
* Run the script

```python main.py --mode survey```

```python main.py --mode within_group```

```python main.py --mode between_group```

