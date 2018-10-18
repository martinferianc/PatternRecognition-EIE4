# Pattern-Recognition-EIE4
Supplementary repository for Pattern Recognition EE4-68 at Imperial College London

## Coursework 1:

## Coursework 2:
### Coursework on Representation and Distance Metrics Learning
This work looks at finding and demonstrating the parameters that lead to optimal performance and validation of the parameters by presenting supporting results.

We performed k-nearest neighbour (kNN) and k-Means retrieval experiments according to standard practices in pattern recognition. We used retrieval error (i.e. @rank1, @rank10) as the performance metric to evaluate different methods. Our baseline approach is kNN on provided features. We used different distance metric learning methods to improve a baseline performance.

This project was a part of coursework for module EE4-68 Machine Learning at Imperial College London, [Link](http://intranet.ee.ic.ac.uk/electricalengineering/eecourses_t4/course_content.asp?c=EE4-68&s=E3#start).

### Structure
```bash
# Inside Coursework 2 folder
.
├── Examples
├── Resources
├── Report.pdf
├── results
├── kmeans.py
├── kNN.py
├── post_process.py
├── pre_process.py
└── data
    ├── processed
    └── raw
```
The main directory contains all the the source code to generate the best models for both considered models and generate all the results used in the report.

All the outcomes are summarized in the [report](CW2_Report.pdf).


## Building & Running
To train and generate all the figures for the winning models just run inside the `src` directory:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python3 Coursework\ 2/kmeans.py
python3 Coursework\ 2/kNN.py
```

and the figures will be found in `results/<Method>/`

## Authors:

- Martin Ferianc

- Alexander Alex Montgomerie-Corcoran

2018
