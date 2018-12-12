# Pattern-Recognition-EIE4
Supplementary repository for Pattern Recognition EE4-68 at Imperial College London. This project was a part of coursework for module EE4-68 Pattern Recognition at Imperial College London, [Link](http://intranet.ee.ic.ac.uk/electricalengineering/eecourses_t4/course_content.asp?c=EE4-68&s=E3#start).

## Coursework 2:
### Coursework on Representation and Distance Metrics Learning
This work looks at finding and demonstrating the parameters that lead to optimal performance and validation of the parameters by presenting supporting results.

We performed k-nearest neighbour (kNN) and k-Means retrieval experiments according to standard practices in pattern recognition. We used retrieval error (i.e. @rank1, @rank10) as the performance metric to evaluate different methods.

The baseline explores k-Nearest Neighbours (kNN), employing Euclidean and Manhattan distance and k-Means algorithms. This work further explores dimensionality reduction techniques, Neighbourhood Components analysis and neural networks for learning a distance metric that improves the baseline k-NN approach. The best result was achieved by NCA with the resultant accuracy 49\% and 0.45 mAP at rank 1.

### Structure
```bash
# Inside Coursework 2 folder
.
├── README.md
├── Resources           # Instructions and papers
├── data                # Where the rand and processed matrix data is stored
│   ├── processed
│   └── raw
├── kNN.py              # Main script generating the results
│                       # kNN_*.py Respective methods
├── kNN_euclidean.py
├── kNN_improved_NN.py
├── kNN_improved_PCA.py
├── kNN_improved_RCA_NCA.py
├── kNN_improved_cosine.py
├── kNN_manhattan.py
├── kmeans.py
├── nca.py
├── nn_network.py
├── nn_preprocess.py   # Data pre-processing for neural network
├── post_process.py    
├── pre_process.py     # Pre-processing for other methods
├── process.py
├── results            # All the figures which are going to be generated will be stored here
└── weights            # Model for the neural network

```
The main directory contains all the the source code to generate the best models for both considered models and generate all the results used in the report.

All the outcomes are summarised in the [report](CW2_Report.pdf).

## Building & Running
To train and generate all the figures for the winning models just run inside the main directory, make sure that the data is included in the `data/raw` directory if you want to generate completely new results:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python3 kNN.py
python3 kmeans.py
```

and the figures will be found in `results/` and the numerical results will be printed in your terminal.

# Authors:

- Martin Ferianc (mf2915@ic.ac.uk)

- Alexander Montgomerie-Corcoran (am9215@ic.ac.uk)

2018
