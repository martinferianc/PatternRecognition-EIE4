# Pattern-Recognition-EIE4
Supplementary repository for Pattern Recognition EE4-68 at Imperial College London. This project was a part of coursework for module EE4-68 Pattern Recognition at Imperial College London, [Link](http://intranet.ee.ic.ac.uk/electricalengineering/eecourses_t4/course_content.asp?c=EE4-68&s=E3#start).

## Coursework 2:
### Coursework on Representation and Distance Metrics Learning
This work looks at finding and demonstrating the parameters that lead to optimal performance and validation of the parameters by presenting supporting results.

We performed k-nearest neighbour (kNN) and k-Means retrieval experiments according to standard practices in pattern recognition. We used retrieval error (i.e. @rank1-10) as the performance metric to evaluate different methods.

The baseline explores k-Nearest Neighbours (kNN), employing Euclidean and Manhattan distance and k-Means algorithms. This work further explores dimensionality reduction techniques, Neighbourhood Components analysis and neural networks for learning a distance metric that improves the baseline k-NN approach. The best result was achieved by NCA with the resultant accuracy 49\% and 0.45 mAP at rank 1.

### Structure
```bash
# Inside Coursework 2 folder
.
├── README.md
├── Resources           # Instructions and papers
├── data                # Where the raw and processed matrix data is stored
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
├── kmeans.py          # k-Means method
├── nca.py
├── nn_network.py      # Neural network supplementary script
├── nn_preprocess.py   # Data pre-processing for neural network
├── post_process.py    # Plotting the Confusion matrix
├── pre_process.py     # Pre-processing for other methods
├── process.py         # Weighting and voting
├── results            # All the figures which are going to be generated will be stored here
└── weights            # Model/weights for the neural network

```
The main directory contains all the the source code to generate the best models for both considered methods (k-NN and k-Means) and generate all the results used in the report.

All the outcomes are summarised in the [report](CW2_Report.pdf).

## Building & Running
To train and generate all the figures for the winning models just run the script below inside the main directory, make sure that the data for this coursework is included in the `data/raw` directory, such that inside `data/raw`:

```bash
.
├── README.txt
├── cuhk03_new_protocol_config_labeled.mat
├── feature_data.json
└── images_cuhk03
```

Note that the scripts are only going to work for Python lower than 3.7.x, please use Python 3.6.4.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 kNN.py
python3 kmeans.py
```

and the figures will be found in `results/` and the numerical results will be printed in your terminal.

# Authors:

- Martin Ferianc (mf2915@ic.ac.uk)

- Alexander Montgomerie-Corcoran (am9215@ic.ac.uk)

2018
