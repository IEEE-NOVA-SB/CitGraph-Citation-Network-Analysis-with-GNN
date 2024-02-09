# CitGraph: Citation Network Analysis with GNN

![Code review](https://github.com/IEEE-NOVA-SB/CitGraph-Citation-Network-Analysis-with-GNN/blob/main/code_review.png)
CitGraph employs Graph Neural Networks (GNNs) to analyze citation networks, specifically focusing on node classification tasks. 

Leveraging PyTorch Geometric and the Planetoid dataset, this project aims to classify nodes within citation graphs, offering insights into citation patterns and relationships.



## Project Description

### What your application does?

This project implements a Graph Neural Network (GNN) for node classification using PyTorch Geometric. 

It leverages the Planetoid dataset to demonstrate the application of GNNs in classifying nodes in a citation network. 
 
### Why you used the technologies you used?

The project utilizes PyTorch Geometric due to its efficient implementation of GNN layers and convenient dataset handling. 

PyTorch Lightning is employed for streamlined training and experimentation.    
    
### Some of the challenges you faced and features you hope to implement in the future?

Challenges encountered include dataset preparation and hyperparameter tuning. 

Future features may include implementing more advanced GNN architectures and exploring additional datasets for node classification tasks.

# Table of Contents
### [ How to Install and Run the Project ](#How_to_install)

### [ How to Use the Project ](#How_to_use)

### [ Include Credits, Authors and acknowledgment for contributions ](#credits)

----



<a name="How_to_install">

# How to Install and Run the Project

### 1. Create a Virtual Environment (if not already created):
If you haven't already created a virtual environment for your project, you can do so using virtualenv or venv. Here's an example using venv:

```
python -m venv myenv
```


Replace ```myenv``` with the desired name for your virtual environment.

### 2. Activate the Virtual Environment:
On Windows, activate the virtual environment using:

```
myenv\Scripts\activate
```


On macOS and Linux, use:
```
source myenv/bin/activate
```
Replace ```myenv``` with the name of your virtual environment.


### 3. Install dependencies:
Once the virtual environment is activated, you can install Jupyter Notebook using pip:

```
pip install jupyter ipykernel torch lightning torch_geometric
```
This will install Jupyter Notebook within your virtual environment.

### 4. Verify Jupyter Installation:
To verify that Jupyter Notebook is installed in your virtual environment, you can run:


```
jupyter --version
```

This should display the version of Jupyter Notebook installed within your virtual environment.

### 5. Create a Jupyter Notebook Kernel for the Virtual Environment:
You need to create a Jupyter Notebook kernel that is associated with your virtual environment. This allows you to use the packages installed in your virtual environment within Jupyter Notebook.

#### a. First, activate your virtual environment (if it's not already activated).

#### b. Install the ipykernel package within the virtual environment:

```
pip install ipykernel
```
#### c. Now, you can create a Jupyter Notebook kernel for your virtual environment:


```
python -m ipykernel install --user --name=myenv --display-name="name"
```

Replace ```myenv``` with the name of your virtual environment and choose a suitable display name.

### 6. Start Jupyter Notebook:
Now, you can start Jupyter Notebook from within your virtual environment:

```
jupyter notebook
```
This will open a new Jupyter Notebook session in your web browser, and you should be able to select the "My Virtual Environment" kernel when creating a new notebook. This kernel will use the packages installed in your virtual environment.
</a>

<a name="How_to_use">


#### How to Use the Project

Run all jupyter notebooks cells
</a>


<a name="how_to_contribute">


#### How to Contribute to the Project

Make a pull request

</a>

<a name="credits">

#### Include Credits, Authors and acknowledgment for contributions

</a>

[Tiago Monteiro](https://www.linkedin.com/in/tiago-monteiro-)
