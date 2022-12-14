# npretor_udacity_deepRL

<b><i>This Deep Q network maps the 37 state spaces of the Unity environment to a best policy for choosing the 4 possible actions in the environment. More detail is given in the [report](Report.md). </b></i> The majority of the project code is taken from this Unity project: https://github.com/udacity/Value-based-methods




# 1. Installation 
1. Clone the repo 
    ```
    git clone https://github.com/npretor/npretor_udacity_deepRL
    cd npretor_udacity_deepRL
    ```

2. Install conda if not installed
3. a. Install dependancies: Installing the unity dependencies was a pain (running OS X Big Sur). That can be another blog post. Install non-unity reqs using: 
    ```
    conda create --name deepqn --file package-list.txt
    ```

    If using jupyterlab: 
    ```
    python3 -m ipykernel install --user --name=deepqn
    source activate deepqn
    python3 -m jupyterlab 
    ```
    If using bash
    ```
    source activate deepqn
    ```


# 3. Training - 
### This runs without the [Weights and Biases](http://wandb.ai) module. Use the jupyter notebook for weights and biases tracking, but you will need to login and config wandb yourself. 
```
python3 Navigation_Training.py
```


# 4. Demo 
```
python3 Navigation_Demo.py
```


# Report 
- [Report](Report.md)
