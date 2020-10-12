# PRGNN_fMRI

## Requirments 
Installation 1:
```
pip install -r requirements.txt
```
Installation 2:
Install pytorch geometric 1.6.0 through
```
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/
```

## How to run the code ? 

### Custermize Dataset
Please follow the guideline in pytorch geometric 1.6.0 and create your own preprocessed data. 
### Run main function to classify brain graphs
```
python main.py --net ${GNN} --dataroot ${YOUR_PROCCESSED_DATA_DIR}
```

## To Do
- [ ] Provide synthetic data.
