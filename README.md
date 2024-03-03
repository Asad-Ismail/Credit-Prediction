# Credit-Prediction

Prediction of whether a customer will apply for credit.

See Credit_Application.pptx for detail information.

## Getting Started

To get started with this project, first clone the repository to your local machine.

```bash
git clone https://github.com/Asad-Ismail/Credit-Prediction
cd Credit-Prediction
```

## Install Requirements 

```bash
pip install -r requirements.txt

```

### Run Data preprocessor

```bash
python src/data_preprocessing.py
```


### Train Model

```bash
python src/train_model.py
```

### Test Model

```bash
python src/inference.py
```

### Serve Model

```bash
python src/serve_model.py
```

To test the serving see test_serving.py
