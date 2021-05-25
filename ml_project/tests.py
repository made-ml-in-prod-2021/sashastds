import pytest
from textwrap import dedent
from faker import Faker
import pandas as pd
from src.parametrization import SplittingParams, TrainingParams, InferenceParams
from src.parametrization import read_config, load_training_params, load_inference_params
from src.transformers import CategoricalEncoder, HotEncoder
from src.training import make_split, train_pipeline
from src.inference import inference_pipeline

TARGET_NAME = 'target'

@pytest.fixture
def fake_dataset(tmpdir):
    """
    makes fake dataset
    """
    faker = Faker()
    faker.set_arguments('age', {'min_value': 10, 'max_value': 100})
    faker.set_arguments('sex', {'min_value': 0, 'max_value': 1})
    faker.set_arguments('cp', {'min_value': 0, 'max_value': 3})
    faker.set_arguments('trestbps', {'min_value': 90, 'max_value': 210})
    faker.set_arguments('chol', {'min_value': 120, 'max_value': 600})
    faker.set_arguments('oldpeak', {'min_value': 0, 'max_value': 7})
    faker.set_arguments(TARGET_NAME, {'min_value': 0, 'max_value': 1})
    fake_data = faker.csv(
        header=(
            'age', 'sex', 'cp',
            'trestbps', 'chol',
            'oldpeak', TARGET_NAME
            ),
        data_columns=(
            '{{pyint:age}}','{{pyint:sex}}', '{{pyint:cp}}',
            '{{pyint:trestbps}}', '{{pyint:chol}}',
            '{{pyfloat:oldpeak}}','{{pyint:target}}',
            ),
        num_rows=100,
        include_row_ids=False).replace('\r', '')


    fout = tmpdir.join('fake_data.csv', encoding = 'utf-8')
    fout.write(fake_data)
    return fout
    
@pytest.fixture
def splitting_params():
    return SplittingParams(random_state=8, test_size=0.2)
    
def test_can_split_dataset(fake_dataset, splitting_params):
    data = pd.read_csv(fake_dataset)
    train, test = make_split(data, TARGET_NAME, splitting_params)
    assert train.shape[0] > 0 and test.shape[0] > 0 and train.shape[0] / test.shape[0] > 3.5

    
def test_transformer_can_make_label_encoding(fake_dataset, splitting_params):
    data = pd.read_csv(fake_dataset)
    encoder = CategoricalEncoder(
        cat_na_fill_value="(UNK)",
        map_unknown_to_na=True,
        recognize=False,
        verbose=False,
    )
    train, test = make_split(data, TARGET_NAME, splitting_params)
    cat_features = ['sex', 'cp']
    train_cat = encoder.fit_transform(train[cat_features])
    test_cat = encoder.transform(test[cat_features])
    nunique_max = max(len(data[f].unique()) for f in cat_features)
    assert train_cat.min().min() == 0 and train_cat.max().max() <= nunique_max + 1 and test_cat.min().min() == 0 and test_cat.max().max() <= nunique_max + 1
    
    
def test_transformer_can_make_onehot_encoding(fake_dataset, splitting_params):
    data = pd.read_csv(fake_dataset)
    encoder = HotEncoder(
        force_nan_and_unknown_category=True,
        map_unknown_to_na=True,
        cat_na_fill_value="(UNK)",
        drop=None,
        recognize=False,
        sparse=False,
        verbose=False,
    )
    train, test = make_split(data, TARGET_NAME, splitting_params)
    cat_features = ['sex', 'cp']
    train_cat = encoder.fit_transform(train[cat_features])
    test_cat = encoder.transform(test[cat_features])
    total_dims = sum(len(data[f].unique()) + 1 for f in cat_features)
    assert train_cat.shape[1] == total_dims and test_cat.shape[1] == total_dims
    assert max(train_cat.sum(axis = 1)) == len(cat_features) and min(train_cat.sum(axis = 1)) == len(cat_features)
    assert max(test_cat.sum(axis = 1)) <= len(cat_features) and min(test_cat.sum(axis = 1)) <= len(cat_features)
    
    
@pytest.fixture
def train_config_file(tmpdir):
    config = dedent("""\
        # @package _group_
        path_dataset: data/raw/heart_disease_uci.csv
        path_to_models: models/
        path_to_report: models/report/
        splitting_params:
          test_size: 0.2
          random_state: 8
        feature_params:
          target_name: target
        encoder_params:
          encoder_type: categorical
        classifier_params:
          classifier_type: lgbm
          random_state: 8
          n_jobs: 2
          n_estimators: 10
          max_depth: 5
          num_leaves: 15
    """)
    fout = tmpdir.join('tr_config.yaml', encoding = 'utf-8')
    fout.write(config)
    return fout
 
def test_can_train_pipeline(fake_dataset, train_config_file, tmpdir, capsys):
    data = pd.read_csv(fake_dataset)
    
    train_config = read_config(train_config_file)
    train_config['path_dataset'] = str(fake_dataset)
    train_config['path_to_models'] = str(tmpdir)
    train_config['path_to_report'] = str(tmpdir)
    train_pipeline(train_config)
    out, err = capsys.readouterr()
    assert 'Finished' in out and out.count('Saving') >= 5

@pytest.fixture
def inference_config_file(tmpdir):
    config = dedent("""\
        # @package _group_
        path_dataset: data/raw/heart_disease_uci.csv
        path_to_models: models/
        path_to_predictions: data/results/predictions.csv
        binary: false
        cutoff: 0.48
        path_to_report: data/results/
        target_name: target
    """)
    fout = tmpdir.join('inf_config.yaml', encoding = 'utf-8')
    fout.write(config)
    return fout
    
def test_can_train_and_inference_pipeline(fake_dataset, train_config_file, inference_config_file, tmpdir, capsys):
    data = pd.read_csv(fake_dataset)

    train_config = read_config(train_config_file)
    train_config['path_dataset'] = str(fake_dataset)
    train_config['path_to_models'] = str(tmpdir)
    train_config['path_to_report'] = str(tmpdir)
    
    train_pipeline(train_config)
    
    inference_config = read_config(inference_config_file)
    inference_config['path_dataset'] = str(fake_dataset)
    inference_config['path_to_models'] = str(tmpdir)
    inference_config['path_to_report'] = str(tmpdir)
    inference_config['path_to_predictions'] = str(tmpdir) + '/predictions.csv'
    
    inference_pipeline(inference_config)
    
    out, err = capsys.readouterr()
    assert out.count('Finished') == 2 and out.count('Saving predictions') == 1
