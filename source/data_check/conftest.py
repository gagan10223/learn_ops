import wandb
import pytest
import pandas as pd

print('--------------------------------------------------------------------------------------------')
def pytest_addoption(parser):
    parser.addoption('--csv', action='store')
    parser.addoption('--ref', action='store')
    parser.addoption('--kl_threeshold', action='store')
    parser.addoption('--min_price', action='store')
    parser.addoption('--max_price', action='store')
    
@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type='tests',resume=True)
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail('must provide csv')
    return pd.read_csv(data_path)

@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type='tests',resume=True)
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail('must provide ref')
    return pd.read_csv(data_path)

@pytest.fixture(scope='session')
def kl_threeshold(request):
    kl_threeshold = request.config.option.kl_threeshold
    if kl_threeshold is None:
        pytest.fail('must provide kl')
    return float(kl_threeshold)


@pytest.fixture(scope='session')
def min_price(request):
    kl_threeshold = request.config.option.min_price
    if kl_threeshold is None:
        pytest.fail('must provide min_price')
    return float(kl_threeshold)

@pytest.fixture(scope='session')
def max_price(request):
    max_ = request.config.option.max_price
    if max_ is None:
        pytest.fail('must provide max_price')
    return float(max_)
print('--------------------------------------------------------------------------------------------')
