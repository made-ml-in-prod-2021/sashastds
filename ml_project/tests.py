import pytest
from faker import Faker


@pytest.fixture
def make_fake_dataset(tmpdir):
    """
    makes fake dataset
    """
    faker = Faker()
    faker.set_arguments('age', {'min_value': 1, 'max_value': 100})
    faker.set_arguments('trestbps', {'min_value': 90, 'max_value': 200})
    faker.set_arguments('chol', {'min_value': 110, 'max_value': 570})
    faker.set_arguments('restecg_and_slope', {'min_value': 0, 'max_value': 2})
    faker.set_arguments('thalach', {'min_value': 70, 'max_value': 205})
    faker.set_arguments('oldpeak', {'min_value': 0, 'max_value': 6.8})
    faker.set_arguments('ca', {'min_value': 0, 'max_value': 4})
    faker.set_arguments('thal_and_cp', {'min_value': 0, 'max_value': 3})
    faker.set_arguments('binary', {'min_value': 0, 'max_value': 1})
    fake_data = fake.csv(
        header=("age", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal", "cp", "sex", "target"),
        data_columns=('{{pyint:age}}', '{{pyint:trestbps}}', '{{pyint:chol}}',
                      '{{pyint:binary}}', '{{pyint:restecg_and_slope}}',
                      '{{pyint:thalach}}', '{{pyint:binary}}', '{{pyfloat:oldpeak}}',
                      '{{pyint:restecg_and_slope}}', '{{pyint:ca}}', '{{pyint:thal_and_cp}}',
                      '{{pyint:thal_and_cp}}', '{{pyint:binary}}', '{{pyint:binary}}'),
        num_rows=100,
        include_row_ids=False).replace('\r', '')

    fout = tmpdir.join('fake_data.csv', encoding = 'utf-8')
    fout.write(fake_data)
    return fout