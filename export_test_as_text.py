import pandas as pd
import numpy as np

BEST_TEST_PATH = 'data/test_experiment_4.csv'
EXPORT_FILE_NAME = 'output.txt'


def _run():
    df = pd.read_csv(BEST_TEST_PATH, header=None)

    np.savetxt(EXPORT_FILE_NAME, df[0].values, fmt='%d')


if __name__ == '__main__':
    _run()
