from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


from data.db import get_training_data
from modeling.config import TARGET


def main():
    df = get_training_data()
    df = df.drop(labels=['client_id', 'id', 'meta__inserted_at'], axis=1)
    df[TARGET] = np.where(df[TARGET] == 'yes', 1, 0)
    train_df = df.sample(frac=0.7, random_state=200)
    test_df = df.drop(train_df.index)
    train_df.to_csv('train_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)
    rd = Reader(sep=',')
    df = rd.train_test_split(['train_df.csv', 'test_df.csv'], TARGET)
    opt = Optimiser(scoring='neg_log_loss', n_folds=5)
    opt.evaluate(None, df)


if __name__ == "__main__":
    main()
