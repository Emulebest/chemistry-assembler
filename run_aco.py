import pandas as pd

from sklearn import linear_model

from aco.ant_colony import BinaryFeatureSelectionAntColony


def main():
    xs = pd.read_csv('x.dat', sep='\t', header=0, skipinitialspace=True)
    ys = pd.read_csv('y.dat', sep='\t', header=0, skipinitialspace=True)
    # regr = linear_model.LinearRegression()
    # x = xs.iloc[1:32, 1:15]
    # y = ys.iloc[1:32, :]
    # x_rest = xs.iloc[32:76, 1:15]
    # y_rest = ys.iloc[32:76, :]
    #
    # try:
    #     regr.fit(x, y)
    #     # TODO: this is calculated as 1, IDK why
    #     print(regr.score(x_rest, y_rest))
    # except Exception as e:
    #     print(e)
    xs['Ys'] = ys["kj/mol"].values
    del xs["Name"]
    ant_colony = BinaryFeatureSelectionAntColony(100, 4, 0.05, xs, 24, 2)
    result_path = ant_colony.run()
    print(f"Result {result_path}")


if __name__ == "__main__":
    main()
