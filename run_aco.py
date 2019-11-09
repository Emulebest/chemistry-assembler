import pandas as pd

from sklearn import linear_model

from aco.ant_colony import BinaryFeatureSelectionAntColony


def main():
    xs = pd.read_csv('x.dat', sep='\t', header=0, skipinitialspace=True)
    ys = pd.read_csv('y.dat', sep='\t', header=0, skipinitialspace=True)
    regr = linear_model.LinearRegression()
    x = xs.iloc[:, 1:500]
    y = ys["kj/mol"]

    try:
        regr.fit(x, y)
        # TODO: this is calculated as 1, IDK why
        print(regr.score(x, y))
    except Exception as e:
        print(e)
    # xs['Ys'] = ys["kj/mol"].values
    # del xs["Name"]
    # ant_colony = BinaryFeatureSelectionAntColony(3, 4, 0.05, xs, 2, 25)
    # result_path = ant_colony.run()
    # print(f"Result {result_path}")


if __name__ == "__main__":
    main()
