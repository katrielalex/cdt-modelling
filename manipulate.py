import logging
import numpy as np
import os
import pandas as pd
import pickle
import pylab as pl
import scipy.io.arff as arff
import sklearn.svm, sklearn.decomposition

logging.basicConfig(level=logging.DEBUG)

COLUMN_TO_HUMAN_NAME_MAPPING = {
    0: "server_port",
    1: "client_port",
    4: "median_interarrival_time",
    18: "median_data_ip",
    30: "total_packets_ab",
    31: "total_packets_ba",
    212: "percent_bulk",
    214: "percent_idle",
    248: "class"
}

def _load(filename):
    logging.getLogger("loader").debug("Loading %s", filename)
    """Load an ARFF file from the traffic dataset into a pandas
    dataframe, selecting a few columns that look interesting."""
    
    df = pd.DataFrame(arff.loadarff(filename)[0]) # [0] is data, [1] metadata
    df.columns = range(len(df.columns)) # arff.loadarff mangles the indices
    df = df[COLUMN_TO_HUMAN_NAME_MAPPING.keys()]
    df.columns = map(COLUMN_TO_HUMAN_NAME_MAPPING.get, df.columns)
    df.drop_duplicates(inplace=True)

    return df

def load(*args):
    if not args:
        raise ValueError("You must supply at least one filename to load.")
    elif len(args) == 1 and isinstance(args[0], basestring):
        return _load(args[0])
    else:
        dfs = pd.concat(map(_load, args[0]))
        dfs.index = range(len(dfs.index))
        return dfs

def split(df, proportion=10):
    """ Split a dataframe randomly into training and testing data,
    and return them in separate frames. """
    test_rows = np.random.choice(df.index.values, len(df) // proportion)
    return df.drop(test_rows), df.ix[test_rows]

def train(df, CLF=None):
    """ Train an SVC on a dataframe, using the "class" column as the class
    to be predicted and the others as features. """
    if CLF is None:
        CLF = sklearn.svm.LinearSVC
        
    clf = CLF()
    clf.fit(df.drop("class", axis=1), df["class"])

    return clf

def resample_to(df, cls):
    """ Resample `df` so every entry has the same number of objects of class `cls`
    as `cls`, duplicating and sampling rows as necessary. """

    counts = df["class"].value_counts()
    sample_ix = lambda g: g.ix[np.random.choice(g.index, counts[cls])]

    # .apply gives a multiindex<class, index>. To fix this, drop the class column
    # and reset_index(), which unstacks the multiindex into two columns (replacing
    # said column in the process), and then reindex onto the original index.
    return df.groupby("class").apply(sample_ix).drop("class", axis=1).reset_index().set_index("level_1")

def test(clf, df):
    """ Test the classifier `clf` on the dataframe `df`, using the "class" column
    as the value to be predicted. """
    results = pd.DataFrame({
        "prediction": clf.predict(df.drop("class", axis=1)),
        "ground": df["class"]
        }, index=df.index)

    summary = results.apply(pd.Series.value_counts).fillna(0)

    # Measure recall and precision for each class. These are defined similarly:
    #   - precision: what % of objects classified as `cls` actually were
    #   - recall: what % of `cls` objects were classified thus
    percent_correct = lambda s: s.fillna(0).to_dict().get(s.name, 0) / float(s.sum()) * 100
    measure = lambda s: s.value_counts().unstack().apply(percent_correct)
    summary["precision"] = measure(results.groupby("ground").prediction)
    summary["recall"] = measure(results.groupby("prediction").ground)

    # If nothing is classified as `cls`, the above makes it NaN instead of 0.
    summary = summary.fillna(0).sort("ground", ascending=False)
    
    print summary
    return summary
        
def pca(df, components=2):
    log = logging.getLogger("pca")
    
    # Don't PCA the classification

    # PCA the data -- project down to the principal two components
    log.debug("Running PCA to project to %s components", components)
    pca = sklearn.decomposition.RandomizedPCA(n_components=components)
    pcaed = pd.DataFrame(pca.fit(df.drop("class", axis=1)).transform(df.drop("class", axis=1)))

    # Drop nulls and outliers, and replace the classification
    pcaed = pcaed[pcaed[0] > 0].dropna(how="any", axis=0)
    pcaed["c"] = df["class"]

    # Choose a linearly spaced rainbow color for each class
    log.debug("Selecting colors")
    values = sorted(pcaed.c.unique())
    colormap = dict(zip(values, pl.cm.rainbow(np.linspace(0, 1, len(values)))))
    
    # Plot it
    log.debug("Plotting")
    axes = pl.subplot(111)
    for cls, indices in pcaed.groupby('c').groups.items():
        data = pcaed.ix[indices]
        axes.scatter(data[0], data[1], c=colormap[cls], label=cls)

    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=3, fancybox=True, shadow=True)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    pl.show()

def pie(df):
    classes = (df['class'].value_counts() / float(len(df)) * 100)
    classes.sort(ascending=False)
    classes = classes[classes > 1]

    pl.pie(list(classes) + [100 - classes.sum()], labels=list(classes.index) + ['OTHER'])
    pl.show()

def main(skip_cache=False):
    log = logging.getLogger("main")
    
    if os.path.exists("cache.csv") and not skip_cache:
        log.debug("Found cache, loading it...")
        df = pd.DataFrame.from_csv("cache.csv").drop_duplicates()
    else:
        log.debug("Loading from files...")
        df = load(glob.glob("*.arff"))

    log.debug("...done. Splitting into training and testing subsets...")
    training_df, testing_df = split(df)
    del df

    log.debug("...done. Training a classifier (warning: timesink!)...")
    #clf = SGDClassifier(loss='hinge', penalty='l2')
    #clf = train(training_df)
    clf = pickle.load(open("clf.pickle"))
    #clf = None

    log.debug("...done. Testing the classifier...")
    results = test(clf, testing_df)

    log.debug("...done. Goodbye!")
    return training_df, testing_df, clf, results

if __name__ == "__main__":
    training_df, testing_df, clf, results = main()
    # pie(training_df)
    # pca(training_df)
