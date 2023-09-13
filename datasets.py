import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def feature_standardization(X):
    """
    Performs feature standardization on the input feature matrix X.

    Args:
        X (numpy.ndarray): Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.

    Returns:
        numpy.ndarray: Standardized feature matrix of shape (m, n).

    """

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    X_standardized = scaler.fit_transform(X)

    return X_standardized

def feature_normalization(X):
    """
    Performs feature normalization on the input feature matrix X to the range [-1, 1].

    Args:
        X (numpy.ndarray): Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.

    Returns:
        numpy.ndarray: Normalized feature matrix of shape (m, n).

    """

    # Create a MinMaxScaler object with feature range set to (-1, 1)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    X_normalized = scaler.fit_transform(X)

    return X_normalized

def return_data_matrix(path,column_names,transform):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    if transform == 1:
        X = feature_standardization(X)
    elif transform == 2:
        X = feature_normalization(X)     
    X = pd.DataFrame(X,columns=column_names)
    return X,y

def iris(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/IRIS/iris.csv"
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return return_data_matrix(path,feature_names,transformation)

def wine(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/WINE/wine.csv"
    feature_names = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
    ]
    return return_data_matrix(path,feature_names,transformation)

def pima_diabetes(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/Pima%20Indians%20Diabetes/diabetes.csv"
    feature_names = [
    "Number of times pregnant",
    "Plasma glucose concentration",
    "Diastolic blood pressure",
    "Triceps skinfold thickness",
    "2-Hour serum insulin",
    "Body mass index",
    "Diabetes pedigree function",
    "Age"
    ]
    return return_data_matrix(path,feature_names,transformation)

def seeds(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/SEEDS/seeds.csv"
    feature_names = [
    "Area",
    "Perimeter",
    "Compactness",
    "Length of kernel",
    "Width of kernel",
    "Asymmetry coefficient",
    "Length of kernel groove"
    ]
    return return_data_matrix(path,feature_names,transformation)

def glass(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/Glass/glass.csv"
    feature_names = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
    return return_data_matrix(path,feature_names,transformation)

def yeast(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/Yeast/yeast.csv"
    feature_names = ["Mcg","Gvh","Alm","Mit","Erl","Pox","Vac","Nuc"]
    return return_data_matrix(path,feature_names,transformation)

def ceramic(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/Chemical%20Composition%20of%20Ceramic%20Samples/Chemical%20Composion%20of%20Ceramic.csv"
    feature_names = ["Na2O", "MgO", "Al2O3", "SiO2", "K2O", "CaO", "TiO2", "Fe2O3", "MnO", "CuO", "ZnO", "PbO2", "Rb2O", "SrO", "Y2O3", "ZrO2", "P2O5"]
    df = pd.read_csv(path)
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    if transformation == 1:
        X = feature_standardization(X)
    elif transformation == 2:
        X = feature_normalization(X)     
    X = pd.DataFrame(X,columns=feature_names)
    return X,y

def wdbc(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/%20Breast%20Cancer%20Wisconsin%20(Diagnostic)/wdbc.csv"
    feature_names = [
    "Radius (mean)",
    "Texture (mean)",
    "Perimeter (mean)",
    "Area (mean)",
    "Smoothness (mean)",
    "Compactness (mean)",
    "Concavity (mean)",
    "Concave points (mean)",
    "Symmetry (mean)",
    "Fractal dimension (mean)",
    "Radius (standard error)",
    "Texture (standard error)",
    "Perimeter (standard error)",
    "Area (standard error)",
    "Smoothness (standard error)",
    "Compactness (standard error)",
    "Concavity (standard error)",
    "Concave points (standard error)",
    "Symmetry (standard error)",
    "Fractal dimension (standard error)",
    "Radius (worst)",
    "Texture (worst)",
    "Perimeter (worst)",
    "Area (worst)",
    "Smoothness (worst)",
    "Compactness (worst)",
    "Concavity (worst)",
    "Concave points (worst)",
    "Symmetry (worst)",
    "Fractal dimension (worst)"
    ]

    return return_data_matrix(path,feature_names,transformation)

def banknote(transformation = 0):
    path = "https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/%20Banknote%20authentication/banknote_authentication.csv"
    feature_names = ["Variance","Skewness","Curtosis","Entropy"]
    return return_data_matrix(path,feature_names,transformation)

def mnist(mode):
    import keras
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if mode == 'train':
        return train_X, train_y
    else:
        return test_X, test_y

def voting():
    path = 'https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/Congressional%20Voting%20Records/voting_with_imputation.csv'
    feature_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    return return_data_matrix(path,feature_names,0)

def bcwo():
    path = 'https://raw.githubusercontent.com/Srinath7008/Datasets/main/Datasets/%20Breast%20Cancer%20Wisconsin%20(Original)/breast-cancer-wisconsin_processed.csv'
    feature_names = ["1", "2", "3", "4", "5", "6", "7", "8","9"]
    return return_data_matrix(path,feature_names,0)