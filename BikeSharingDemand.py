import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mstats, stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_log_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV  # , cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
# from xgboost import XGBRegressor
from sklearn.svm import SVR


class BikeSharingDemand:
    def __init__(self, train, test, flag):
        self.df_train, self.df_test = pd.read_csv(train), pd.read_csv(test)
        self.debug(self.df_train)
        # self.visualize_data()
        print('Stage 1: transforming data...')
        self.df_train, self.df_test = self.transform_data(self.df_train, True), self.transform_data(self.df_test, False)
        # self.visualize_opt_components_number(self.df_train)
        self.debug(self.df_train)
        if flag:
            y, X = self.df_train['count'], self.df_train.drop(['count'], axis=1)
            self.local_evaluation(X, y)
        else:
            self.kaggle_submission()

# Simple Preview of Data
    @staticmethod
    def debug(df):
        with pd.option_context('display.max_rows', 25, 'display.max_columns', 63): print(df.head())

# Renaming, Dropping Columns and Changing to Categorical
    @staticmethod
    def transform_data(df, flag):
        df.rename(columns={'weathersit': 'weather', 'mnth': 'month', 'hr': 'hour', 'yr': 'year', 'hum': 'humidity', 'cnt': 'count'}, inplace=True)
        if flag:
            df = df.drop(['registered', 'casual'], axis=1)
        df = df.drop(['atemp', 'windspeed'], axis=1)
        df['season'] = df.season.astype('category')
        df['year'] = df.year.astype('category')
        df['month'] = df.month.astype('category')
        df['hour'] = df.hour.astype('category')
        df['holiday'] = df.holiday.astype('category')
        df['weekday'] = df.weekday.astype('category')
        df['workingday'] = df.workingday.astype('category')
        df['weather'] = df.weather.astype('category')
        df['humidity'] = ((df['humidity']*100).astype('int64')).astype('category')
        return df

# PCA attempt
    @staticmethod
    def visualize_opt_components_number(dataframe):
        dataVisual = dataframe.values
        scaler = MinMaxScaler(feature_range=[0, 1])
        data_rescaled = scaler.fit_transform(dataVisual[1:, 0:8])
        pca = PCA().fit(data_rescaled)
        # Plotting the Cumulative Summation of the Explained Variance
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title('Bike sharing Dataset Explained Variance')
        plt.show()

    def pca_dim_reduction(self, dataframe, components):
        pca = PCA(n_components=components)
        y = dataframe['count']
        dataframe = dataframe.drop(['count'], axis=1)
        X = pca.fit_transform(dataframe)
        self.local_evaluation(X, y)

# Scaling data
    @staticmethod
    def standard_scaling(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    @staticmethod
    def robust_scaling(data):
        scaler = RobustScaler()
        return scaler.fit_transform(data)

# One-Hot Encoding
    @staticmethod
    def one_hot_encoding(label, X):
        onehot = pd.get_dummies(X[label], prefix=label, drop_first=True)
        return pd.concat([X, onehot], axis=1)

    @staticmethod
    def fill_missing_column(df):
        missing = [0] * (df.shape[0])
        df['weather_4'] = missing
        return df

# Graphical Representations
    def visualize_data(self):
        # humidity quartiles
        sn.boxplot(x=self.df_train['hum'])
        plt.show()
        # count quartiles
        sn.boxplot(x=self.df_train['cnt'])
        plt.show()
        # weather quartiles
        sn.boxplot(x=self.df_train['weathersit'])
        plt.show()
        # graphical representation of weather in count
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(self.df_train['weathersit'], self.df_train['cnt'])
        ax.set_xlabel('weather')
        ax.set_ylabel('cnt')
        plt.show()
        # count distribution
        sn.distplot(self.df_train['cnt'])
        plt.show()
        # log(count) distribution
        sn.distplot(np.log(self.df_train['cnt']))
        plt.show()
        # temp, atemp, humidity and windspeed quartiles
        sn.boxplot(data=self.df_train[['temp', 'atemp', 'hum', 'windspeed']])
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.show()
        # casual, registered, count quartiles
        sn.boxplot(data=self.df_train[['casual', 'registered', 'cnt']])
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.show()

        # total boxplots of some important features
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(12, 10)
        sn.boxplot(data=self.df_train, y="cnt", orient="v", ax=axes[0][0])
        sn.boxplot(data=self.df_train, y="cnt", x="season", orient="v", ax=axes[0][1])
        sn.boxplot(data=self.df_train, y="cnt", x="hr", orient="v", ax=axes[1][0])
        sn.boxplot(data=self.df_train, y="cnt", x="workingday", orient="v", ax=axes[1][1])

        axes[0][0].set(ylabel='Count', title="Box Plot On Count")
        axes[0][1].set(xlabel='', ylabel='Count', title="Box Plot On Count Across Season")
        axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count', title="Box Plot On Count Across Hour Of The Day")
        axes[1][1].set(xlabel='Working Day', ylabel='Count', title="Box Plot On Count Across Working Day")
        plt.show()

# Handling Outiers
    def zscore_remove_outliers(self):
        z = np.abs(stats.zscore(self.df_train['count']))
        threshold = 3
        print(np.where(z > threshold))
        self.df_train = self.df_train[(z < 3)]

    def quantiles_remove_outliers(self, df_train):
        print("Samples in train set with outliers: {}".format(len(self.df_train)))
        q1 = df_train['count'].quantile(0.25)
        q3 = df_train['count'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        train_preprocessed = df_train.loc[(df_train['count'] >= lower_bound) & (df_train['count'] <= upper_bound)]
        print("Samples in train set without outliers: {}".format(len(train_preprocessed)))
        sn.distplot(train_preprocessed['count'])
        plt.show()
        return train_preprocessed

    def winsorizing(self, df):
        return df.apply(self.using_mstats, axis=0)

    @staticmethod
    def using_mstats(s):
        return mstats.winsorize(s, limits=[0.25, 0.25])

# Custom RMSLE Scorer for GridSearchCV
    def make_scorer_for_GridSearch(self):
        return make_scorer(self.rmsle, greater_is_better=False, size=10)

    @staticmethod
    def rmsle(predicted, actual, size):
        return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1))) / float(size))

    @staticmethod
    def rmsle2(y, y_, convertExp=True):
        if convertExp:
            y = np.exp(y),
            y_ = np.exp(y_)
        log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
        log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
        calc = (log1 - log2) ** 2
        return np.sqrt(np.mean(calc))

    # LOCAL PREDICTIONS AND EVALUATION
    def local_evaluation(self, X, y):
        print('Stage 2: performing one_hot encoding...')
        X = self.one_hot_encoding('weather', self.one_hot_encoding('season', self.one_hot_encoding('year', self.one_hot_encoding('month', X))))
        X.drop(['season', 'weather', 'year', 'month'], inplace=True, axis=1)
        # self.debug(X)
        self.final_model(X, y)
        print("\n\nModels that produced best results using GridSearchCV to tune parameters, one-hot encoding and log transformation : \n")
        self.best_model_scores(X, y)
        print("\nRest of the results using GridSearchCV to tune parameters, one-hot encoding and log transformation: \n")
        self.rest_model_scores(X, y)

    def final_model(self, X, y):
        print('Stage 3: fitting best models...')
        bagging_base_tree = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', min_samples_leaf=1, min_samples_split=2, splitter='best')
        model1 = ExtraTreesRegressor(bootstrap=False, max_features='auto', n_estimators=3000, max_depth=None, min_samples_split=5)
        model2 = RandomForestRegressor(bootstrap=True, max_features='auto', n_estimators=2000, max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_jobs=2)  # RandomForestRegressor(n_estimators=500, n_jobs=-1, max_features='auto')
        model3 = BaggingRegressor(base_estimator=bagging_base_tree, n_estimators=2000, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_jobs=None)
        # metamodel = BaggingRegressor(base_estimator=bagging_base_tree, n_estimators=1000, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_jobs=None)
        # regressors = [('rf', model2), ('et', model1)]
        # model = StackingRegressor(estimators=regressors, final_estimator=metamodel)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model1.fit(X=x_train, y=np.log(y_train))
        model2.fit(X=x_train, y=np.log(y_train))
        model3.fit(X=x_train, y=np.log(y_train))
        print('Stage 4: assigning weights on each prediction and combining them for final prediction...')
        pred = (np.exp(model1.predict(X=x_test)) * 0.6) + (np.exp(model2.predict(X=x_test)) * 0.28) + (np.exp(model3.predict(X=x_test)) * 0.12)

        print("\n\nfinal model: \n")
        self.evaluation(y_test=y_test, predictions=pred)

# Top Six Models
    @staticmethod
    def best_model_scores(X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        bagging_base_tree = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', min_samples_leaf=1, min_samples_split=2, splitter='best')
        adaboost_base_tree = bagging_base_tree
        models = [RandomForestRegressor(bootstrap=True, max_features='auto', n_estimators=2000, max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_jobs=2),
                  ExtraTreesRegressor(bootstrap=False, max_features='auto', n_estimators=3000, max_depth=None, min_samples_split=5),
                  BaggingRegressor(base_estimator=bagging_base_tree, n_estimators=2000, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_jobs=None),
                  AdaBoostRegressor(base_estimator=adaboost_base_tree, learning_rate=0.01, n_estimators=2000),
                  GradientBoostingRegressor(n_estimators=1000, min_samples_split=2, tol=0.00001, learning_rate=0.01)]
        model_names = ['RandomForest ', 'ExtraTrees ', 'Bagging(D.T.)', 'AdaBoost(D.T.)', 'GradientBoost ']
        rmsle = []
        r2 = []
        for model in range(len(models)):
            regressor = models[model]
            regressor.fit(X=x_train, y=np.log(y_train))
            predictions = np.exp(regressor.predict(X=x_test))
            for i, y in enumerate(predictions):
                if predictions[i] < 0:
                    predictions[i] = 0
            rmsle.append(np.sqrt(mean_squared_log_error(y_test, predictions)))
            r2.append(r2_score(y_test, predictions))
        d = pd.DataFrame({'Regression Model': model_names, 'RMSLE': rmsle, 'r2': r2}).sort_values('RMSLE').reset_index(drop=True)
        d.index += 1
        print(d)

# Other Tested Models (produced bad results)
    @staticmethod
    def rest_model_scores(X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        adaboost_base_tree = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', min_samples_leaf=1, min_samples_split=2, splitter='best')
        models = [
                    SVR(gamma=0.0001, C=1.0, epsilon=0.2, kernel='linear'),
                    LassoLars(normalize=False, copy_X=True, alpha=0.0009),
                    LinearRegression(normalize=True),
                    Lasso(alpha=0.001, copy_X=True, precompute=True, selection='random'),
                    ExtraTreeRegressor(ccp_alpha=0.008, splitter='best', random_state=96, max_features='auto', min_samples_leaf=6, min_samples_split=3),
                    KNeighborsRegressor(n_jobs=-1, n_neighbors=6)
                  ]
        model_names = ['SVR', 'LassoLeastAngle', 'Linear', 'Lasso', 'ExtraTree', 'KNeighbors']
        rmsle = []
        r2 = []
        for model in range(len(models)):
            regressor = models[model]
            regressor.fit(x_train, np.log(y_train))
            predictions = np.exp(regressor.predict(x_test))
            for i, y in enumerate(predictions):
                if predictions[i] < 0:
                    predictions[i] = 0
            rmsle.append(np.sqrt(mean_squared_log_error(y_test, predictions)))
            r2.append(r2_score(y_test, predictions))
        d = pd.DataFrame({'Regression Model': model_names, 'RMSLE': rmsle, 'r2': r2}).sort_values('RMSLE').reset_index(drop=True)
        d.index += 1
        print(d)

# FINAL PREDICTIONS
    def kaggle_submission(self):
        y, X = self.df_train['count'], self.df_train.drop(['count'], axis=1)

        # one hot encoding
        print('Stage 2: performing one_hot encoding...')
        X = self.one_hot_encoding('weather', self.one_hot_encoding('season', self.one_hot_encoding('year', self.one_hot_encoding('month',  X))))
        X_test = self.one_hot_encoding('weather', self.one_hot_encoding('season', self.one_hot_encoding('year', self.one_hot_encoding('month',  self.df_test))))  # self.one_hot_encoding('weather', self.one_hot_encoding('season', X)), self.one_hot_encoding('weather', self.one_hot_encoding('season', self.df_test))
        X.drop(['season', 'weather', 'year', 'month'], inplace=True, axis=1)
        X_test = self.fill_missing_column(X_test)
        X_test.drop(['season', 'weather', 'year', 'month'], inplace=True, axis=1)
        self.debug(X)
        self.debug(X_test)

        # model
        print('Stage 3: fitting best models...')
        bagging_base_tree = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', min_samples_leaf=1, min_samples_split=2, splitter='best')
        model1 = ExtraTreesRegressor(bootstrap=False, max_features='auto', n_estimators=3000, max_depth=None, min_samples_split=5)
        model2 = RandomForestRegressor(bootstrap=True, max_features='auto', n_estimators=2000, max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_jobs=2)
        model3 = BaggingRegressor(base_estimator=bagging_base_tree, n_estimators=2000, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_jobs=None)
        # regressors = [('rf', model2), ('et', model1)]
        # model = StackingRegressor(estimators=regressors)
        # best_estimator = self.use_GridSearch_CrossValidation(model, X, X_test, y)
        # predictions = best_estimator.predict(X=X_test)
        model1.fit(X=X, y=np.log(y))
        model2.fit(X=X, y=np.log(y))
        model3.fit(X=X, y=np.log(y))

        print('Stage 4: assigning weights on each prediction and combining them for final prediction...')
        predictions = (np.exp(model1.predict(X=X_test)) * 0.6) + (np.exp(model2.predict(X=X_test)) * 0.28) + (np.exp(model3.predict(X=X_test)) * 0.12)
        for i, y in enumerate(predictions):
            if predictions[i] < 0:
                predictions[i] = 0
        self.export_results(predictions)

    def use_GridSearch_CrossValidation(self, model, X, X_test, y):
        scorer = self.make_scorer_for_GridSearch()
        paramgrid = {'n_estimators': [100, 200, 300]}  # params
        grid_search = GridSearchCV(estimator=model, param_grid=paramgrid, scoring=scorer, cv=10, n_jobs=2)
        X, X_test = self.standard_scaling(X, X_test)
        grid_search.fit(X=X, y=y)
        print(grid_search.best_estimator_)
        return grid_search.best_estimator_

    @staticmethod
    def export_results(y_pred):
        submission = pd.DataFrame()
        submission["Id"] = range(y_pred.shape[0])
        submission['Predicted'] = y_pred
        submission.to_csv("KaggleSubmission\\submission.csv", index=False)
        print(y_pred.shape[0], ' predictions exported')

# RMSLE evaluation
    @staticmethod
    def evaluation(y_test, predictions):
        for i, y in enumerate(predictions):
            if predictions[i] < 0:
                predictions[i] = 0
        df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        print(df.tail(5))
        print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, predictions)))
        print('R2:', r2_score(y_test, predictions))


print("\n\n --Make sure to create a folder \"KaggleSubmission\" in the current working directory--\n\n\n")
print("run Local or Kaggle Evaluation?\n ")
in_put = input("1: Local Evaluation\n2: Kaggle Evaluation\n")
if in_put is '1':
    regression = BikeSharingDemand('datasets\\train.csv', 'datasets\\test.csv', True)
elif in_put is '2':
    regression = BikeSharingDemand('datasets\\train.csv', 'datasets\\test.csv', False)
else:
    print("Wrong input")
