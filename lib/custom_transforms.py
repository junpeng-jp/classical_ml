import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator

class DtypeMapper(BaseEstimator, TransformerMixin):
    def __init__(self, dtype_map, copy=False):
        self.dtype_map = dtype_map if dtype_map else {}
        self.copy = copy

    def fit(self, X, y=None):
        self.col_dtype_ = X.dtypes.astype(str)

        for dtype, cols in self.dtype_map.items():
            for col in cols:
                if self.col_dtype_[col] != dtype:
                    self.col_dtype_[col] = dtype

        return self
    
    def transform(self, X):
        if self.copy:
            X = X.copy()
        for col, dtype in self.col_dtype_.iteritems():
            if X[col].dtype.name != dtype:
                X[col] = X[col].astype(dtype)

        return X


class GroupMinority(BaseEstimator, TransformerMixin):
    def __init__(self, cols, threshold, copy=False):
        self.cols = cols
        self.threshold = threshold
        self.copy = copy
    
    def fit(self, X, y=None):
        self.remap_ = {}
        
        for col in self.cols:
            temp = X.groupby(col).agg({col: 'count'})
            largest = max(temp.index) + 1
            temp = temp.apply(lambda x: np.where(x > len(X) * self.threshold, x.index, largest)).to_dict()
            self.remap_.update(temp)

        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()
        
        for col, remap in self.remap_.items():
            X[col] = X[col].map(remap)

        return X

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols, copy=False):
        self.drop_cols = drop_cols
        self.copy = copy

    def fit(self, X, y=None):
        self.remaining_col_ = []

        for col in X.columns:
            if col not in self.drop_cols:
                self.remaining_col_.append(col)

        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()
        
        X.drop(self.drop_cols, axis=1, inplace=True)

        return X

class TransformByDtype(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, include_dtypes, combine_strategy = 'append', copy=False):
        self.transformer = transformer
        self.include_dtypes = include_dtypes
        self.copy = copy
        self.combine_strategy = combine_strategy

    def fit(self, X, y=None):
        if self.combine_strategy not in ['append', 'delete_old', 'reassign']:
            raise AttributeError('Attribute combine_strategy can only be "append", "delete_old" or "reassign" strategy')
        self.col_subset_ = X.select_dtypes(self.include_dtypes).columns.to_list()
        self.transformer.fit(X[self.col_subset_], y)
        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()

        transformed = self.transformer.transform(X[self.col_subset_])
        if self.combine_strategy == 'reassign':
            X.loc[:, self.col_subset_] = transformed

        else:
            transformed = pd.DataFrame(
                data = transformed, 
                columns = self.transformer.get_feature_names(self.col_subset_), 
                index = X.index)

            X = pd.concat([X, transformed], axis=1)

            if self.combine_strategy == 'delete_old':
                X.drop(self.col_subset_, axis = 1, inplace = True)

        return X

class PdDummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, prefix = None, prefix_sep = '_', dummy_na = False, columns = None,
        sparse = False, drop_first = False, dtype = np.uint8, copy = False):

        self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.dummy_na = dummy_na
        self.columns = columns
        self.sparse = sparse
        self.drop_first = drop_first
        self.dtype = dtype
        self.copy = copy

    def fit(self, X, y = None):
        self.col_category_ = {}
        self.object_col_ = []

        if self.columns:
            self.select_cols_ = self.columns
        else:
            self.select_cols_ = X.select_dtypes('category').columns.to_list()
            self.object_col_ += X.select_dtypes('object').columns.to_list()

        for col in self.select_cols_ + self.object_col_:
            if X[col].dtype.name == 'object':
                self.col_category_[col] = np.sort(X[col].unique()).tolist()
            else:
                self.col_category_[col] = X[col].cat.categories.tolist()

        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()

        for col in self.object_col_:
            X[col] = X[col].astype('category')

        return pd.get_dummies(
            data = X,
            prefix = self.prefix,
            prefix_sep = self.prefix_sep,
            dummy_na = self.dummy_na,
            columns = self.select_cols_ + self.object_col_,
            sparse = self.sparse,
            drop_first = self.drop_first,
            dtype = self.dtype
        )