import re
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .preprocessing import seek_types

class CategoricalEncoder:
    """
    encodes and keeps encodings for multiple categorical features to their corresponding 0..N-1 integer values
    helpful for boosting models which deal with this type of categorical input with no need for ordering
    """
    def __init__(import re
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .preprocessing import seek_types


class CategoricalEncoder:
    """
    encodes and keeps encodings for multiple categorical features to their corresponding 0..N-1 integer values
    helpful for boosting models which deal with this type of categorical input with no need for ordering
    """

    def __init__(
        self,
        cat_na_fill_value="(MISSING_OR_UNKNOWN_VALUE)",
        map_unknown_to_na=True,
        recognize=False,
        verbose=False,
    ):
        self.cat_na_fill_value = cat_na_fill_value
        self.recognize = recognize
        self.verbose = verbose
        self.fitted = False
        self.category_to_index_transformers = None
        self.index_to_category_transformers = None

    def fit(self, dataframe, cat_features=None):
        self.fit_transform(dataframe, cat_features, transform_inplace=False)

    def fit_transform(self, dataframe, cat_features=None, transform_inplace=False):
        LE = LabelEncoder()
        if cat_features is None:
            if self.recognize:
                cat_features = seek_types(dataframe).get("cat", [])
                if self.verbose:
                    print(
                        f"The following features recognised as categorical: {cat_features}"
                    )
            else:
                cat_features = list(dataframe.columns)
        df_cat_features = [col for col in dataframe.columns if col in cat_features]
        if self.verbose:
            print(f"Categorical features in dataframe are: {df_cat_features}")
        if transform_inplace:
            df_cat = dataframe[df_cat_features]
        else:
            df_cat = deepcopy(dataframe[df_cat_features])
        ### when encoding nans are filled with a certain value
        df_cat.fillna(self.cat_na_fill_value, inplace=True)
        df_cat = df_cat.applymap(lambda x: str(x))

        category_to_index_transformers = []
        index_to_category_transformers = []

        if self.verbose:
            col_iterator = tqdm(df_cat_features)
        else:
            col_iterator = df_cat_features
        for col in col_iterator:
            mapped = LE.fit_transform(df_cat[col])
            if self.verbose:
                print("-" * 80)
                print(col)
                print(f"Initial categories: {LE.classes_}")
            df_cat[col] = mapped
            index_to_category = dict(enumerate(LE.classes_))
            category_to_index = {v: k for k, v in index_to_category.items()}
            index_to_category_transformers.append(index_to_category)
            category_to_index_transformers.append(category_to_index)
        self.category_to_index_transformers = dict(
            zip(df_cat_features, category_to_index_transformers)
        )
        self.index_to_category_transformers = dict(
            zip(df_cat_features, index_to_category_transformers)
        )
        self.fitted = True

        return df_cat

    def transform(
        self, dataframe, cat_features_for_transform=None, transform_inplace=False
    ):

        if not self.fitted:
            raise ValueError("encoder should be fitted first")
        if cat_features_for_transform is None:
            df_cat_features = [
                col
                for col in dataframe.columns
                if col in self.category_to_index_transformers.keys()
            ]
        else:
            df_cat_features = [
                col
                for col in dataframe.columns
                if col in self.category_to_index_transformers.keys()
                and col in cat_features_for_transform
            ]
        if self.verbose:
            print(f"The following features will be transformed: {df_cat_features}")
        if transform_inplace:
            df_cat = dataframe[df_cat_features]
        else:
            df_cat = deepcopy(dataframe[df_cat_features])
        ### when encoding nans are filled with a certain value
        df_cat.fillna(self.cat_na_fill_value, inplace=True)
        df_cat = df_cat.applymap(lambda x: str(x))

        if self.verbose:
            col_iterator = tqdm(df_cat_features)
        else:
            col_iterator = df_cat_features
        for col in col_iterator:
            if self.verbose:
                print("-" * 80)
                print(col)
            transformer = self.category_to_index_transformers[col]

            ### first filling nans
            df_cat[col] = df_cat[col].fillna(self.cat_na_fill_value)

            known_categories = set(transformer.keys())
            current_categories = set(df_cat[col].unique())
            unknown_categories = current_categories.difference(known_categories)
            unk_mask = df_cat[col].isin(unknown_categories)

            if len(unknown_categories) and self.verbose:
                print(
                    f"Previously unknown categories were met: {unknown_categories}.\
                    They will be replaced with value for NA"
                )
            ### if code for missing values already in dictionary then it's okay, if not - adding it to both transformers
            if self.cat_na_fill_value not in known_categories:
                new_index = max(transformer.values()) + 1
                transformer[self.cat_na_fill_value] = new_index

                self.category_to_index_transformers[col][
                    self.cat_na_fill_value
                ] = new_index
                self.index_to_category_transformers[col][
                    new_index
                ] = self.cat_na_fill_value
            ### assigning code for missing value for all unknown categories
            ### order of these two calls matters
            df_cat.loc[unk_mask, col] = self.cat_na_fill_value
            df_cat[col].replace(transformer, inplace=True)
        return df_cat


class HotEncoder:
    def __init__(
        self,
        force_nan_and_unknown_category=True,
        map_unknown_to_na=True,
        cat_na_fill_value="(MISSING_OR_UNKNOWN_VALUE)",
        drop="first",
        recognize=False,
        sparse=False,
        verbose=False,
    ):

        """
        'drop' is either "first" or "last" or None
        when 'force_nan_and_unknown_category' is True,
        even if nans are not present in training set corresponding column would be added
        it would not be dropped either way and would not cause multicollinearity in linear models
        or carry any info as it's constantly 0,
        but it allows safely handling new unknown values or nans on inference

        when 'map_unknown_to_na' is True, on transform new values goes to 'cat_na_fill_value' first
        and then strategy depends on whether 'force_nan_and_unknown_category' was set to True
        if not, it would raise an error

        """

        self.force_nan_and_unknown_category = force_nan_and_unknown_category
        self.map_unknown_to_na = map_unknown_to_na
        self.cat_na_fill_value = cat_na_fill_value
        assert drop is None or drop in ["first", "last"]
        self.drop = drop
        self.recognize = recognize
        self.sparse = sparse
        self.verbose = verbose
        self.transformers = None
        self.fitted = False

    def fit(self, dataframe, cat_features=None):

        if cat_features is None:
            if self.recognize:
                cat_features = seek_types(dataframe).get("cat", [])
                if self.verbose:
                    print(
                        f"The following features recognised as categorical: {cat_features}"
                    )
            else:
                cat_features = list(dataframe.columns)
        transformers = defaultdict(lambda: defaultdict(str))

        for cat_feature in cat_features:

            if (
                any(dataframe[cat_feature].isna())
                or self.force_nan_and_unknown_category
            ):
                fit_categories = pd.concat(
                    [
                        dataframe[cat_feature].fillna(self.cat_na_fill_value),
                        pd.Series(self.cat_na_fill_value),
                    ],
                    ignore_index=True,
                )
            else:
                fit_categories = dataframe[cat_feature].copy()
            ### если фича не строковая, но есть миссинги или поставили self.force_nan_and_unknown_category
            ### то будет смешанный тип, и OHE не отработает
            str_mapper = {v: str(v) for v in set(fit_categories)}
            ohe = OneHotEncoder(drop=None, handle_unknown="error", sparse=self.sparse)
            ohe.fit(fit_categories.map(str_mapper)[:, None])
            transformers[cat_feature]["transformer"] = ohe
            transformers[cat_feature]["categories"] = list(str_mapper.keys())
            transformers[cat_feature]["str_mapper"] = str_mapper
        self.transformers = {k: dict(v) for k, v in transformers.items()}
        self.fitted = True

    def transform(self, dataframe):

        X_out = []

        for cat_feature in self.transformers:
            transformer = self.transformers[cat_feature]["transformer"]
            str_mapper = self.transformers[cat_feature]["str_mapper"]
            categories_ordered = list(
                transformer.categories_[0]
            )  ### [0] is a first and single array in a list

            X = transformer.transform(
                dataframe[cat_feature]
                .fillna(self.cat_na_fill_value)
                .map(str_mapper)[:, None]
            )
            X = pd.DataFrame(X, columns=categories_ordered, index=dataframe.index)

            if self.drop is None:
                drop = []
            else:
                non_na_cols = [f for f in X.columns if f != self.cat_na_fill_value]
                if self.drop == "first":
                    drop = [non_na_cols[0]]
                else:
                    drop = [non_na_cols[-1]]
            X.drop(drop, axis=1, inplace=True)
            X.columns = [cat_feature + "__" + re.sub("\s+", "_", f) for f in X.columns]
            X_out.append(X)
        if X_out:
            return pd.concat(X_out, axis=1)
        else:
            return pd.DataFrame()

    def fit_transform(self, dataframe, cat_features=None):

        self.fit(dataframe, cat_features)
        return self.transform(dataframe)

        self,
        cat_na_fill_value = '(MISSING_OR_UNKNOWN_VALUE)',
        map_unknown_to_na = True,
        recognize = False,
        verbose = False,
    ):
        self.cat_na_fill_value = cat_na_fill_value
        self.recognize = recognize
        self.verbose = verbose
        self.fitted = False
        self.category_to_index_transformers = None
        self.index_to_category_transformers = None
        
    def fit(self, dataframe, cat_features = None):
        self.fit_transform(dataframe, cat_features, transform_inplace = False);
        
    def fit_transform(self, dataframe, cat_features = None, transform_inplace = False):
        LE = LabelEncoder()
        if cat_features is None:
            if self.recognize:
                cat_features = seek_types(dataframe).get('cat', [])
                if self.verbose:
                    print(f"The following features recognised as categorical: {cat_features}")
            else:
                cat_features = list(dataframe.columns)
        df_cat_features = [col for col in dataframe.columns if col in cat_features]
        if self.verbose:
            print(f"Categorical features in dataframe are: {df_cat_features}")
        
        if transform_inplace:
            df_cat = dataframe[df_cat_features]
        else:
            df_cat = deepcopy(dataframe[df_cat_features])
            
        ### when encoding nans are filled with a certain value
        df_cat.fillna(self.cat_na_fill_value, inplace = True)
        df_cat = df_cat.applymap(lambda x: str(x))

        category_to_index_transformers = []
        index_to_category_transformers = []
        
        if self.verbose:
            col_iterator = tqdm(df_cat_features)
        else:
            col_iterator = df_cat_features
            
        for col in col_iterator:
            mapped = LE.fit_transform(df_cat[col])
            if self.verbose:
                print("-"*80)
                print(col)
                print(f"Initial categories: {LE.classes_}")
            df_cat[col] = mapped
            index_to_category = dict(enumerate(LE.classes_))
            category_to_index = {v:k for k,v in index_to_category.items()}
            index_to_category_transformers.append(index_to_category)
            category_to_index_transformers.append(category_to_index)

        self.category_to_index_transformers = dict(zip(df_cat_features, category_to_index_transformers))
        self.index_to_category_transformers = dict(zip(df_cat_features, index_to_category_transformers))
        self.fitted = True
               
        return df_cat
    
    def transform(self, dataframe, cat_features_for_transform = None, transform_inplace = False):
        
        if not self.fitted:
            raise ValueError("encoder should be fitted first")
            
        if cat_features_for_transform is None:
            df_cat_features = [col for col in dataframe.columns if col in self.category_to_index_transformers.keys()]
        else:
            df_cat_features = [col for col in dataframe.columns if col in self.category_to_index_transformers.keys()\
                               and col in cat_features_for_transform]
        
        if self.verbose:
            print(f"The following features will be transformed: {df_cat_features}")
        
        if transform_inplace:
            df_cat = dataframe[df_cat_features]
        else:
            df_cat = deepcopy(dataframe[df_cat_features])
            
        ### when encoding nans are filled with a certain value
        df_cat.fillna(self.cat_na_fill_value, inplace = True)
        df_cat = df_cat.applymap(lambda x: str(x))
        
        if self.verbose:
            col_iterator = tqdm(df_cat_features)
        else:
            col_iterator = df_cat_features
            
        for col in col_iterator:
            if self.verbose:
                print("-"*80)
                print(col)
            
            transformer = self.category_to_index_transformers[col]
            
            ### first filling nans
            df_cat[col] = df_cat[col].fillna(self.cat_na_fill_value)
            
            known_categories = set(transformer.keys())
            current_categories = set(df_cat[col].unique())
            unknown_categories = current_categories.difference(known_categories)
            unk_mask = df_cat[col].isin(unknown_categories)
            
            if len(unknown_categories) and self.verbose:
                print(
                    f"Previously unknown categories were met: {unknown_categories}.\
                    They will be replaced with value for NA"
                )
                    
            ### if code for missing values already in dictionary then it's okay, if not - adding it to both transformers
            if self.cat_na_fill_value not in known_categories:
                new_index = max(transformer.values()) + 1
                transformer[self.cat_na_fill_value] = new_index
                
                self.category_to_index_transformers[col][self.cat_na_fill_value] = new_index
                self.index_to_category_transformers[col][new_index] = self.cat_na_fill_value
            
            ### assigning code for missing value for all unknown categories
            ### order of these two calls matters
            df_cat.loc[unk_mask, col] = self.cat_na_fill_value 
            df_cat[col].replace(transformer, inplace = True) 

        return df_cat


class HotEncoder:
    def __init__(
        self,
        force_nan_and_unknown_category = True,
        map_unknown_to_na = True, 
        cat_na_fill_value = '(MISSING_OR_UNKNOWN_VALUE)',
        drop = 'first',
        recognize = False,
        sparse = False,
        verbose = False,
    ):
        
        """
        'drop' is either "first" or "last" or None
        when 'force_nan_and_unknown_category' is True,
        even if nans are not present in training set corresponding column would be added
        it would not be dropped either way and would not cause multicollinearity in linear models
        or carry any info as it's constantly 0,
        but it allows safely handling new unknown values or nans on inference
        
        when 'map_unknown_to_na' is True, on transform new values goes to 'cat_na_fill_value' first
        and then strategy depends on whether 'force_nan_and_unknown_category' was set to True
        if not, it would raise an error
        
        """
        
        self.force_nan_and_unknown_category = force_nan_and_unknown_category
        self.map_unknown_to_na = map_unknown_to_na
        self.cat_na_fill_value = cat_na_fill_value
        assert drop is None or drop in ['first', 'last']
        self.drop = drop
        self.recognize = recognize
        self.sparse = sparse
        self.verbose = verbose
        self.transformers = None
        self.fitted = False
        
    def fit(self, dataframe, cat_features = None):
        
        if cat_features is None:
            if self.recognize:
                cat_features = seek_types(dataframe).get('cat', [])
                if self.verbose:
                    print(f"The following features recognised as categorical: {cat_features}")
            else:
                cat_features = list(dataframe.columns)
        
        transformers = defaultdict(lambda: defaultdict(str))

        for cat_feature in cat_features:

            if any(dataframe[cat_feature].isna()) or self.force_nan_and_unknown_category:
                fit_categories = (
                    pd.concat(
                        [
                            dataframe[cat_feature].fillna(self.cat_na_fill_value),
                            pd.Series(self.cat_na_fill_value)
                        ],
                        ignore_index = True
                    )
                )
            else:
                fit_categories = dataframe[cat_feature].copy()
                
            ### если фича не строковая, но есть миссинги или поставили self.force_nan_and_unknown_category
            ### то будет смешанный тип, и OHE не отработает
            str_mapper = {v:str(v) for v in set(fit_categories)}
            ohe = OneHotEncoder(drop = None, handle_unknown = 'error', sparse = self.sparse)
            ohe.fit(fit_categories.map(str_mapper)[:, None])
            transformers[cat_feature]['transformer'] = ohe
            transformers[cat_feature]['categories'] = list(str_mapper.keys())
            transformers[cat_feature]['str_mapper'] = str_mapper

        self.transformers = {k:dict(v) for k, v in transformers.items()}
        self.fitted = True
    
    def transform(self, dataframe):
        
        X_out = []
        
        for cat_feature in self.transformers:
            transformer = self.transformers[cat_feature]['transformer']
            str_mapper =  self.transformers[cat_feature]['str_mapper']
            categories_ordered = list(transformer.categories_[0]) ### [0] is a first and single array in a list

            X = transformer.transform(dataframe[cat_feature].fillna(self.cat_na_fill_value).map(str_mapper)[:, None])
            X = pd.DataFrame(X, columns = categories_ordered, index = dataframe.index)

            if self.drop is None:
                drop = []
            else:
                non_na_cols = [f for f in X.columns if f != self.cat_na_fill_value]
                if self.drop == 'first':
                    drop = [non_na_cols[0]]
                else:
                    drop = [non_na_cols[-1]]
            X.drop(drop, axis = 1, inplace = True)
            X.columns = [cat_feature + '__' + re.sub('\s+', '_', f) for f in X.columns]
            X_out.append(X)
            
        if X_out:
            return pd.concat(X_out, axis = 1)
        else:
            return pd.DataFrame()
            
    def fit_transform(self, dataframe, cat_features = None):
        
        self.fit(dataframe, cat_features)
        return self.transform(dataframe)