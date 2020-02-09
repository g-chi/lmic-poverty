import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from scipy.stats import pearsonr


def r2_scoring_func(y_true, y_pred):
    """ calculate pearson r-squared
    @param y_true np.array: true values
    @param y_true np.array: predicted values
    @return r2 float: pearson r-squared
    """
    r2 = pearsonr(y_true, y_pred)[0] ** 2
    return r2


def mean_squared_error_scoring_func(y_true, y_pred):
    """
    @param y_true np.array: true values
    @param y_true np.array: predicted values
    @return mse float: mean squared error
    """
    mse = np.average((y_true - y_pred) ** 2, axis=0)
    return mse


# create scoring functions for choosing optimal hyperparameters
pearson_r2_score = make_scorer(r2_scoring_func, greater_is_better=True)
mean_squared_error_score = make_scorer(mean_squared_error_scoring_func,
                                       greater_is_better=False)


def get_optimal_hyperparameters(training_data, feature_names=all_features, cv=10):
    """ optimize model by gridsearch
    @param training_data pd.DataFrame:
    @param feature_names list: features used to train a model
    @param cv int or other cross validation results:
    @return scores dict:
    @return best_params dict:
    """
    xgb_model = xgb.XGBRegressor(n_jobs=-1)
    xgb_params = {
        'max_depth':  [1, 3, 5, 10, 15, 20, 30],
        'min_child_weight': [1, 3, 5, 7, 10],
    }
    regressor = GridSearchCV(xgb_model, 
                             xgb_params, 
                             scoring={'r2': pearson_r2_score,
                                      'mse': mean_squared_error_score}, 
                             cv=cv,
                             refit='mse')
    regressor.fit(np.array(training_data[feature_names]),
                  np.array(training_data['rwi']))
    best_params = regressor.best_params_
    results = regressor.cv_results_
    best_index = np.nonzero(results['rank_test_mse'] == 1)[0][0]
    scores = {}
    for scorer in ['r2', 'mse']:
        best_score = results['mean_test_%s' % scorer][best_index]
        scores[scorer] = np.absolute(best_score)
    return scores, best_params, regressor


def get_accuracy_basic_cv(training_data_dict):
    """ calculate the accuracy of the model using the basic cross-validation
    @param training_data_dict dict: training data
    @return r2 list: r2 of all the 50 countries
    @return mse list: mse of all the 50 countries
    """
    r2 = []
    mse = []
    for country in countries_dhs:
        training_data = training_data_dict[country]
        scores, best_params, _ = get_optimal_hyperparameters(training_data)
        r2.append(scores['r2'])
        mse.append(scores['mse'])
    return r2, mse


def get_accuracy_spatial_cv(training_data_dict):
    """ calculate the accuracy of the model using the spatial cross-validation
    @param training_data_dict dict: training data
    @return r2 list: r2 of all the 50 countries
    @return mse list: mse of all the 50 countries
    """
    r2 = []
    mse = []
    for country in countries_dhs:
        training_data = training_data_dict[country]
        points = training_data[['cluster', 'longitude', 'latitude']]
        points = points.reset_index()
        scores, best_params, _ = get_optimal_hyperparameters(
            training_data, 
            cv=spatial_cv(points, k=10)
        )
        r2.append(scores['r2'])
        mse.append(scores['mse'])
    return r2, mse


def get_accuracy_leaveoneout(training_data_dict):
    """ calculate the accuracy of the model by leave one country out
    @param training_data_dict dict: training data
    @return r2 list: r2 of all the 50 countries
    @return mse list: mse of all the 50 countries
    """
    r2 = []
    mse = []
    for left_country in countries_dhs:
        training_data = training_data_dict[left_country]
        dfs = [training_data_dict[country] for country in countries_dhs if country != left_country]
        merged_df = pd.concat(dfs, axis=0)
        training_data = shuffle(merged_df)
        validation_data = training_data_dict[left_country]
        scores, best_params, _ = get_optimal_hyperparameters(training_data)
        y_pred = regressor.predict(np.array(validation_data[feature_names]))
        y = validation_data['rwi']
        r2 = r2_scoring_func(y, y_pred)
        mse = mean_squared_error_scoring_func(y, y_pred)
        r2.append(scores['r2'])
        mse.append(scores['mse'])
    return r2, mse


def train_final_model(training_data):
    """ calculate the accuracy of the model by leave one country out
    @param training_data_dict dict: training data
    @return regressor list: r2 of all the 50 countries
    """
    xgb_model = xgb.XGBRegressor(n_jobs=-1, random_state=123)
    xgb_params = {
        'max_depth':  [1, 3, 5, 10, 15, 20, 30],
        'min_child_weight': [1, 3, 5, 7, 10],
    }
    training_data_final = shuffle(training_data)
    logo = LeaveOneGroupOut()
    cv = logo.split(np.array(training_data_final[all_features]),
                    groups=training_data_final.group)
    regressor = GridSearchCV(xgb_model,
                             xgb_params,
                             scoring={'r2': pearson_r2_score ,
                                      'mse': mean_squared_error_score},
                             cv=cv,
                             refit='mse')
    regressor.fit(np.array(training_data_final[feature_names]),
                  np.array(training_data_final['rwi']))
    return regressor


def spatial_cv(points, k=10):
    """ spatial cross validation to avoid the spatial autocorrelation issue
    @param points: df
    @param k: int
    """
    left = points.longitude.min()
    right = points.longitude.max() + 0.01
    top = points.latitude.max()
    bottom = points.latitude.min() - 0.01
    width = right - left
    height = top - bottom
    size_x = width / 10
    size_y = -height / 10
    centroid_x = []
    centroid_y = []
    for i in range(10):
        centroid_x.append(left + size_x / 2 + size_x * i)
        centroid_y.append(top + size_y / 2 + size_y * i)
    cell = []
    for y in range(10):
        for x in range(10):
            cell.append([x, y])
    id_chosen = np.random.choice(range(len(cell)), k, replace=False)
    cell_chosen = [cell[i] for i in id_chosen]
    train_n = int(len(points) / k) * (k - 1)
    test_n = len(points) - train_n
    train_test_split = []
    for c in cell_chosen:
        x, y = centroid_x[c[0]], centroid_y[c[1]]
        points_c= points.copy()
        points_c['x_diff'] = points_c['longitude'] - x
        points_c['y_diff'] = points_c['latitude'] - y
        points_c['dis'] = (points_c['x_diff'] ** 2 + points_c['y_diff'] ** 2) ** .5
        points_c = points_c.sort_values(by='dis')
        id_order = points_c.index.tolist()
        test_id = id_order[:test_n]
        train_id = id_order[test_n:]
        train_test_split.append((train_id, test_id))
        yield (train_id, test_id)


# 50 countries with the DHS
countries_dhs = [
    'ZW', 'ZM', 'UG', 'TZ', 'TL', 'TJ', 'TG', 'TD', 'SZ', 'SN',
    'SL', 'RW', 'PH', 'PE', 'NP', 'NG', 'MZ', 'MW', 'MM', 'ML',
    'MG', 'MD', 'MA', 'LS', 'LR', 'KH', 'KG', 'KE', 'ID', 'HT',
    'HN', 'GT', 'GN', 'GH', 'GA', 'ET', 'EG', 'DO', 'CM', 'CD',
    'BO', 'BJ', 'BI', 'BF', 'BD', 'AO', 'AM', 'AL', 'NA', 'CI'
]

# countries to estimate RWI
countries_to_estimate = [
    'ZW', 'ZM', 'ZA', 'YT', 'YE', 'WS', 'WF', 'VU', 'VN', 'VI',
    'VG', 'VE', 'VC', 'VA', 'UZ', 'UY', 'US', 'UA', 'TZ', 'UG',
    'TT', 'TW', 'TV', 'TR', 'TM', 'TN', 'TK', 'TO', 'TJ', 'TL',
    'SV', 'SZ', 'SY', 'SX', 'TH', 'TC', 'TG', 'TD', 'SR', 'SS',
    'SN', 'SO', 'ST', 'SK', 'SL', 'SJ', 'SI', 'SM', 'SC', 'SE',
    'SH', 'SD', 'SG', 'SB', 'RW', 'RU', 'SA', 'RS', 'RO', 'RE',
    'QA', 'PY', 'PW', 'PT', 'PR', 'PS', 'PM', 'PN', 'PK', 'PL',
    'PH', 'PE', 'PG', 'PF', 'PA', 'OM', 'NZ', 'NU', 'NR', 'NP',
    'NO', 'NL', 'NI', 'NG', 'NF', 'NE', 'NC', 'NA', 'MZ', 'MY',
    'MX', 'MW', 'MV', 'MT', 'MU', 'MS', 'MR', 'MP', 'MQ', 'MO',
    'MN', 'MM', 'ML', 'MK', 'MH', 'MG', 'MF', 'ME', 'MD', 'MC',
    'MA', 'LY', 'LV', 'LU', 'LT', 'LS', 'LR', 'LK', 'LI', 'LC',
    'LB', 'LA', 'KZ', 'KY', 'KW', 'KR', 'KP', 'KN', 'KM', 'KI',
    'KH', 'KG', 'KE', 'JP', 'JO', 'JM', 'JE', 'IT', 'IS', 'IR',
    'IQ', 'IN', 'IM', 'IL', 'IE', 'ID', 'HU', 'HT', 'HR', 'HK',
    'HN', 'GY', 'GW', 'GU', 'GT', 'GR', 'GQ', 'GP', 'GN', 'GL',
    'GM', 'GI', 'GH', 'GG', 'GF', 'GE', 'GD', 'GB', 'GA', 'FR',
    'FM', 'FO', 'FK', 'FJ', 'FI', 'ET', 'ES', 'ER', 'EH', 'EG',
    'EE', 'EC', 'DZ', 'DO', 'DM', 'DK', 'DJ', 'DE', 'CZ', 'CY',
    'CW', 'CV', 'CU', 'CR', 'CO', 'CN', 'CM', 'CL', 'CK', 'CI',
    'CH', 'CG', 'CF', 'CD', 'CA', 'BZ', 'BY', 'BW', 'BT', 'BS',
    'BR', 'BQ', 'BO', 'BN', 'BM', 'BL', 'BJ', 'BI', 'BH', 'BG',
    'BF', 'BE', 'BD', 'BB', 'BA', 'AZ', 'AX', 'AW', 'AU', 'AT',
    'AS', 'AR', 'AO', 'AM', 'AL', 'AI', 'AG', 'AF', 'AE', 'AD',
]

print(len(countries_dhs))
print(len(countries_to_estimate))

all_features = [
    'road_density',
    'if_urban_builtup',
    'elevation',
    'slope',
    'precipitation',
    'population',
    'cell_tower_count',
    'wifi_count',
    'num_mobile_devices',
    'num_android_devices',
    'num_ios_devices',
    'radiance'
]

for i in range(1, 101):
    all_features.append('dg_pca_feature_' + str(i))

# load training data
training_data_dict = pickle.load(open("training_data_dict.pickle", "rb" ))
country_feature_dict = pickle.load(open("country_feature_dict.pickle", "rb" ))

# calculate the accuracy (R^2) using different cross validation methods
r2_basic_cv, mse_basic_cv = get_accuracy_basic_cv(training_data_dict)
r2_spatial_cv, mse_spatial_cv = get_accuracy_spatial_cv(training_data_dict)
r2_leaveoneout, mse_leaveoneout = get_accuracy_leaveoneout(training_data_dict)

# prepare the training data
# and train a final model using the grouth truth in 50 countries
training_data_final = []
for country in countries_dhs:
    this = training_data_dict[country]
    this['group'] = i
    if i == 0:
        training_data_final = this
    else:
        training_data_final = training_data_final.append(this)
regressor = train_final_model(training_data_final)

# make estimation in each country
for country in countries_to_estimate:
    df = country_feature_dict[country]
    y = regressor.predict(np.array(df[feature_names]))
    df['rwi_pred'] = y
    df['bing_tile_x'] = df.apply(lambda x: x['bing_tile'][0], axis=1)
    df['bing_tile_y'] = df.apply(lambda x: x['bing_tile'][1], axis=1)
    df[['bing_tile_x', 'bing_tile_y', 'rwi_pred']].to_csv(country + '_rwi.csv', index=None)

# feature importance
# method 1: R2 from a univariate regression
feature_importance_r2 = []
feature_importance_mse = []
for country in countries_dhs:
    training_data = training_data_dict[country]
    for feature in all_features:
        scores, best_params, _ = get_optimal_hyperparameters(training_data, [feature])
        feature_importance_r2.append(scores['r2'])
        feature_importance_mse.append(scores['mse'])

# method 2: gain
feature_importance_gain = []
for country in countries_dhs:
    training_data = training_data_dict[country]
    scores, best_params, regressor = get_optimal_hyperparameters(training_data)
    score_result = regressor.get_booster().get_score(importance_type='gain')
    feature_importance_gain.append(score_result)
