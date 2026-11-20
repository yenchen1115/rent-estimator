# usage: python build.py apartments_segmented.csv <output folder>
import base64, gzip, json, sys, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV, OUT = sys.argv[1], sys.argv[2]
SEGMENTS = ['High', 'Medium-High', 'Medium-Low', 'Low']
FILES = {'High': 'rf_high.txt', 'Medium-High': 'rf_medium_high.txt', 'Medium-Low': 'rf_medium_low.txt', 'Low': 'rf_low.txt'}
TUNED = {'High': 10, 'Medium-High': 10, 'Medium-Low': 8, 'Low': 10}
AMEN = ['AC', 'Alarm', 'Basketball', 'Cable or Satellite', 'Clubhouse', 'Dishwasher', 'Doorman', 'Elevator', 'Fireplace',
        'Garbage Disposal', 'Gated', 'Golf', 'Gym', 'Hot Tub', 'Internet Access', 'Luxury', 'Parking', 'Patio/Deck',
        'Playground', 'Pool', 'Refrigerator', 'Storage', 'TV', 'Tennis', 'View', 'Washer Dryer', 'Wood Floors']

df = pd.read_csv(CSV, low_memory=False, dtype={'county_fips': str})
df['county_fips'] = df.county_fips.str.replace(r'\.0$', '', regex=True).str.zfill(5)  # keep leading zeros: 01001, not 1001
# the 5-digit county FIPS code is the identifier: one code per state and county name, and one name per code
assert df.groupby('county_fips')[['state', 'county']].nunique().max().max() == 1
assert df.groupby(['state', 'county']).county_fips.nunique().max() == 1
df['log_square_feet'] = np.log(df['square_feet'])
df['log_county_population_density'] = np.log(df['county_population_density_2019'])
assert np.allclose(df['log_price'], np.log(df['price']))

x_labels = ['log_square_feet', 'county_median_income_2019', 'log_county_population_density',
            'bathrooms', 'bedrooms', 'state'] + AMEN
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
to_dummies = ColumnTransformer(transformers=[('cats', ohe, ['state'])], remainder='passthrough')
base = Pipeline([('dummy', to_dummies), ('randomforestregressor', RandomForestRegressor(random_state=0))])
cv = KFold(n_splits=5, shuffle=True, random_state=0)

segments = {}
for seg in SEGMENTS:
    t0 = time.time()
    data = df[df['market_segment'] == seg]
    x, y = data[x_labels], data['log_price']
    pipe = clone(base).set_params(randomforestregressor__min_samples_leaf=1,
                                  randomforestregressor__max_features=TUNED[seg])
    yhat = cross_val_predict(pipe, x, y, cv=cv, n_jobs=-1)
    folds = []
    for _, te in cv.split(x):
        yt, yp = y.values[te], yhat[te]
        folds.append(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    price, pred = data['price'].values, np.exp(yhat)
    mdape = np.median(np.abs(price - pred) / price)
    lo, hi = np.quantile(price / pred, [0.1, 0.9])

    model = clone(pipe).fit(x, y)
    names = model.named_steps['dummy'].get_feature_names_out().tolist()
    assert len(names) < 255
    feats, vals, rights, roots, off = [], [], [], [], 0
    for est in model.named_steps['randomforestregressor'].estimators_:
        t = est.tree_
        n = t.node_count
        internal = t.children_left != -1
        assert (t.children_left[internal] == np.arange(n)[internal] + 1).all()
        roots.append(off)
        feats.append(np.where(internal, t.feature, 255).astype(np.uint8))
        vals.append(np.where(internal, t.threshold, t.value[:, 0, 0]).astype(np.float32))
        rights.append(np.where(internal, t.children_right - np.arange(n), 0).astype(np.uint32))
        off += n
    feat, val, right = np.concatenate(feats), np.concatenate(vals), np.concatenate(rights)
    nn = len(feat)
    pad = (-nn) % 4
    blob = feat.tobytes() + b'\0' * pad + val.tobytes() + right.tobytes()
    layout = {'feat': 0, 'val': nn + pad, 'right': nn + pad + 4 * nn}

    # check the exported arrays give the same predictions as scikit-learn
    X = model.named_steps['dummy'].transform(x.iloc[:300])
    js = []
    for row in X:
        s = 0.0
        for r in roots:
            k = r
            while feat[k] != 255:
                k = k + 1 if row[feat[k]] <= val[k] else k + int(right[k])
            s += float(val[k])
        js.append(s / len(roots))
    diff = np.abs(np.exp(js) - np.exp(model.predict(x.iloc[:300]))).max()

    with open(f'{OUT}/{FILES[seg]}', 'w') as fh:
        fh.write(base64.b64encode(gzip.compress(blob, 9, mtime=0)).decode())
    segments[seg] = {
        'file': FILES[seg], 'bytes': len(blob), 'n_nodes': nn, 'roots': roots, 'features': names, 'layout': layout,
        'params': {'min_samples_leaf': 1, 'max_features': TUNED[seg]}, 'n': len(data),
        'states': sorted(data.state.unique()), 'cv_r2': round(float(np.mean(folds)), 4),
        'median_abs_pct_error': round(float(mdape), 3),
        'interval': {'level': 0.8, 'low_mult': round(float(lo), 4), 'high_mult': round(float(hi), 4)},
    }
    print(f'{seg}: n={len(data)} CV R2(log)={np.mean(folds):.4f} folds={np.round(folds, 4).tolist()} '
          f'MdAPE={mdape:.4f} band={lo:.4f}-{hi:.4f} nodes={nn:,} maxdiff=${diff:.4f} {time.time() - t0:.0f}s', flush=True)

seg_of = df.groupby('state')['market_segment'].first()
state_stats = {s: {'n': int(len(g)), 'median_pps': round(float((g.price / g.square_feet).median()), 3), 'segment': seg_of[s]}
               for s, g in df.groupby('state')}
cities_by_state = {s: sorted(g.cityname.unique().tolist()) for s, g in df.groupby('state')}

counties, city_county = {}, {}
for s, g in df.groupby('state'):
    c = g.groupby('county').agg(n=('price', 'size'), income=('county_median_income_2019', 'first'),
                                logd=('log_county_population_density', 'first'))
    counties[s] = {k: [float(r.income), float(r.logd), int(r.n)] for k, r in c.sort_index().iterrows()}
    cc = g.groupby(['cityname', 'county']).size().reset_index(name='n').sort_values(['n', 'county'], ascending=[False, True])
    city_county[s] = cc.drop_duplicates('cityname').set_index('cityname')['county'].sort_index().to_dict()

model = {
    'type': 'segmented_random_forest_gzip',
    'name': 'Random forest per market segment, RandomForestRegressor(random_state=0) tuned (min_samples_leaf, max_features), on log price',
    'data': 'apartments_for_rent_classified_100K with 2019 county median income and population density, listings Dec 2018 to Dec 2019',
    'target': 'log_price',
    'segment_method': 'four market segments assigned by state in the team dataset',
    'segments': segments, 'amenities': AMEN, 'cities_by_state': cities_by_state, 'state_stats': state_stats,
    'counties': counties, 'city_county': city_county,
    # county FIPS code -> [state, county name]; zip_county.json maps each ZIP code to one of these codes
    'fips': {k: [r.state, r.county] for k, r in df.drop_duplicates('county_fips').set_index('county_fips').sort_index().iterrows()},
}
json.dump(model, open(f'{OUT}/model.json', 'w'), separators=(',', ':'))

QS = np.linspace(0.05, 0.95, 19)
def bench(g):
    return {'median': float(g.price.median()), 'n': int(len(g)), 'q': [round(float(v), 1) for v in np.quantile(g.price, QS)],
            'sqft': float(g.square_feet.median()), 'baths': float(g.bathrooms.median()),
            'amen': {a: round(float(g[a].mean()), 3) for a in AMEN}}
df['beds'] = df.bedrooms.astype(int)
B = {'min_n': 20,
     'by_state': {f'{s}|{b}': bench(g) for (s, b), g in df.groupby(['state', 'beds'])},
     'by_segment': {f'{s}|{b}': bench(g) for (s, b), g in df.groupby(['market_segment', 'beds'])},
     'by_city': {f'{s}|{c}|{b}': {'median': float(g.price.median()), 'n': int(len(g)),
                                  'q': [round(float(v), 1) for v in np.quantile(g.price, QS)]}
                 for (s, c, b), g in df.groupby(['state', 'cityname', 'beds']) if len(g) >= 20}}
json.dump(B, open(f'{OUT}/benchmarks.json', 'w'), separators=(',', ':'))

W = np.array([1 << i for i in range(len(AMEN))])
LS = {'amenities': AMEN, 'states': {}}
for s, g in df.groupby('state'):
    cities, county_list = cities_by_state[s], list(counties[s])
    ci = {c: i for i, c in enumerate(cities)}
    ki = {c: i for i, c in enumerate(county_list)}
    mask = (g[AMEN].values.astype(int) * W).sum(axis=1)
    rows = [[int(b), float(ba), int(round(sq)), float(p), ci[c], int(m), ki[k]]
            for b, ba, sq, p, c, m, k in zip(g.beds, g.bathrooms, g.square_feet, g.price, g.cityname, mask, g.county)]
    LS['states'][s] = {'cities': cities, 'counties': county_list, 'rows': rows}
json.dump(LS, open(f'{OUT}/listings.json', 'w'), separators=(',', ':'))
print('done')
