# usage: python zip_county.py apartments_segmented.csv tab20_zcta520_county20_natl.txt <output folder>
import json, sys
import pandas as pd

CSV, REL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(CSV, low_memory=False, usecols=['state', 'county', 'county_fips'])
fips = df.drop_duplicates('county_fips').set_index('county_fips')
assert fips.index.is_unique and df.groupby('county_fips').county.nunique().max() == 1
rel = pd.read_csv(REL, sep='|', dtype=str, encoding='utf-8-sig').dropna(subset=['GEOID_ZCTA5_20'])
rel['land'] = rel.AREALAND_PART.astype(int)
# each ZIP goes to the county holding most of its land; ZIPs whose main county isn't in the data are left out
main = rel.sort_values('land', ascending=False).drop_duplicates('GEOID_ZCTA5_20')
main = main[main.GEOID_COUNTY_20.astype(int).isin(fips.index)]
labels = sorted({f"{fips.at[int(g), 'state']}|{fips.at[int(g), 'county']}" for g in main.GEOID_COUNTY_20})
idx = {k: i for i, k in enumerate(labels)}
z = {r.GEOID_ZCTA5_20: idx[f"{fips.at[int(r.GEOID_COUNTY_20), 'state']}|{fips.at[int(r.GEOID_COUNTY_20), 'county']}"]
     for r in main.sort_values('GEOID_ZCTA5_20').itertuples()}
json.dump({'source': 'US Census Bureau 2020 ZCTA to county relationship file; each ZIP is assigned the county with most of its land area',
           'counties': labels, 'zip': z}, open(f'{OUT}/zip_county.json', 'w'), separators=(',', ':'))
print(len(z), 'zips,', len(labels), 'counties, of', len(fips), 'in data')
