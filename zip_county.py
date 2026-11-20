# usage: python zip_county.py apartments_segmented.csv tab20_zcta520_county20_natl.txt <output folder>
# builds zip_county.json: ZIP code -> 5-digit county FIPS code (the page then looks the FIPS code up in model.json "fips")
import json, sys
import pandas as pd

CSV, REL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# read county_fips as text and pad to 5 digits, so 1001 and 01001 are the same code and leading zeros never get lost
df = pd.read_csv(CSV, low_memory=False, usecols=['state', 'county', 'county_fips'], dtype={'county_fips': str})
df['county_fips'] = df.county_fips.str.replace(r'\.0$', '', regex=True).str.zfill(5)
fips = df.drop_duplicates('county_fips').set_index('county_fips')
assert fips.index.is_unique and df.groupby('county_fips').county.nunique().max() == 1
rel = pd.read_csv(REL, sep='|', dtype=str, encoding='utf-8-sig').dropna(subset=['GEOID_ZCTA5_20'])
rel['land'] = rel.AREALAND_PART.astype(int)
# each ZIP goes to the county holding most of its land; ZIPs whose main county isn't in the data are left out
main = rel.sort_values('land', ascending=False).drop_duplicates('GEOID_ZCTA5_20')
main = main[main.GEOID_COUNTY_20.isin(fips.index)]
z = {r.GEOID_ZCTA5_20: r.GEOID_COUNTY_20 for r in main.sort_values('GEOID_ZCTA5_20').itertuples()}
json.dump({'source': 'US Census Bureau 2020 ZCTA to county relationship file; each ZIP is assigned the 5-digit FIPS code of the county '
                     'with most of its land area; model.json "fips" gives the state and county name for each FIPS code',
           'zip': z}, open(f'{OUT}/zip_county.json', 'w'), separators=(',', ':'))
print(len(z), 'zips,', len(set(z.values())), 'counties, of', len(fips), 'in data')
