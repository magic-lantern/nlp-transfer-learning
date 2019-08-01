---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Based on our custom MIMIC language model, train a classifier

Make sure mimic_nlp_lm has been run first and sucessfully completed. That notebook builds the language model that allows classificiation to occur efficiently.

## Create a classifier to predict Length of Stay (LOS)

Steps:
1. Load clinical Notes
1. Load Admissions data
1. Calculate Length of Stay
1. Join Admissions data with Notes data (on HADM_ID) - Columns needed for classifier: LOS, TEXT

Would also be nice to see a graphical summary of LOS.

```python
from fastai.text import *
from sklearn.model_selection import train_test_split
import glob
import gc
import altair as alt
```

Setup filenames and paths

```python
# pandas doesn't understand ~, so provide full path
base_path = Path.home() / 'mimic'

# files used during processing - all aggregated here
admissions_file = base_path/'ADMISSIONS.csv'
notes_file = base_path/'NOTEEVENTS.csv'

class_file = 'los_cl_data.pickle'
notes_pickle_file = base_path/'noteevents.pickle'
lm_file = 'mimic_lm.pickle' # actual file is at base_path/lm_file but due to fastai function, have to pass file name separately
init_model_file = base_path/'los_cl_head'
cycles_file = base_path/'cl_num_iterations.pickle'
enc_file = 'mimic_fine_tuned_enc'
ft_file = 'los_cl_fine_tuned_'
```

Setup parameters for models

```python
# original data set too large to work with in reasonable time due to limted GPU resources
pct_data_sample = 0.1
# how much to hold out for validation
valid_pct = 0.2
# for repeatability - different seed than used with language model
seed = 1776
# for language model building - not sure how this will translate to classifier
# batch size of 128 GPU uses 14GB RAM
# batch size of 96 GPU uses 9GB RAM
# batch size of 48 GPU uses 5GB RAM
bs=96
```

```python
orig_df = pd.DataFrame()
if os.path.isfile(notes_pickle_file):
    print('Loading noteevent pickle file')
    orig_df = pd.read_pickle(notes_pickle_file)
    print(orig_df.shape)
else:
    print('Could not find noteevent pickle file; creating it')
    # run this the first time to covert CSV to Pickle file
    orig_df = pd.read_csv(notes_file, low_memory=False, memory_map=True)
    orig_df.to_pickle(notes_pickle_file)
```

Since seed is different, this should be quite different than the language model dataset.

Should I show details on how many records are in language model dataset?

```python
notes_df = orig_df[orig_df.HADM_ID.notnull()].copy()
notes_df.HADM_ID = notes_df.HADM_ID.astype(int)
notes_df.shape
```

```python
notes_df.head()
```

```python
print('Loading ADMISSIONS.csv')
a_orig = pd.read_csv(admissions_file, low_memory=False, memory_map=True)
a_orig.shape
```

```python
a_df = a_orig[['HADM_ID', 'ADMITTIME', 'DISCHTIME']].copy()
#pd.to_datetime('2014-04-09T152959.999993-0500', utc=True)
# passing format just to make sure conversion doesn't mess something up
a_df['admittime'] = pd.to_datetime(a_df.ADMITTIME, format='%Y-%m-%d %H:%M:%S')
a_df['dischtime'] = pd.to_datetime(a_df.DISCHTIME, format='%Y-%m-%d %H:%M:%S')
a_df['los'] = (a_df['dischtime'] - a_df['admittime']).astype('timedelta64[D]')
# there are 98 admissions where length of stay is negative. change to 0
a_df.loc[a_df.los < 0, 'los'] = 0
a_df.head()
```

### Histogram of Length of Stay

```python
alt.Chart(a_df.head(4000)).mark_bar().encode(
    alt.X('los',
         bin=alt.BinParams(maxbins=50)),
    y='count()',
)
```

```python
# make sure we only keep rows with notes
combined_df = pd.merge(a_df, notes_df, on='HADM_ID', how='right')

# passing format just to make sure conversion doesn't mess something up
combined_df['charttime'] = pd.to_datetime(combined_df.CHARTTIME, format='%Y-%m-%d %H:%M:%S')
combined_df = combined_df['HADM_ID', 'admittime', 'dischtime', 'los', 'charttime', 'TEXT']
combined_df.rename(columns={"HADM_ID": "hadm_id", "TEXT": "text"})
```

```python
combined_df.shape
```

```python
# these should all be zero
print(combined_df[combined_df.los.isnull()].shape)
print(combined_df[combined_df.HADM_ID.isnull()].shape)
print(combined_df[combined_df.TEXT.isnull()].shape)
```

```python
combined_df.head()
```

### As an alternative - how about using notes from day 1 of stay to predict LOS?


    For each admission
        do they have notes on day 1 of stay

```python
combined_df.dtypes
```

```python
combined_df[combined_df.HADM_ID == 100006]
```

```python
h = 100006
#for h in combined_df.HADM_ID.unique():
combined_df[(combined_df.HADM_ID == h) & 
            (combined_df.charttime >= combined_df.admittime) &
            (combined_df.charttime < (combined_df.admittime + pd.Timedelta(hours=24)))]
```

### Histogram of number of notes by Hospital Admission - 10% random sample

```python
alt.Chart(
    combined_df.groupby('HADM_ID', as_index=False).TEXT.count().sample(frac=.1, random_state=seed)
).mark_bar().encode(
    alt.X('TEXT', bin=alt.BinParams(maxbins=50)),
    y='count()',
)
```

### Scatter plot of Number of Notes vs Length of Stay

```python
combined_df[['HADM_ID', 'los']].drop_duplicates().shape            #42,195
combined_df.groupby('HADM_ID', as_index=False).TEXT.count().shape  #42,195
```

```python
los_v_num_notes = pd.merge(combined_df[['HADM_ID', 'los']].drop_duplicates(), 
          combined_df.groupby('HADM_ID', as_index=False).TEXT.count(),
          on='HADM_ID')
los_v_num_notes.shape
```

```python
alt.Chart(los_v_num_notes.sample(frac=.1, random_state=seed)).mark_point().encode(
    x=alt.X('los', axis=alt.Axis(title='Length of Stay (Days)')),
    y=alt.Y('TEXT', axis=alt.Axis(title='Number of Notes')))
```

```python

combined_df.groupby('HADM_ID')
    
```

```python
df[df.groupby('session')['url'].transform(lambda x : x.isin(valid_urls).any())]
```

```python
combined_df[(combined_df.charttime >= combined_df.admittime) &
            (combined_df.charttime < (combined_df.admittime + pd.Timedelta(hours=24)))].sort_values('HADM_ID')

fd_df = combined_df[(combined_df.charttime >= combined_df.admittime) &
                    (combined_df.charttime < (combined_df.admittime + pd.Timedelta(hours=24)))].sort_values('HADM_ID')
len(fd_df.HADM_ID.unique())
```

```python

```

```python

```

```python

```

### Continuing on with Deep Learning

```python
if os.path.isfile(base_path/lm_file):
    print('loading existing language model')
    lm = load_data(base_path, lm_file, bs=bs)
else:
    print('ERROR: language model file not found.')
```

#### This is a very CPU and RAM intensive process - no GPU involved

```python
filename = base_path/class_file
if os.path.isfile(filename):
    data_cl = load_data(base_path, class_file, bs=bs)
else:
    # do I need a vocab here? test with and without...
    data_cl = (TextList.from_df(df, base_path, cols='TEXT', vocab=lm.vocab)
               #df has several columns; actual text is in column TEXT
               .split_by_rand_pct(valid_pct=valid_pct, seed=seed)
               #We randomly split and keep 20% for validation, set see for repeatability
               .label_from_df(cols='los')
               #building classifier to automatically determine DESCRIPTION
               .databunch(bs=bs))
    data_cl.save(filename)
```

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5)
learn.load_encoder(enc_file)
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

Change learning rate based on results from the above plot

```python
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
```

Now need to fine tune

```python
learn.unfreeze()
```

```python

```

```python

```

```python

```
