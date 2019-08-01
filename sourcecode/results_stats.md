---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python (fastai)
    language: python
    name: fastai
---

## Notebook with various summary statistics for results section

```python
from fastai.text import *
```

```python
# pandas doesn't understand ~, so provide full path
base_path = Path.home() / 'mimic'

# files used during processing - all aggregated here
lm_file = 'mimic_lm.pickle' # actual file is at base_path/lm_file but due to fastai function, have to pass file name separately
admissions_file = base_path/'ADMISSIONS.csv'
notes_file = base_path/'NOTEEVENTS.csv'
notes_pickle_file = base_path/'noteevents.pickle'

bs = 96
```

```python
tmpfile = base_path/lm_file

if os.path.isfile(tmpfile):
    print('loading existing language model')
    data_lm = load_data(base_path, lm_file, bs=bs)
```

### Compare vocabulary between pre-trained WT-103 and MIMIC

```python
f = '/home/seth/models/wt103-fwd/itos_wt103.pkl'
if os.path.isfile(f):
    with open(f, 'rb') as f:
        wt103_itos = pickle.load(f)
```

```python
len(wt103_itos)
```

```python
len(data_lm.vocab.itos)
```

```python
len(set(wt103_itos) & set(data_lm.vocab.itos))
#set(wt103_itos) & set(data_lm.vocab.itos)
```

```python
len(set(data_lm.vocab.itos) - set(wt103_itos))
#set(data_lm.vocab.itos) - set(wt103_itos)
```

```python
if 'bronchiectasis' in wt103_itos:
    print('found in wt103 lm')
if 'bronchiectasis' in data_lm.vocab.itos:
    print('found in mimic lm')
```

### Overlap between clinical notes used for language model and clinical notes used for DESCRIPTION classifier

```python
# original data set too large to work with in reasonable time due to limted GPU resources
pct_data_sample = 0.1
# for repeatability - different seed than used with language model
desc_seed = 1776
lm_seed = 42
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

```python
desc_df = orig_df.sample(frac=pct_data_sample, random_state=desc_seed)
```

```python
desc_df.shape
```

```python
lm_df = orig_df.sample(frac=pct_data_sample, random_state=lm_seed)
```

```python
lm_df.shape
```

```python
len(set(desc_df.ROW_ID.unique()) & set(lm_df.ROW_ID.unique()))
```

```python

```

```python

```
