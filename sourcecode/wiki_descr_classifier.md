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

# Based on general purpose language model, train a 'DESCRIPTION' classifier

Instead of building from a MIMIC trained language model, use the general purpose ULMFit Wiki trained model

```python
from fastai.text import *
from sklearn.model_selection import train_test_split
import glob
import gc
```

Setup filenames and paths

```python
# pandas doesn't understand ~, so provide full path
base_path = Path.home() / 'mimic'

# files used during processing - all aggregated here
admissions_file = base_path/'ADMISSIONS.csv'
notes_file = base_path/'NOTEEVENTS.csv'

class_file = 'wiki_cl_data.pickle'
notes_pickle_file = base_path/'noteevents.pickle'
lm_file = 'cl_lm.pickle' # actual file is at base_path/lm_file but due to fastai function, have to pass file name separately
init_model_file = base_path/'wiki_cl_head'
cycles_file = base_path/'wiki_cl_num_iterations.pickle'
enc_file = 'wiki_cl_enc'
descr_ft_file = 'wiki_cl_fine_tuned_'
```

Setup parameters for models

```python
# original data set too large to work with in reasonable time due to limted GPU resources
pct_data_sample = 0.1
# how much to hold out for validation
valid_pct = 0.2
# for repeatability - different seed than used with language model
seed = 1776
# batch size of 96 GPU needs more than 16GB RAM
# batch size of 64 GPU uses 16GB RAM
# batch size of 48 GPU uses ??GB RAM
# changing batch size affects learning rate
bs=64
```

```python
# if this doesn't free memory, can restart Python kernel.
# if that still doesn't work, try OS items mentioned here: https://docs.fast.ai/dev/gpu.html
def release_mem():
    gc.collect()
    torch.cuda.empty_cache()
release_mem()
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
df = orig_df.sample(frac=pct_data_sample, random_state=seed)
```

```python
df.head()
```

```python
print('Unique Categories:', len(df.CATEGORY.unique()))
print('Unique Descriptions:', len(df.DESCRIPTION.unique()))
```

<!-- #region -->
Original section from lesson3
```python
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)
             #grab all the text files in path
             .split_by_folder(valid='test')
             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
             .label_from_folder(classes=['neg', 'pos'])
             #label them all with their folders
             .databunch(bs=bs))

data_clas.save('data_clas.pkl')
```
<!-- #endregion -->

### Normally you would use transfer learning to adjust the language model to the new data.

In this case, I just want to test how the classifier would work without fine-tuning the language model

```python
%%time

tmpfile = base_path/lm_file

if os.path.isfile(tmpfile):
    print('loading existing language model')
    lm = load_data(base_path, lm_file, bs=bs)
else:
    print('creating new language model')
    lm = (TextList.from_df(df, base_path, cols='TEXT')
               #df has several columns; actual text is in column TEXT
               .split_by_rand_pct(valid_pct=valid_pct, seed=seed)
               #We randomly split and keep 10% for validation
               .label_for_lm()
               #We want to do a language model so we label accordingly
               .databunch(bs=bs))
    lm.save(tmpfile)
```

```python
learn = language_model_learner(lm, AWD_LSTM, drop_mult=0.3)
learn.save_encoder(enc_file)
```

#### This is a very CPU and RAM intensive process - no GPU involved

Also, since there are a wide range of descriptions, not all descriptions present in the test set are in the validation set, so cannot learn all of them.

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
               .label_from_df(cols='DESCRIPTION')
               #building classifier to automatically determine DESCRIPTION
               .databunch(bs=bs))
    data_cl.save(filename)
```

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5)
#learn.load_encoder(enc_file)
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

Change learning rate based on results from the above plot

First unfrozen training with `learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))` results in 

    Total time: 22:36

    epoch 	train_loss 	valid_loss 	accuracy 	time
        0 	0.967378 	0.638532 	0.870705 	22:36

Without loading existing encoder (customized encoder) and using `learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))`

    Total time: 20:26

    epoch 	train_loss 	valid_loss 	accuracy 	time
        0 	0.873634 	0.651192 	0.864657 	20:26
```python
if os.path.isfile(str(init_model_file) + '.pth'):
    learn.load(init_model_file)
    print('loaded initial learner')
else:
    print('Training new initial learner')
    learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))
    print('Saving new learner')
    learn.save(init_model_file)
    print('Finished generating new learner')
```

Now need to fine tune

```python
learn.unfreeze()
```

```python
release_mem()
```

```python
num_cycles = 5
prev_cycles = 0

file = descr_ft_file + str(prev_cycles)
learner_file = base_path/file
callback_save_file = str(learner_file) + '_auto'

learn.fit_one_cycle(num_cycles, 5e-3, moms=(0.8,0.7),
                    callbacks=[
                        callbacks.SaveModelCallback(learn, every='epoch', monitor='accuracy', name=callback_save_file),
                        # CSVLogger only logs when num_cycles are complete
                        callbacks.CSVLogger(learn, filename='descr_fine_tune_history', append=True)
                    ])
file = descr_ft_file + str(prev_cycles + num_cycles)
learner_file = base_path/file
learn.save(learner_file)

with open(cycles_file, 'wb') as f:
    pickle.dump(num_cycles + prev_cycles, f)
release_mem()
```

```python

```
