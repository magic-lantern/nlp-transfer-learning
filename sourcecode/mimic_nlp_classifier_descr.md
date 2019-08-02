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

# Based on our custom MIMIC language model, train a 'DESCRIPTION' classifier

Make sure mimic_nlp_lm has been run first and sucessfully completed. That notebook builds the language model that allows classificiation to occur.

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

class_file = 'descr_cl_data.pickle'
notes_pickle_file = base_path/'noteevents.pickle'
lm_file = 'mimic_lm.pickle' # actual file is at base_path/lm_file but due to fastai function, have to pass file name separately
init_model_file = base_path/'descr_cl_head'
cycles_file = base_path/'descr_cl_num_iterations.pickle'
enc_file = 'mimic_fine_tuned_enc'
freeze_two = 'descr_cl_freeze_two'
freeze_three = 'descr_cl_freeze_three'
descr_ft_file = 'descr_cl_fine_tuned_'

training_history_file = 'descr_cl_history'
```

Setup parameters for models

```python
# original data set too large to work with in reasonable time due to limted GPU resources
pct_data_sample = 0.1
# how much to hold out for validation
valid_pct = 0.2
# for repeatability - different seed than used with language model
seed = 1776
# batch size of 96 GPU uses 15GB RAM
# batch size of 64 GPU uses 11GB RAM
# batch size of 48 GPU uses ??GB RAM
# changing batch size affects learning rate
bs=96
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

Since seed is different, this should be quite different than the language model dataset.

Should I show details on how many records are in language model dataset?

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

```python
# quote an imbalance between various DESCRIPTIONS
df.DESCRIPTION.value_counts()
```

```python
len(df.ROW_ID.unique())
```

```python
if os.path.isfile(base_path/lm_file):
    print('loading existing language model')
    lm = load_data(base_path, lm_file, bs=bs)
else:
    print('ERROR: language model file not found.')
```

#### This is a very CPU and RAM intensive process - no GPU involved

Also, since there are a wide range of descriptions, not all descriptions present in the test set are in the validation set, so cannot learn all of them.

```python
filename = base_path/class_file
if os.path.isfile(filename):
    data_cl = load_data(base_path, class_file, bs=bs)
    print('loaded existing data bunch')
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
    print('created new data bunch')
```

### Using weighted F1 to account for class imbalance

See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5, metrics=[accuracy, FBeta(average='weighted', beta=1)])
learn.load_encoder(enc_file)
```

```python
learn.lr_find()
```

This rate will vary based on batch size. 

      For bs=96, 5e-2 worked well.
      For bs=48, looks like 1e-1 would work

```python
learn.recorder.plot()
```

## Now train model

Change learning rate based on results from the above plot

First unfrozen training results in approximately 90% accuracy with `learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))`

     epoch 	train_loss 	valid_loss 	accuracy 	f_beta 	time
        0 	0.508831 	0.406110 	0.907089 	0.888122 	18:41
        
By comparison, a smaller learning rate takes longer to get to similar accuracy (`learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))`)

    Total time: 25:38

    epoch 	train_loss 	valid_loss 	accuracy 	time
        0 	0.451051 	0.413487 	0.909619 	25:38

<!-- #region -->
Evaluate some different learning rates:

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5)
learn.load_encoder(enc_file)
learn.fit_one_cycle(3, 1e-1, moms=(0.8,0.7))
```

```python
learn = text_classifier_learner(data_cl, AWD_LSTM, drop_mult=0.5)
learn.load_encoder(enc_file)
learn.fit_one_cycle(3, 5e-2, moms=(0.8,0.7))
```
<!-- #endregion -->

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

<!-- #region -->
```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-2/(2.6**4),5e-2), moms=(0.8,0.7))
```

    epoch 	train_loss 	valid_loss 	accuracy 	f_beta 	time
        0 	0.344032 	0.281245 	0.943689 	0.931991 	20:37
<!-- #endregion -->

```python
if os.path.isfile(str(freeze_two) + '.pth'):
    learn.load(freeze_two)
    print('loaded freeze_two learner')
else:
    print('Training new freeze_two learner')
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(5e-2/(2.6**4),5e-2), moms=(0.8,0.7))
    print('Saving new freeze_two learner')
    learn.save(freeze_two)
    print('Finished generating new freeze_two learner')
```

<!-- #region -->
```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
```

    epoch 	train_loss 	valid_loss 	accuracy 	f_beta 	time
        0 	0.323191 	0.250100 	0.948268 	0.937939 	31:50
<!-- #endregion -->

```python
if os.path.isfile(str(freeze_three) + '.pth'):
    learn.load(freeze_three)
    print('loaded freeze_three learner')
else:
    print('Training new freeze_three learner')
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
    print('Saving new freeze_three learner')
    learn.save(freeze_three)
    print('Finished generating new freeze_three learner')
```

```python
learn.unfreeze()
```

```python
learn.lr_find()
learn.recorder.plot()
```

```python
release_mem()
```

```python
if os.path.isfile(cycles_file):
    with open(cycles_file, 'rb') as f:
        prev_cycles = pickle.load(f)
    print('This model has been trained for', prev_cycles, 'epochs already')  
else:
    prev_cycles = 0
```

    epoch 	train_loss 	valid_loss 	accuracy 	f_beta 	time
        0 	0.326372 	0.248257 	0.947810 	0.939235 	42:19
        1 	0.264299 	0.233219 	0.951448 	0.941490 	43:48
        2 	0.241548 	0.217816 	0.952870 	0.942942 	42:32
        3 	0.262864 	0.202371 	0.957014 	0.947445 	35:17
        4 	0.248916 	0.201936 	0.957111 	0.948590 	39:39

```python
num_cycles = 5

file = descr_ft_file + str(prev_cycles)
learner_file = base_path/file
callback_save_file = str(learner_file) + '_auto'

learn.fit_one_cycle(num_cycles, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7),
                    callbacks=[
                        callbacks.SaveModelCallback(learn, every='epoch', monitor='accuracy', name=callback_save_file),
                        # CSVLogger only logs when num_cycles are complete
                        callbacks.CSVLogger(learn, filename=training_history_file, append=True)
                    ])
file = descr_ft_file + str(prev_cycles + num_cycles)
learner_file = base_path/file
learn.save(learner_file)

with open(cycles_file, 'wb') as f:
    pickle.dump(num_cycles + prev_cycles, f)
release_mem()
```

```python
num_cycles = 2

file = descr_ft_file + str(prev_cycles)
learner_file = base_path/file
callback_save_file = str(learner_file) + '_auto'

learn.fit_one_cycle(num_cycles, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7),
                    callbacks=[
                        callbacks.SaveModelCallback(learn, every='epoch', monitor='accuracy', name=callback_save_file),
                        # CSVLogger only logs when num_cycles are complete
                        callbacks.CSVLogger(learn, filename=training_history_file, append=True)
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
