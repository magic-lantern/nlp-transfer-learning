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

## Using FAST.AI for Medical NLP - Step 1 Build a langauge model

Exploring the MIMIC III data set medical notes.

Tried working with the full dataset, but almost every training step takes many hours (~13 for initial training), predicted 14+ per epoch for fine tuning, and we need to do many epochs.

Instead will try to work with just 10% sample... Not sure that will work though

A few notes:
* See https://docs.fast.ai/text.transform.html#Tokenizer for details on what various artificial tokens (e.g xxup, xxmaj, etc.) mean
* To view nicely formatted documentation on the fastai library, run commands like: ` doc(learn.lr_find)`

```python
from fastai.text import *
from sklearn.model_selection import train_test_split
import glob
import gc
from pympler import asizeof
```

<!-- #region -->
If you want to verify that Torch can find and use your GPU, run the following code:

```python
import torch

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```
<!-- #endregion -->

<!-- #region -->
These next cells can be used to get an idea of the speed up provided by a GPU for some operations (from https://course.fast.ai/gpu_tutorial.html)
```python
import torch
t_cpu = torch.rand(500,500,500)
%timeit t_cpu @ t_cpu
# separate cell 
t_gpu = torch.rand(500,500,500).cuda()
%timeit t_gpu @ t_gpu
```
<!-- #endregion -->

```python
# original data set too large to work with in reasonable time due to limted GPU resources
pct_data_sample = 0.1
# how much to hold out for validation
valid_pct = 0.1

# pandas doesn't understand ~, so provide full path
base_path = Path.home() / 'mimic'

# files used during processing - all aggregated here
notes_file = base_path/'noteevents.pickle'
lm_file = 'mimic_lm.pickle' # actual file is at base_path/lm_file but due to fastai function, have to pass file name separately
init_model_file = base_path/'mimic_fit_head'
cycles_file = base_path/'num_iterations.pickle'
lm_base_file = 'mimic_lm_fine_tuned_'
enc_file = 'mimic_fine_tuned_enc'
```

```python
# if this doesn't free memory, can restart Python kernel.
# if that still doesn't work, try OS items mentioned here: https://docs.fast.ai/dev/gpu.html
def release_mem():
    gc.collect()
    torch.cuda.empty_cache()
```

```python
# run this to see what has already been imported
#whos
```

### Set Random Number seed for repeatability; set Batch Size to control GPU memory

See **"Performance notes"** section below for how setting batch size impacts GPU memory

```python
seed = 42
# previously used 48; worked fine but never seemed to use even half of GPU memory; 64 still on the small side
bs=96
```

While parsing a CSV and converting to a dataframe is pretty fast, loading a pickle file is much faster.

For load time and size comparison:
* `NOTEEVENTS.csv` is ~ 3.8GB in size
  ```
  CPU times: user 51.2 s, sys: 17.6 s, total: 1min 8s
  Wall time: 1min 47s
  ```
* `noteevents.pickle` is ~ 3.7 GB in size
  ```
  CPU times: user 2.28 s, sys: 3.98 s, total: 6.26 s
  Wall time: 6.26 s
  ```

```python
%%time

orig_df = pd.DataFrame()
if os.path.isfile(notes_file):
    print('Loading noteevent pickle file')
    orig_df = pd.read_pickle(notes_file)
else:
    print('Could not find noteevent pickle file; creating it')
    # run this the first time to covert CSV to Pickle file
    orig_df = pd.read_csv(base_path/'NOTEEVENTS.csv', low_memory=False, memory_map=True)
    orig_df.to_pickle(notes_file)
```

Due to data set size and performance reasons, working with a 10% sample. Use same random see to get same results from subsequent runs.

```python
df = orig_df.sample(frac=pct_data_sample, random_state=seed)
```

```python
# if you want to free up some memory
# orig_df = None
# del orig_df
# gc.collect()
```

```python
print('df:', int(asizeof.asizeof(df) / 1024 / 1024), 'MB')
#print('orig_df:', asizeof.asizeof(orig_df))
#print('data_lm:', asizeof.asizeof(data_lm, detail=1))
#print asizeof.asized(obj, detail=1).format()
```

```python
df.head()
```

```python
df.dtypes
```

```python
df.shape
```

Code to build initial version of language model; If running with full dataset, requires a **LOT** of RAM; using a **LOT** of CPU helps it to happen quickly as well

**Note:** By default, this only tracks up to 60,000 tokens (words usually). In my testing that is sufficient to get high accuracy

Questions:

* why does this only seem to use CPU? (applies to both both textclasdatabunch and textlist)
* for 100% of the mimic noteevents data:
  * run out of memory at 32 GB, error at 52 GB, trying 72GB now... got down to only 440MB free; if crash again, increase memory
  * now at 20vCPU and 128GB RAM; ok up to 93%; got down to 22GB available
  * succeeded with 20CPU and 128GB RAM...
* try smaller batch size? will that reduce memory requirements?
* with 10% dataset sample, it seems I could get by with perhaps 32GB system RAM

For comparison:
* 10% language model is ~ 1.2 GB in size
  * Time to load existing language model:
    ```
    CPU times: user 3.29 s, sys: 844 ms, total: 4.14 s
    Wall time: 12.6 s
    ```
  * Time to build language model:
    ```
    CPU times: user 36.9 s, sys: 8.56 s, total: 45.4 s
    Wall time: 3min 27s
    ```
* 100% language model is...
  * Time to load existing language model:
  * Time to build language model:

```python
%%time

tmpfile = base_path/lm_file

if os.path.isfile(tmpfile):
    print('loading existing language model')
    data_lm = load_data(base_path, lm_file, bs=bs)
else:
    print('creating new language model')
    data_lm = (TextList.from_df(df, 'texts.csv', cols='TEXT')
               #df has several columns; actual text is in column TEXT
               .split_by_rand_pct(valid_pct=valid_pct, seed=seed)
               #We randomly split and keep 10% for validation
               .label_for_lm()
               #We want to do a language model so we label accordingly
               .databunch(bs=bs))
    data_lm.save(tmpfile)
```

<!-- #region -->
If need to view more data, run appropriate line to make display wider/show more columns...
```python
# default 20
pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_columns', None) # show all
# default 50
pd.get_option('display.max_colwidth')
pd.set_option('display.max_colwidth', -1) # show all
```
<!-- #endregion -->

```python
data_lm.show_batch()
# how to look at original version of text
#df[df['TEXT'].str.contains('being paralyzed were discussed', case=False)].TEXT
```

```python
# as of June 2019, this automatically loads and initializes the model based on WT103 from
# https://s3.amazonaws.com/fast-ai-modelzoo/wt103-fwd.tgz; will auto download if not already on disk
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
```

```python
release_mem()
```

### Generate Learning rate graph.

```python
learn.lr_find()
learn.recorder.plot(skip_end=15)
```

### Initial model training

Time to run:

* Full data set took about 13 hours using the Nvidia P1000
* Full data set was predicted to take about 25 hours with the T4
* 10% data took about 1 hour (1:08) using the Nvidia P1000
* 10% data is predicted to take about 2.5 hour (actual 2:42) using the Nvidia GTX 1060


```python
release_mem()
```

```python
# no idea how long nor how much resources this will take
# not sure 1e-2 is the right learning rate; maybe 1e-1 or between 1e-2 and 1e-1
# using t4
# progress bar says this will take around 24 hours... ran for about 52 minutes
# gpustat/nvidia-smi indicates currently only using about 5GB of GPU RAM
# using p100
# progress bar says this will take around 12 hours; took 13:16
# at start GPU using about 5GB RAM
# after about 8 hours GPU using about 7.5GB RAM.
# looks like I could increase batch size...
# with bs=64, still only seems to be using about 7GB GPU RAM after running for 15 minutes. 
# will check after a bit, but likely can increase batch size further
#
# note about number of epochs/cycle length: Using a value of 1 does a rapid increase and
# decrease of learning rate and end result gets almost the save result as 2 but in half
# the time
if os.path.isfile(str(init_model_file) + '.pth'):
    learn.load(init_model_file)
    print('loaded learner')
else:
    learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
    learn.save(init_model_file)
    print('generated new learner')
```

```python
release_mem()
```

continue from initial training - reload in case just want to continue processing from here.

As an FYI pytorch automatically appends .pth to the filename, you cannot provide it

```python
#learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
#learn.load(init_model_file)
#print('done')
```

```python
learn.show_results()
```

```python
with open(cycles_file, 'wb') as f:
    pickle.dump(8, f)
```

```python
prev_cycles = 0

if os.path.isfile(cycles_file):
    with open(cycles_file, 'rb') as f:
        prev_cycles = pickle.load(f)
print('This model has been trained for', prev_cycles, 'epochs already')
```

```python
#temp_files = glob.glob(str(base_path/'*_auto_*'))
#if len(training_files) > 0:
rfiles = glob.glob(str(base_path/'*_auto_*'))
rfiles.sort()
if (len(rfiles) > 0):
    print('There are pre-existing automatic save states. Remove these files if no longer needed.')
for f in rfiles:
    print(f)
```

### Now fine tune language model

Performance notes w/P100 GPU:

* at batch size of 128 takes about 1:14:00 per epoch; GPU usage is about 14GB; RAM usage is about 10GB
* at batch size of 96 takes about 1:17:00 per epoch; GPU usage is about  9GB; RAM usage is about 10GB
* at batch size of 48 takes about 1:30:00 per epoch; GPU usage is about  5GB; RAM usage is about 10GB

With `learn.fit_one_cycle(8, 5e-3, moms=(0.8,0.7))` (8 cycles)
* gets from about 62.7% accuracy to 67.6% accuracy
* Total time: 9:54:16


    epoch 	train_loss 	valid_loss 	accuracy 	time
        0 	1.926960 	1.832659 	0.627496 	1:14:14
        1 	1.808083 	1.755725 	0.637424 	1:14:15
        2 	1.747903 	1.697741 	0.645431 	1:14:15
        3 	1.714081 	1.652703 	0.652703 	1:14:19
        4 	1.637801 	1.602961 	0.660170 	1:14:15
        5 	1.596906 	1.553225 	0.668557 	1:14:14
        6 	1.572020 	1.519172 	0.674477 	1:14:26
        7 	1.517364 	1.510010 	0.676342 	1:14:14
    
    
With `learn.fit_one_cycle(num_cycles, 5e-3, moms=(0.8,0.7)` (10 cycles)
* batch size `bs=96`
* Total time: 12:17:26


    epoch 	train_loss 	valid_loss 	accuracy 	time
        0 	1.876292 	1.813362 	0.630908 	1:13:40
        1 	1.816879 	1.770555 	0.635667 	1:13:41
        2 	1.833764 	1.769055 	0.635783 	1:13:45
        3 	1.765977 	1.729675 	0.641041 	1:13:43
        4 	1.672098 	1.683195 	0.648317 	1:13:52
        5 	1.639705 	1.637336 	0.655466 	1:13:43
        6 	1.600122 	1.589719 	0.663033 	1:13:45
        7 	1.529386 	1.546841 	0.670321 	1:13:43
        8 	1.527369 	1.518421 	0.675460 	1:13:41
        9 	1.512422 	1.511458 	0.676779 	1:13:42

    completed 10 new training epochs
    completed 10 total training epochs

Interesting to note, training for fewer epochs with the one cycle policy results in faster training. In either case, as the validation loss is still improving, can continue to train more to improve model.
```python
def custom_learner_load(lf):
    if os.path.isfile(str(lf) + '.pth'):
        learn.load(lf)
        print('loaded existing learner from', str(lf))
    else:
        # should not continue as could not find specified file
        print('existing learner file (', lf, ') not found, cannot continue')
        print('previous epoch may have only partially completed')
        print(' --- try updating prev_cycles to match or copy file to correct name.')
        assert(False)
    return learn
```

```python
# if want to continue training existing model, set to True
# if want to start fresh from the initialized language model, set to False
# also, make sure to remove any previously created saved states before changing
# flag back to continue
continue_flag = True
# Resume interrupted training
resume_flag = True
########################################################
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

########################################################
# set this to how many cycles you want to run
num_cycles = 10
########################################################

if continue_flag:
    if os.path.isfile(cycles_file):
        with open(cycles_file, 'rb') as f:
            prev_cycles = pickle.load(f)
        print('This model has been trained for', prev_cycles, 'epochs already')  
else:
    prev_cycles = 0

file = lm_base_file + str(prev_cycles)
learner_file = base_path/file
callback_save_file = str(learner_file) + '_auto'
fn_pattern = callback_save_file + '*'


# for one cycle learning with learning rate annealing - where to resume from
start_epoch = 0

if resume_flag:
    training_files = glob.glob(str(base_path/fn_pattern))
    if len(training_files) > 0:
        training_files.sort()
        completed_cycles = int(re.split('_|\.', training_files[-1])[-2])
        if completed_cycles < (num_cycles - 1):
            # need to load the last file
            print('Previous training cycle of', num_cycles, 'did not complete; finished',
                  completed_cycles + 1, 'cycles. Loading last save...')
            # load just filename, drop extension of .pth as that is automatically appended inside load function
            learn.load(os.path.splitext(training_files[-1])[0])
            start_epoch = completed_cycles + 1
        else:
            print('Previous training cycle of', num_cycles, 'completed fully.')
            learn = custom_learner_load(learner_file)
    else:
        print('No auto save files exist from interupted training.')
        if continue_flag:
            learn = custom_learner_load(learner_file)
        else:
            print('Starting training with base language model')
else:
    if continue_flag:
        learn = custom_learner_load(learner_file)
    else:
        print('Starting training with base language model')
    # remove any auto saves
    training_files = glob.glob(str(base_path/fn_pattern))
    if len(training_files) > 0:
        for f in training_files:
            print('Deleting', f)
            os.remove(f)

learn.unfreeze()
#learn.fit_one_cycle(num_cycles, 5e-3, moms=(0.8,0.7),
learn.fit_one_cycle(num_cycles, 5e-3, moms=(0.8,0.7),
                    callbacks=[
                        callbacks.SaveModelCallback(learn, every='epoch', monitor='accuracy', name=callback_save_file),
                        # CSVLogger only logs when num_cycles are complete
                        callbacks.CSVLogger(learn, filename='mimic_lm_fine_tune_history', append=True),
                        callbacks.EarlyStoppingCallback(learn, monitor='accuracy', min_delta=0.0025, patience=5)
                    ],
                    start_epoch=start_epoch)
file = lm_base_file + str(prev_cycles + num_cycles)
learner_file = base_path/file
learn.save(learner_file)

with open(cycles_file, 'wb') as f:
    pickle.dump(num_cycles + prev_cycles, f)
release_mem()
    
print('completed', num_cycles, 'new training epochs')
print('completed', num_cycles + prev_cycles, 'total training epochs')
```

<!-- #region -->
### Evaluate different learning rates.

Use this block of code to compare how well a few different learning rates work

Found that `5e-3` works best with `learn.unfreeze()`

```python
num_cycles = 4
prev_cycles = 4

#for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
    print('now testing with multiple epochs and learning rate of', lr)
    print('This model has been trained for', prev_cycles, 'epochs already')    
    file = lm_base_file + str(prev_cycles)
    learner_file = base_path/file
    learn.load(learner_file)
    learn.unfreeze()
    print('loaded existing learner from', str(learner_file))


    learn.fit_one_cycle(num_cycles, lr, moms=(0.8,0.7))
    file = lm_base_file + str(prev_cycles + num_cycles + 1)
    learner_file = base_path/file
    learn.save(learner_file)
    release_mem()

    print('completed', num_cycles, 'new training epochs')
    print('completed', num_cycles + prev_cycles, 'total training epochs')
```
<!-- #endregion -->

```python
# test the language generation capabilities of this model (not the point, but is interesting)
TEXT = "For confirmation, she underwent CTA of the lung which was negative for pulmonary embolism"
N_WORDS = 40
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
```

```python
learn.save_encoder(enc_file)
```

<!-- #region -->
To load the encoder:

```python
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.load_encoder(enc_file)
```
<!-- #endregion -->

```python
learn.summary()
```

```python
# see if learning rate has changed with training
learn.lr_find()
learn.recorder.plot(skip_end=15)
```

```python
learn.unfreeze()
```

```python

```
