# BeatUNet

To run the code, first you will have to run the following commands

```mkdir models```

```mkdir figures```

## Training the model
---
```python3 train.py```

The model requires two files, namely padded_max_logmels and padded_frameset.


These files will have to be created using Librosa Mel_spectrum.


Expected input is n*128*4821, and the output will be n*5*4821.

## Evaluation
---
For evaluating, you will have to replace the varaible in eval.py named as MODEL_PATH to the model saved with the best path.
Then run

```python3 eval.py```

## Output format
---
The output consists of 4 instruments and one silence.

1 : 'kick'

2 : 'snare'

3 : 'hihat'

4 : 'clap'
