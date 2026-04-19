# human-powered-gpt

AI datacenters consume an alarming amount of power. We fixed this.

This repo trains AI with the exact same neural network architecture as GPT-2, but using a more sustainable energy source: you. Just get on the bike.

## How

Target hardware: Raspberry Pi 3 (2017) powered by a 5W bike dynamo. 1 hour of riding at this wattage is sufficient to train a character-level language model that can properly speak English.

```
pip install -r requirements.txt
```

```
python dataset_gen.py   # generate the mad-libs chat corpus (deterministic)
python train.py         # train a char-level GPT
python train.py chat    # chat with it

python sweep.py         # IsoFLOPs scaling sweep across compute budgets
python plot_sweep.py    # Chinchilla-style plots of the sweep
```

Run `python train.py` to create the model.

After training is complete, run `python chat.py` to interactively chat with the model.
