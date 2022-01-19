# Gabriele Catanese - Aspect Based Sentiment Analysis on airline customer feedbacks
### Master's Degree in "*Linguistics: Text Mining*", VU Amsterdam.
This repository contains the code that was implemented for the Master's Thesis Project "*A Transfer Learning approach to Aspect Based Sentiment Analysis for airline customer feedbacks*".\
The project was carried out in collaboration with [Underlined](https://underlined.nl/).\
The author of this work is Gabriele Catanese, supervised by dr. Isa Maks and Lotte van Bakel.



The project consisted in fine-tuning [BERT](https://huggingface.co/bert-base-uncased) and [RoBERTa](https://huggingface.co/roberta-base) for Aspect Based Sentiment Analysis on airline customer feedback data. The retrievement of the sentiment information corresponding to a certain service purchased by the client provides an in-depth insight into the passengers satisfaction, serving as a key tool for an improved Customer Experience.\
The training involved 2 classifiers for each language model (BERT & RoBERTa): one for Aspect Category Detection, and the other for Sentiment Polarity classification.\
A comparative evalutation of results was carried out and reported on.\
The details about the development process, models setup, and results can be found in the thesis report, `Catanese-MasterThesis.pdf`.



## Folders
### /code
This folder contains the scripts and notebook used for this project. They were run in this order for the project.

* `create_annotation_data.ipynb` this notebook creates the datasets used for the annotation process from the original dataset provided by the client.
* `agreement.ipynb` this notebook calculated the Inter Annotator Agreement between the three annotators on the same batch of 50 sentences.
* `unify_dataset.ipynb` this notebook merges the three annotated sets into one complete dataset consisting of 2495 sentences.
* `stats.ipynb` this notebook extract the statistics regarding the complete annotated dataset.
* `train_aspect.ipynb` this notebook finetunes a selected language model between BERT and RoBERTa for Aspect Category Detection. The fine-tuned model is eventually saved to `/models`, to be loaded later on.
* `train_sentiment.ipynb` this notebook finetunes a selected language model between BERT and RoBERTa for Sentiment Polarity classification. The fine-tuned model is eventually saved to `/models`, to be loaded later on.
* `baseline.py` this script runs and evaluates a Majority Baseline used for comparison purposes.
* `results.ipynb` this notebook loads the finetuned models for evalutation and results analysis. Output files containing the results are generated when run.
* `utils.py` this script contains some helper functions.
* `absa_output.txt` this file is an explainatory example of the output of the `results.ipynb` notebook. 


### /data
This folder contained private airline customers feedback data used for the project.\
For illustration purposes, an example file, showing the structure of the datasets used for training, validation, and testing, is included to this folder.

### `Catanese-MasterThesis.pdf`
This file contains the thesis report.

### `requirements.txt`
The required Python 3.8 packages for running the code contained in this repository can be found in the `requirements.txt` file and installed directly through pip.

### /models
This folder stores the finetuned models after running `train_aspect.ipynb` and `train_sentiment.ipynb`. Then, they can be loaded from here.

## References
The code used for the fine-tuning step was adopted and modified for this task from the work of [George Mihaila](https://github.com/gmihaila).\
Source code: https://gmihaila.medium.com/fine-tune-transformers-in-pytorch-using-transformers-57b40450635
