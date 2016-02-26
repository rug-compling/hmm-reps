#Hidden Markov models for word representations

Learn discrete and continuous word representations with Hidden Markov models, including variants defined over unlabeled and labeled parse trees.

(c) Simon Šuster, 2016

If you use this code for any of the tree HMM types, please cite our paper:

[Word Representations, Tree Models and Syntactic Functions](http://arxiv.org/abs/1508.07709). Simon Šuster, Gertjan van Noord and Ivan Titov. arXiv preprint arXiv:1508.07709, 2015. [bibtex](http://simonsuster.github.io/publications/SynFunc.bibtex) 
 


##Features
###Architectures and training
- Sequential HMM trained with the Baum-Welch algorithm. 
- Tree HMM trained with sum-product message passing (belief propagation)
    - unlabeled
    - labeled (includes syntactic functions as additional observed variables, effectively an Input- Output HMM)

###Training regimes
- Batch EM
- Online (mini-batch step-wise) EM (Liang and Klein 2009)

###Inference and decoding options
- Viterbi/Max-product message passing
- Posterior decoding (aka Minimum risk)
- Posterior distribution with inference
- Averaged posterior distribution per word type
- Maximum emission decoding (ignoring transitions)

###Implemented refinements
- Approximation of belief vectors (regularization) (Grave et al. 2013)
- Initialization of model parameters with Brown clusters
- Splitting and merging of states for progressive introduction of complexity (Petrov et al. 2006)

##Implementation
- Logspace approach instead of Rabiner rescaling: a C extension is used for fast log-sum exponent calculation
- Parallelization on the sentence level

Despite the mentioned, the running time is relatively slow and is especially sensitive to the number of states. A speed-up would be possible through the use of sparse matrices, but at several places the replacement is not trivial. 

##Input format
- For sequential HMM: plain text, one sentence per line, space-separated tokens
- For tree HMMs: CoNLL format

## Usage notes
The three main classes define the type of architecture that can be used:
- [hmm.py](hmm.py) for Hidden Markov models
- [hmtm.py](hmtm.py) for Hidden Markov tree models
- [hmrtm.py](hmrtm.py) for Hidden Markov tree models with syntactic relations (functions)

Models can be trained by invoking `run.py`. Use `--tree` to train a Hidden Markov tree model, and `--rel` to train a Hidden Markov tree model with syntactic relations.
A full list of options can be found by running:

```sh
python3.3 run.py --help
```

### Example runs

Sequential Hidden Markov model:

```sh
python3.3 readers/corpus_normalize.py --dataset data/sample.en --output data/sample.en.unk --freq_thresh 1
python3.3 run.py --dataset data/sample.en.unk --desired_n_states 60 --max_iter 2 --n_proc 1 --approx
```

This will create an output directory `hmm_...`. 

The following configuration will train the same model, but with the splitting procedure and Brown initialization:

```sh
python3.3 run.py --dataset data/sample.en --start_n_states 30 --desired_n_states 60 -brown sample.en.64.paths --max_iter 2 --n_proc 1 --approx
```


To train a Hidden Markov tree model:

```sh
python3.3 run.py --tree --dataset data/sample.en.conll --start_n_states 60 --desired_n_states 128 
 --max_iter 2 --n_proc 1 --approx
```

To train a Hidden Markov tree model with syntactic relations:

```sh
python3.3 run.py --rel --dataset data/sample.en.conll --desired_n_states 60 --max_iter 2 --n_proc 1 --approx
```

### Evaluating the representations
See the scripts under [output](output/) (qualitative) and under [eval](eval/) (named entity recognition).

##Requirements
- Python3.3 or higher
- Numpy 1.9 or higher
- [Fast logsumexp](https://github.com/rmcgibbo/logsumexp) 