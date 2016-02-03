import argparse
import os
import logging
import shutil
import sys
import subprocess

import eval.ner.sequences.extended_feature as exfc
import eval.ner.sequences.structured_perceptron as spc
from eval.ner.PrepareEmbedRep import PrepareEmbedRep
from eval.ner.PrepareHmmRep import PrepareHmmRep
from eval.ner.lxmls.readers.Conll2003NerCorpus import Conll2003NerCorpus, eng_train, eng_test, eng_dev, muc_test


def setup_logging(dirname, level):
    """
    :param level: logging.INFO or logging.DEBUG, etc.
    """
    logfile = "{}/log".format(dirname)
    logging.basicConfig(filename=logfile, level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    return logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="output directory, e.g. ../data/Conll2002/data/output/I20")
    parser.add_argument("-rep", "--rep_path", help="directory containing (hmm) word representations files")
    parser.add_argument("-d", "--decoding",
                        choices=["viterbi", "max-emission", "max-product", "posterior", "posterior_cont",
                                 "posterior_cont_type"],
                        help="method used for decoding: viterbi, posterior,...")
    parser.add_argument("-brown", "--brown_cluster_file", help="path to file with brown clusters")
    parser.add_argument("--rel_spec", action='store_true', default=False,
                        help="if wordreps are based on specific (syntactic) relations")
    parser.add_argument("--ignore_rel",
                        help="dependency relation name to ignore when decoding. Makes sense only together with rel_spec")
    parser.add_argument("--embed", help="path to file with word embeddings")
    parser.add_argument("--embed_v", help="path to vocabulary file of the text used for inducing word embeddings")
    args = parser.parse_args()
    if args.ignore_rel is not None and not args.rel_spec:
        sys.exit("Ignore relation but no rel_spec option specified.")
    if args.output_dir is not None:
        outdir = args.output_dir
    else:
        sys.exit("Output directory path missing!")
    if not "Conll2003" in outdir:
        sys.exit("Output directory probably wrong!")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        # prevent overwriting old stuff
        import datetime

        now = datetime.datetime.now().isoformat().replace(":", "-")
        outdir = "{}_{}".format(outdir.rstrip("/"), now)
        print("Output directory already exists, appending with timestamp.")
        os.makedirs(outdir)

    logger = setup_logging(outdir, logging.DEBUG)

    brown_cluster_file = args.brown_cluster_file
    hmm_rep_path = args.rep_path
    lr = ("_lr_" in hmm_rep_path) if hmm_rep_path is not None else False
    if hmm_rep_path is not None and "tree" in hmm_rep_path:
        use_wordrep_tree = True
        use_wordrep_rel = False
        logger.info("Using tree representations.")
    elif hmm_rep_path is not None and ("_rel_" in hmm_rep_path or lr):
        use_wordrep_rel = True
        use_wordrep_tree = False
        logger.info("Using tree representations.")
    else:
        use_wordrep_tree = False
        use_wordrep_rel = False

    decoding = args.decoding
    #decoding = "max_emission"
    #decoding = "max-product"
    # use hmm-based wordrep
    if hmm_rep_path is not None:
        logger.info("Loading corpora. Decoding word representations.")
        hmmrep = PrepareHmmRep(hmm_rep_path, lang="en", decoding=decoding, use_wordrep_tree=use_wordrep_tree,
                               use_wordrep_rel=use_wordrep_rel, eval_spec_rel=args.rel_spec, logger=logger,
                               ignore_rel=args.ignore_rel, lr=lr)
        corpus = hmmrep.ner_corpus
        train_seq = hmmrep.train_seq
        dev_seq = hmmrep.dev_seq
        test_seq = hmmrep.test_seq
        muc_seq = hmmrep.muc_seq
    elif args.embed is not None:
        logger.info("Loading embeddings.")
        embrep = PrepareEmbedRep(embed=args.embed, embed_v=args.embed_v, lang="en")
        corpus = embrep.ner_corpus
        train_seq = embrep.train_seq
        dev_seq = embrep.dev_seq
        test_seq = embrep.test_seq
        muc_seq = embrep.muc_seq
    else:
        logger.info("Loading corpora.")
        corpus = Conll2003NerCorpus()
        train_seq = corpus.read_sequence_list_conll(eng_train)
        dev_seq = corpus.read_sequence_list_conll(eng_dev)
        test_seq = corpus.read_sequence_list_conll(eng_test)
        muc_seq = corpus.read_sequence_list_conll(muc_test)

    logger.info("Extracting features.")
    #logger.info("Training on dev !!")
    #feature_mapper = exfc.ExtendedFeatures(dev_seq)
    feature_mapper = exfc.ExtendedFeatures(train_seq, brown_cluster_file)
    feature_mapper.set_baseline_features()
    # other/wordrep features
    if brown_cluster_file is not None:
        feature_mapper.brown_id_plus1 = True
        feature_mapper.brown_id_plus2 = True
        feature_mapper.brown_id_minus1 = True
        feature_mapper.brown_id_minus2 = True
        feature_mapper.brown_prefix = True  # prefix length features; same for all brown_id
        feature_mapper.brown_prefix_lengths = [4, 6, 10, 20]
        if feature_mapper.brown_prefix:
            if not feature_mapper.brown_prefix_lengths:
                sys.exit("Brown prefix lengths not defined.")
    else:
        feature_mapper.brown_id = False
        feature_mapper.brown_id_plus1 = False
        feature_mapper.brown_id_plus2 = False
        feature_mapper.brown_id_minus1 = False
        feature_mapper.brown_id_minus2 = False
        feature_mapper.brown_prefix = False  # prefix length features; same for all brown_id
        feature_mapper.brown_prefix_lengths = []

    if hmm_rep_path is not None or args.embed is not None:
        feature_mapper.rep_id = True
        feature_mapper.rep_id_plus1 = False
        feature_mapper.rep_id_plus2 = False
        feature_mapper.rep_id_minus1 = False
        feature_mapper.rep_id_minus2 = False
    else:
        feature_mapper.rep_id = False
        feature_mapper.rep_id_plus1 = False
        feature_mapper.rep_id_plus2 = False
        feature_mapper.rep_id_minus1 = False
        feature_mapper.rep_id_minus2 = False

    feature_mapper.build_features()

    logger.info("Training./Loading model.")

    sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
    sp.num_epochs = 20
    sp.train_supervised(train_seq, dev_seq)

    logger.info("Testing on dev.")
    pred_dev = sp.viterbi_decode_corpus(dev_seq)
    #eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
    #logger.info("Devset acc.: {}".format(eval_dev))
    logger.info("Writing conll eval format.")
    corpus.write_conll_instances(dev_seq, pred_dev, "{}/dev.txt".format(outdir))

    logger.info("Testing on test.")
    pred_test = sp.viterbi_decode_corpus(test_seq)
    logger.info("Writing conll eval format.")
    corpus.write_conll_instances(test_seq, pred_test, "{}/test.txt".format(outdir))

    logger.info("Testing on MUC test.")
    pred_muc = sp.viterbi_decode_corpus(muc_seq)
    logger.info("Writing conll eval format.")
    corpus.write_conll_instances(muc_seq, pred_muc, "{}/muc.txt".format(outdir), is_muc=True)

    logger.info("Saving model, writing the settings.")
    with open("{}/setting".format(outdir), "w") as setting_file:
        setting_file.write("Train file: {}\n".format(eng_train))
        setting_file.write("Dev file: {}\n".format(eng_dev))
        setting_file.write("Test file: {}\n".format(eng_test))
        setting_file.write("MUC test file: {}\n".format(muc_test))
        setting_file.write("Output directory: {}\n".format(outdir))
        setting_file.write("Loaded model parameters: {}\n".format(sp.loaded_model))
        setting_file.write("Number of features: {}\n".format(feature_mapper.get_num_features()))
        setting_file.write("Features used:\n")
        for f in sorted(list(feature_mapper.features_used)):
            setting_file.write("\t{}\n".format(f))
        setting_file.write("Number of labels: {}\n".format(sp.get_num_states()))
        setting_file.write("Classifier: see the experimental py file in the folder\n")
        setting_file.write("Classification task decoder: see the experimental py file in the folder\n")
        setting_file.write("Number of epochs: {}\n".format(sp.num_epochs))
        setting_file.write("Learning rate: {}\n".format(sp.learning_rate))
        setting_file.write("Averaged classifier: {}\n".format(sp.averaged))
        if brown_cluster_file:
            setting_file.write("Brown cluster file: {}\n".format(brown_cluster_file))
        if hmm_rep_path is not None:
            setting_file.write("Word rep (hmm) file: {}\n".format(hmm_rep_path))
            setting_file.write("Word rep decoder: {}\n".format(decoding))
            setting_file.write("Syntactic relation to ignore when decoding: {}\n".format(args.ignore_rel))
        if args.embed is not None:
            setting_file.write("Word rep (embedding) file: {}\n".format(args.embed))
            setting_file.write("Word rep (embedding) vocabulary file: {}\n".format(args.embed_v))
    # Save the model
    sp.save_model(outdir)

    # Copy this file
    curr_file = os.path.realpath(__file__)
    shutil.copy(curr_file, outdir)

    logger.info("Evaluating with official perl script.")
    # Run Perl evaluation

    dev_file = "{}/dev.txt".format(outdir)
    test_file = "{}/test.txt".format(outdir)
    muc_file = "{}/muc.txt".format(outdir)
    eval_script = "conlleval"
    p1_dev = subprocess.Popen(['cat', dev_file], stdout=subprocess.PIPE)
    p1_test = subprocess.Popen(['cat', test_file], stdout=subprocess.PIPE)
    p1_muc = subprocess.Popen(['cat', muc_file], stdout=subprocess.PIPE)
    p2_dev = subprocess.Popen(["perl", eval_script], stdin=p1_dev.stdout, stdout=subprocess.PIPE)
    p2_test = subprocess.Popen(["perl", eval_script], stdin=p1_test.stdout, stdout=subprocess.PIPE)
    p2_muc = subprocess.Popen(["perl", eval_script], stdin=p1_muc.stdout, stdout=subprocess.PIPE)
    p1_dev.stdout.close()
    p1_test.stdout.close()
    p1_muc.stdout.close()
    dev_result = p2_dev.communicate()[0].decode()
    test_result = p2_test.communicate()[0].decode()
    muc_result = p2_muc.communicate()[0].decode()
    with open("{}/dev.result".format(outdir), "w") as dev_out, \
            open("{}/test.result".format(outdir), "w") as test_out, \
            open("{}/muc.result".format(outdir), "w") as muc_out:
        dev_out.write("{}".format(dev_result))
        test_out.write("{}".format(test_result))
        muc_out.write("{}".format(muc_result))
    # extract f-score
    dev_score = dev_result.split("\n")[1].split(";")[-1].split(" ")[-1]
    test_score = test_result.split("\n")[1].split(";")[-1].split(" ")[-1]
    muc_score = muc_result.split("\n")[1].split(";")[-1].split(" ")[-1]
    logger.info("F-score dev: {}".format(dev_score))
    logger.info("F-score test: {}".format(test_score))
    logger.info("F-score MUC test: {}".format(muc_score))
