from multilingualmc.translator.get_terms import TermCollector
tgt_langs = [
    "Chinese",
    "Arabic",
    "French",
    "Japanese",
    "Russian",
]

term_collector = TermCollector(
    "/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final",
    tgt_langs = tgt_langs
)


term = term_collector.find_terminology("""1: For all the methods, we used 10-fold cross validation (i.e., each fold we have 556 training and 62 test samples) to tune free parameters, e.g., the kernel form and parameters for GPOR and LapSVM. Note that all the alternative methods stack X and Z together into a whole data matrix and ignore their heterogeneous nature.<br>2: Features associated one-to-one with a vertical (Clarity, ReDDE, the query likelihood given the vertical's query-log and Soft.ReDDE) were normalized across verticals before scaling. Supervised training/testing was done via 10-fold cross validation. Parameter τ was tuned for each training fold on the same 500 query validation set used for our single feature baselines.""")

print(term)