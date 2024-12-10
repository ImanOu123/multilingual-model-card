# ACL 60-60 evaluation data

## Overview
This zipped repository contains the ACL 60-60 evaluation sets, described in the paper ["Evaluating Multilingual Speech Translation Under Realistic Conditions with Resegmentation and Terminology"](https://aclanthology.org/2023.iwslt-1.2) at IWSLT 2023, and used in the [IWSLT 2023 Multilingual shared task](https://iwslt.org/2023/multilingual).
For further details please consult the paper, or contact the first author by email. 

We ask that if you use these evaluation sets in your work, you cite the following paper:
```
@inproceedings{salesky-2023-evaluating,
    title={{Evaluating Multilingual Speech Translation Under Realistic Conditions with Resegmentation and Terminology}},
    author={Salesky, Elizabeth and Darwish, Kareem and Al-Badrashiny, Mohamed and Diab, Mona and Niehues, Jan},
    booktitle = "Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)",
    month = "july",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```

## Files and Folders

We include both the final dataset (`acl-6060/`) as well as intermediate files (`intermediate_files/`) which may be beneficial for future research.

The `acl-6060/` directory contains for each of the `dev` and `eval` sets the full talk wav files, segmented wavs (following the sentence segmentation as well as a SHAS baseline segmentation), and the text in both xml and text formats, additionally with tagged terminology.

The `intermediate_files/` directory contains the initial ASR and MT output, with post-editing, as well as the terminology files from the ACL 60-60 initiative. The ASR output was initially segmented by VAD and then resegmented to sentences, and both files are included. 

### Directory structure

* acl-6060
  * dev
    * full_wavs
    * segmented_wavs
      * gold
      * shas
    * text
      * xml
      * txt
      * tagged_terminology
  * eval
    * full_wavs
    * segmented_wavs
      * gold
      * shas
    * text
      * xml
      * txt
      * tagged_terminology
* intermediate_files
  * transcription
    * asr_output
    * postedit
    * postedit_domain
  * translation
    * mt_output
    * postedit
  * terminology_glossary.csv
