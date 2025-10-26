# k_anonymity
An anonymization algorithm (k anonymity) is used for a real world dataset. the dataset cannot be directly uploaded to github due to file size limitations.
This is the link for the dataset: https://www.cdc.gov/brfss/annual_data/annual_2021.html
The effects of k anonymity to machine learning models are tested with diffrent k values.

preprocess folder has codes for diffrent types of preprocesses like advanced nan value handling, removing specific instances, manually binning the values etc.

after preprocess the dataset can be inserted to the anonymizer to apply k anonimity.

For the given dataset, codes at the folder cdc_decesiontrees can be used for comparing the effects of the k anonimity on decesion trees and codes at deep_learning folder can be used for comparing the effects of the k anonimity on deep learning.

