'''
Created on Nov 27, 2014

@author: asallab
'''
'''
import os
print(os.getcwd())
'''
import pickle
from datasetbuilder.DatasetBuilder import DatasetBuilder
from languagemodel.LanguageModel import LanguageModel   
from featuresextractor.FeaturesExtractor import FeaturesExtractor
from classifiers.Classifier import Classifier

# Start the DatasetBuilder
#-------------------------
# Configurations file xml of the dataset builder
configFileDatasetBuilder = "configurations\\Configurations_DatasetBuilder.xml"

datasetSerializationFile = "output_results\\results_tweets.bin"

# Initialize the DatasetBuilder from serialization file
datasetBuilder = DatasetBuilder(configFileDatasetBuilder, [], datasetSerializationFile)
#datasetBuilder.LoadDataset()
datasetBuilder.trainSet = []
#datasetBuilder.trainSet.append({'label': 'Positive', 'text':'سهم تاسي سجل ارتفاعا'});
#datasetBuilder.trainSet.append({'label': 'Positive', 'text':'ارتفاعا'});
datasetBuilder.trainSet.append({'label': 'Positive', 'text':'الما'});

# Load the dataset. The TwitterCrawler.GetTweetsByID already loaded the labels and tweets all info
# inlcuding tweet['text'] and tweet['label'] used by FeaturesExtractor
#datasetBuilder.SplitTrainTest()

# Start the LanguageModel
#-------------------------
# Configurations file xml of the language model
configFileLanguageModel = "configurations\\Configurations_LanguageModel-lexicon.xml"

#positiveLangModelTxtLoadFile = ".\\LanguageModel\\Input\\Eshrag-positive.txt"
positiveLangModelTxtLoadFile = "input_data\\positive.txt"
#negativeLangModelTxtLoadFile = ".\\LanguageModel\\Input\\Eshrag-negative.txt"
negativeLangModelTxtLoadFile = "input_data\\negative.txt"
stopWordsFileName = "input_data\\stop_words.txt"
# The serialization file to save the model
languageModelSerializationFile = ".\\LanguageModel\\Output\\language_model.bin"

# Start the LanguageModel:

# Initialize the LanguageModel
languageModel = LanguageModel(configFileLanguageModel, stopWordsFileName, [], [], datasetBuilder.trainSet)
languageModel.LoadSentimentLexiconModelFromTxtFile(positiveLangModelTxtLoadFile, 1)
languageModel.LoadSentimentLexiconModelFromTxtFile(negativeLangModelTxtLoadFile, -1)


# Start the FeaturesExtractor:
#-----------------------------
# Configurations file xml of the features extractor
configFileFeaturesExtractor = "configurations\\Configurations_FeaturesExtractor-Lexicon.xml"
exportFileName = "output_results\\features.txt"
 
# Initialize the FeaturesExtractor
testFeaturesExtractor = FeaturesExtractor(configFileFeaturesExtractor, [], [], languageModel, datasetBuilder.trainSet)
testFeaturesExtractor.ExtractLexiconFeatures()
testFeaturesExtractor.DumpFeaturesToTxt(exportFileName)


# Start the Classifier:
#----------------------
# The serialization file to save the features
configFileClassifier = "configurations\\Configurations_Classifier-lexicon.xml"


classifier = Classifier(configFileClassifier, [],  None, None, testFeaturesExtractor.features, testFeaturesExtractor.labels)


# Train
#classifier.Train()

# Test
print(classifier.LexiconPredict(testFeaturesExtractor.features[0]))