'''
Created on Nov 27, 2014

@author: asallab
'''

import pickle
from DatasetBuilder.DatasetBuilder import DatasetBuilder
from LanguageModel.LanguageModel import LanguageModel   
from FeaturesExtractor.FeaturesExtractor import FeaturesExtractor
from Classifiers.Classifier import Classifier

# Start the DatasetBuilder
#-------------------------
# Configurations file xml of the dataset builder
configFileDatasetBuilder = "configurations\\Configurations_DatasetBuilder.xml"

datasetSerializationFile = "output_results\\results_tweets.bin"

# Initialize the DatasetBuilder from serialization file
datasetBuilder = DatasetBuilder(configFileDatasetBuilder, [], datasetSerializationFile)
datasetBuilder.LoadDataset()

# Load the dataset. The TwitterCrawler.GetTweetsByID already loaded the labels and tweets all info
# inlcuding tweet['text'] and tweet['label'] used by FeaturesExtractor
datasetBuilder.SplitTrainTest()

# Start the LanguageModel
#-------------------------
# Configurations file xml of the language model
configFileLanguageModel = "configurations\\Configurations_LanguageModel-lexicon.xml"

#positiveLangModelTxtLoadFile = ".\\LanguageModel\\Input\\Eshrag-positive.txt"
positiveLangModelTxtLoadFile = ".\\LanguageModel\\Input\\positive.txt"
#negativeLangModelTxtLoadFile = ".\\LanguageModel\\Input\\Eshrag-negative.txt"
negativeLangModelTxtLoadFile = ".\\LanguageModel\\Input\\negative.txt"
stopWordsFileName = ".\\LanguageModel\\Input\\stop_words.txt"
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

 
# Initialize the FeaturesExtractor
testFeaturesExtractor = FeaturesExtractor(configFileFeaturesExtractor, [], [], languageModel, datasetBuilder.testSet)
testFeaturesExtractor.ExtractLexiconFeatures()


# Start the Classifier:
#----------------------
# The serialization file to save the features
configFileClassifier = ".\\Classifier\\Configurations\\Configurations_Classifier-lexicon.xml"


classifier = Classifier(configFileClassifier, [],  None, None, testFeaturesExtractor.features, testFeaturesExtractor.labels)


# Train
#classifier.Train()

# Test
labels, acc, val = classifier.Test()

# Build the confusion matrix
mConfusionMatrix, mNormalConfusionMatrix, vNumTrainExamplesPerClass, vAccuracyPerClass, nOverallAccuracy = classifier.BuildConfusionMatrix(testFeaturesExtractor.labels, labels)
print(mConfusionMatrix)

'''
def LoadDatasetAndLabels(datasetSerializationFileName):
    
    # Load the dataset
    serializatoinDatasetFile = open(datasetSerializationFileName, 'rb')
    
    # Load array of tweets, it include sentiment labels
    dataset = pickle.load(serializatoinDatasetFile)
    
    # Close the serialization file
    serializatoinDatasetFile.close()
'''    