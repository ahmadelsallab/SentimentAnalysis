clear, clc;
features_file_name = 'C:\Non_valeo\Research\PostDoc\Sentiment Analysis\Code\Datasets\ATB\features\arsenl_lemma (SentiScore).csv';
targets_file_name = 'C:\Non_valeo\Research\PostDoc\Sentiment Analysis\Code\Datasets\ATB\annotation_sentiment.txt';
f = csvread(features_file_name);
t = csvread(targets_file_name);

% Split into train and test
mTrainFeatures = f(237:end,:);
mTestFeatures = f(1:236,:);
mTrainTargets = t(237:end,:);
mTestTargets = t(1:236,:);

