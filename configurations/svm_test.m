clear, clc;

% Directory and file names settings
base_dir = 'C:\Non_valeo\Research\PostDoc\Sentiment Analysis\Code\Datasets\ATB\';
targets_file_name = [base_dir 'annotation_sentiment.txt'];
results_file_name = 'OMA_results.xlsx';
features_dir_name = [base_dir '\features\'];
features_dir = dir(features_dir_name);
features_file_names = {features_dir(~[features_dir.isdir]).name};

% Set the results file header
s = xlswrite(results_file_name, {'Linear SVM'}, 'results', 'A1');
s = xlswrite(results_file_name, {'Features'}, 'results', 'A2');
s = xlswrite(results_file_name, {'Accuracy'}, 'results', 'B2');
s = xlswrite(results_file_name, {'Precision'}, 'results', 'C2');
s = xlswrite(results_file_name, {'Recall'}, 'results', 'D2');
s = xlswrite(results_file_name, {'F1 score'}, 'results', 'E2');

% Loop on all features files
for i = 1 : size(features_file_names, 2)
    % Read the features
    results.classifier = 'Linear SVM';
    features_file_names{i}
    s = xlswrite(results_file_name, {features_file_names{i}}, 'results', ['A'  int2str(i+2)]);
    results.features_file_name =  [features_dir_name features_file_names{i}];
    f = csvread(results.features_file_name);
    t = csvread(targets_file_name);
    
    % Split into train and test
    f_train = f(237:end,:);
    f_test = f(1:236,:);
    t_train = t(237:end,:);
    t_test = t(1:236,:);

    % Fit SVM model
    %svmStruct = svmtrain(f_train, t_train, 'kernel_function', 'rbf', 'rbf_sigma', 0.5);
    %svmStruct = svmtrain(f_train, t_train, 'kernel_function', 'mlp');
    svmStruct = svmtrain(f_train, t_train, 'kernel_function', 'linear');

    % Get test error rate
    t_pred = svmclassify(svmStruct, f_test);

    % Calculate accuracy
    results.errRate = sum(t_test ~= t_pred)/size(t_test, 1);
    results.acc = 1 - results.errRate;
    
    s = xlswrite(results_file_name, results.acc, 'results', ['B'  int2str(i+2)]);
    
    % Calculate confusion matrix
    results.conMat = confusionmat(t_test, t_pred);
    
    % Calculate precision
    results.pr = results.conMat(1,1)/(results.conMat(1,1) + results.conMat(2,1));
    
    s = xlswrite(results_file_name, results.pr, 'results', ['C'  int2str(i+2)]);
    
    % Calculate recall
    results.re = results.conMat(1,1)/(results.conMat(1,1) + results.conMat(1,2));
    
    s = xlswrite(results_file_name, results.re, 'results', ['D'  int2str(i+2)]);
    
    % Calculate F1 score
    results.F1 = 2 * (results.pr*results.re)/(results.pr+results.re);
    
    s = xlswrite(results_file_name, results.F1, 'results', ['E'  int2str(i+2)]);
    
    % Save and display results
    results_array{i} = results;
    results
    results.conMat
    save('results.mat', 'results_array', '-append');
end