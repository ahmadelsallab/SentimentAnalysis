load input_data;

% Reduce train features
features = mTrainFeatures;
reduced_size = (size(features, 2) / 3);
num_cases = size(features, 1);
new_features = zeros(num_cases, reduced_size);

% Loop on each entry in the 2x2 matrix
for i = 1 : size(features, 1)
    for j = 1 : reduced_size
        % (+): positive ArSenL score
        % (-): negative ArSenL score
        % (0): neutral ArSenL score
        % The order of entries (=ArSenL scores) is: (+), (-), (0)
        % The new entry is the subtraction of (+) - (-)
        % Neutral to be suppressed
        new_features(i, j) = features(i, j) - features(i, j + 1);
    end
end
clear mTrainFeatures;
mTrainFeatures = new_features;

% Reduce train features
features = mTestFeatures;
reduced_size = (size(features, 2) / 3);
num_cases = size(features, 1);
new_features = zeros(num_cases, reduced_size);

% Loop on each entry in the 2x2 matrix
for i = 1 : size(features, 1)
    for j = 1 : reduced_size
        % (+): positive ArSenL score
        % (-): negative ArSenL score
        % (0): neutral ArSenL score
        % The order of entries (=ArSenL scores) is: (+), (-), (0)
        % The new entry is the subtraction of (+) - (-)
        % Neutral to be suppressed
        new_features(i, j) = features(i, j) - features(i, j + 1);
    end
end
clear mTestFeatures;
mTestFeatures = new_features;
save input_data_senti_neutral_supp_diff_pos_neg

        