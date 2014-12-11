sFileName = 'train_data_file.txt'
[mFeatures, mTargets, nBitfieldLength, vChunkLength, vOffset] = DCONV_convert(sFileName, 'Normal')
collective_features_data = mFeatures(:, 481:482);
for i = 1 : size(mTargets ,1)
    targets(i) = find(mTargets(i, :) == 1);
end

pos = collective_features_data(:, 1);
neg = collective_features_data(:, 2);
scatter(pos, neg, 5, targets);
hold
text(pos, neg, num2str(targets), 'horizontal','left', 'vertical','bottom');

%   <Label label="Positive"></Label>
%   <Label label="Negative"></Label>
%   <Label label="Neutral"></Label>
%   <Label label="Mixed"></Label>

subplot(2,2,1);
i = find(targets == 1);
t = targets(i);
x = pos(i);
y = neg(i);
scatter(x, y, 5, t);
hold
text(x, y, num2str(t), 'horizontal','left', 'vertical','bottom');

subplot(2,2,2);
i = find(targets == 2);
t = targets(i);
x = pos(i);
y = neg(i);
scatter(x, y, 5, t);
hold
text(x, y, num2str(t), 'horizontal','left', 'vertical','bottom');

subplot(2,2,3);
i = find(targets == 3);
t = targets(i);
x = pos(i);
y = neg(i);
scatter(x, y, 5, t);
hold
text(x, y, num2str(t), 'horizontal','left', 'vertical','bottom');

subplot(2,2,4);
i = find(targets == 4);
t = targets(i);
x = pos(i);
y = neg(i);
scatter(x, y, 5, t);
hold
text(x, y, num2str(t), 'horizontal','left', 'vertical','bottom');