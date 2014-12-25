clear, clc;
f = csvread('arsenl_lemma (SentiScore).csv');
t_in = csvread('annotation_sentiment.txt');

t = zeros(size(t_in, 1), 2);
for i = 1 : size(t_in, 1)
   t(i, t_in(i)) =  1;
end
train_x = f(237:end,:);
test_x = f(1:236,:);
train_y = t(237:end,:);
test_y = t(1:236,:);
save('input_data.mat');

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 10 10 10 10];
opts.numepochs =   15;
opts.batchsize = 100;
opts.momentum  =   0.025;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  15;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er

%assert(er < 0.10, 'Too big error');
