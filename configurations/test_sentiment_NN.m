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

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([11385 100 2]);
opts.numepochs =  5;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
er

%assert(er < 0.08, 'Too big error');

%% ex2 neural net with L2 weight decay
rand('state',0)
nn = nnsetup([11385 100 2]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  5;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
er
%assert(er < 0.1, 'Too big error');


%% ex3 neural net with dropout
rand('state',0)
nn = nnsetup([11385 100 2]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  5;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
er
%assert(er < 0.1, 'Too big error');

%% ex4 neural net with sigmoid activation function
rand('state',0)
nn = nnsetup([11385 100 2]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  5;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
er
%assert(er < 0.1, 'Too big error');

%% ex5 plotting functionality
rand('state',0)
nn = nnsetup([11385 20 2]);
opts.numepochs         = 5;            %  Number of full sweeps through data
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
er
%assert(er < 0.1, 'Too big error');

%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

rand('state',0)
nn                      = nnsetup([11385 20 2]);     
nn.output               = 'softmax';                   %  use softmax output
opts.numepochs          = 5;                           %  Number of full sweeps through data
opts.batchsize          = 1000;                        %  Take a mean gradient step over this many samples
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');
