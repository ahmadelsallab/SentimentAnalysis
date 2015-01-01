clear, clc, close all;

fprintf(1, 'Configuring...\n');

% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_deep_auto();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainDeepAuto(CONFIG_strParams);