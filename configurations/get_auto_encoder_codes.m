load input_data;
load final_net;
[mTrainFeatures, err] = TST_computeAutoEncoderCode(mTrainFeatures, NM_strNetParams, 0, 0, 0, 'Normal');
[mTestFeatures, err] = TST_computeAutoEncoderCode(mTestFeatures, NM_strNetParams, 0, 0, 0, 'Normal');
save input_data_auto_encoder_codes;