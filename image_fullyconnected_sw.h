#ifndef SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_
#define SRC_LENET5_SW_LAYERS_IMAGE_FULLYCONNECTED_SW_H_

void FULLY_CONNECTED_LAYER_1_SW(float* input_feature, float* weight, float* bias, float* output_feature);
void FULLY_CONNECTED_LAYER_2_SW(float* input_feature, float* weight, float* bias, float* output_feature);

void FULLY_CONNECTED_LAYER_1_SW(float* input_feature, float* weight, float* bias, float* output_feature)
{
//	cout << "FULLY_CONNECTED_LAYER_1_SW :begin" << endl;
	int batch_cnt;
	int depth_out, depth_in;
	float temp;
	for(batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++) {
		for(depth_out = 0; depth_out < OUTPUT_NN_1_SIZE; depth_out++) {
			temp = 0;
			for(depth_in = 0; depth_in < INPUT_NN_1_SIZE; depth_in++) {
				float in_val = input_feature[batch_cnt*INPUT_NN_1_SIZE + depth_in];
				float w_val = weight[depth_in*OUTPUT_NN_1_SIZE + depth_out];
				temp += in_val*w_val;
			}
			output_feature[batch_cnt*OUTPUT_NN_1_SIZE + depth_out] = tanhf(temp + bias[depth_out]);
		}
	}
//	cout << "FULLY_CONNECTED_LAYER_1_SW :done" << endl;
}

void FULLY_CONNECTED_LAYER_2_SW(float* input_feature, float* weight, float* bias, float* output_feature)
{
//	cout << "FULLY_CONNECTED_LAYER_2_SW :begin" << endl;
	int batch_cnt;
	int depth_out, depth_in;
	float temp;
	for(batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++) {
		for(depth_out = 0; depth_out < OUTPUT_NN_2_SIZE; depth_out++) {
			temp = 0;
			for(depth_in = 0; depth_in < INPUT_NN_2_SIZE; depth_in++) {
				float in_val = input_feature[batch_cnt*INPUT_NN_2_SIZE + depth_in];
				float w_val = weight[depth_in*OUTPUT_NN_2_SIZE + depth_out];
				temp += in_val*w_val;
			}
			output_feature[batch_cnt*OUTPUT_NN_2_SIZE + depth_out] = tanhf(temp + bias[depth_out]);
		}
	}
//	cout << "FULLY_CONNECTED_LAYER_2_SW :done" << endl;
}

void FULLY_CONNECTED_LAYER_1_SW_debug(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
			for (int i = 0; i < OUTPUT_NN_1_SIZE; i++) {
				float temp = 0;
				for (int j = 0; j < INPUT_NN_1_SIZE; j++) {
					float in_val = input_feature[j];
					float w_val = weights[j*84+i];
					temp += in_val*w_val;
				}
				output_feature[batch*84 + i] = tanhf(temp + bias[i]);
			}
		}
}
void FULLY_CONNECTED_LAYER_2_SW_debug(float* input_feature, float* weights, float* bias, float* output_feature){
	for (int batch = 0; batch < image_Batch; batch++) {
		for (int i = 0; i < OUTPUT_NN_2_SIZE; i++) {
			float temp = 0;
			for (int j = 0; j < INPUT_NN_2_SIZE; j++) {
				float in_val = input_feature[j];
				float w_val = weights[j*10+i];
				temp += in_val*w_val;//input_feature[batch*84 + j] * weights[j*10 + i];
			}
			output_feature[batch*10 + i] = tanhf(temp + bias[i]);
		}
	}
}
#endif
