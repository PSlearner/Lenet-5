// image_convolution.h
#ifndef SRC_LENET5_SW_LAYERS_IMAGE_CONVOLUTION_SW_H_
#define SRC_LENET5_SW_LAYERS_IMAGE_CONVOLUTION_SW_H_
#include "common.h"
//#include<iostream>
void CONVOLUTION_LAYER_1_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						);

void CONVOLUTION_LAYER_2_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						);
						
void CONVOLUTION_LAYER_3_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						);	

	
						
void CONVOLUTION_LAYER_1_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						)
{
	int row, col, row_f, col_f;
	int depth_out, batch_cnt;
	
	float temp = 0;
//	cout << "CONVOLUTION_LAYER_1_SW :begin" << endl;
	for(batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++){
		for(depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			for(row = 0; row < CONV_1_OUTPUT_WH; row++) {
				for(col = 0; col < CONV_1_OUTPUT_WH; col++) {
					//compute one output pixel
					temp = 0;
					for(row_f = 0; row_f < CONV_1_WH; row_f++) {
						for(col_f = 0; col_f < CONV_1_WH; col_f++) {
							float in_val = input_feature[batch_cnt*INPUT_SIZE + (row+row_f)*INPUT_WH + col + col_f];
							float w_val = conv_kernel[depth_out*CONV_1_SIZE + row_f*CONV_1_WH + col_f];
							temp += in_val*w_val;
						}
					}
					output_feature[(batch_cnt*CONV_1_TYPE + depth_out)*CONV_1_OUTPUT_SIZE + 
									row*CONV_1_OUTPUT_WH + col] = tanhf(temp + conv_bias[depth_out]);
					
				}
				
			}
		}
	}
//	cout << "CONVOLUTION_LAYER_1_SW :done" << endl;
}
// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14], Conv_kernel[16][6][5x5], Bias[16], Output_feature_map[16][10x10]
void CONVOLUTION_LAYER_2_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						)
{
//	cout << "CONVOLUTION_LAYER_2_SW :begin" << endl;
	int row, col, row_f, col_f; //行列参数
	int depth_in, depth_out; //输入通道和输出通道
	int batch_idx; 
	float temp = 0;
	for(int i = 0; i < image_Batch*CONV_2_TYPE*CONV_2_OUTPUT_SIZE; i++) {
		output_feature[i] = 0;
	}
	for(batch_idx = 0; batch_idx < image_Batch; batch_idx++) {
		for(depth_out = 0; depth_out < CONV_2_TYPE; depth_out++) {
			for(depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
				if(!tbl[depth_in*16 + depth_out])
					continue;
				for(row = 0; row < CONV_2_OUTPUT_WH; row++) {
					for(col = 0; col < CONV_2_OUTPUT_WH; col++) {
						
						//compute one output pixel
						temp = 0;
						for(row_f = 0; row_f < CONV_2_WH; row_f++) {
							for(col_f = 0; col_f < CONV_2_WH; col_f++) {
								float in_val = input_feature[(batch_idx*CONV_1_TYPE + depth_in)*CONV_2_INPUT_SIZE + (row + row_f)*CONV_2_INPUT_WH + col + col_f];  
//								float in_val=input_feature[(batch_idx * CONV_1_TYPE + depth_in)*CONV_2_INPUT_SIZE + CONV_2_INPUT_WH * (row_f + row) + col + col_f];
								float w_val = conv_kernel[(depth_out*CONV_1_TYPE + depth_in)*CONV_2_SIZE + row_f*CONV_2_WH + col_f];
								temp += in_val*w_val;
							}
						}
						
						output_feature[(batch_idx * CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_SIZE + CONV_2_OUTPUT_WH * row + col] += temp; 
					}
				}
			}
			for(int i = 0; i < CONV_2_OUTPUT_SIZE; i++) {
				float old_val = output_feature[(batch_idx*CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_SIZE + i];
				float bias = conv_bias[depth_out];
				output_feature[(batch_idx*CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_SIZE + i] = tanhf(old_val + bias);
			}
		}
	}
}

// Convolution Layer 3(FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5], Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3_SW(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature
						)
{
//	cout << "CONVOLUTION_LAYER_3_SW :begin" << endl;
	int row, col, row_f, col_f; //
	int depth_in, depth_out; //
	int batch_cnt; 
	float temp = 0;
	for(batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++) {
		for(depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {
			//compute one output pixel
			float bias = conv_bias[depth_out];
			temp = 0;
			for(depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
				for(row_f = 0; row_f < CONV_3_WH; row_f++) {
					for(col_f = 0; col_f < CONV_3_WH; col_f++) {
						float in_val = input_feature[(batch_cnt*POOL_2_TYPE + depth_in)*CONV_3_INPUT_SIZE + 
													 row_f*CONV_3_INPUT_WH + col_f];
						float w_val = conv_kernel[(depth_out* POOL_2_TYPE + depth_in)*CONV_3_SIZE + 
													row_f*CONV_3_WH + col_f];
						temp += in_val*w_val;
					}
				}
			}
			output_feature[(batch_cnt*CONV_3_TYPE + depth_out)*CONV_3_OUTPUT_SIZE] = tanhf(temp+bias);
		}
	}
//	cout << "CONVOLUTION_LAYER_3_SW :done" << endl;
}
void CONVOLUTION_LAYER_1_SW_debug(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature)
{

	int col, row, col_f, row_f;
	int depth_out, batch_cnt;

	float temp=0;

	for(batch_cnt=0; batch_cnt<image_Batch; batch_cnt++) {
		for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
				for (col = 0; col < CONV_1_OUTPUT_WH; col++) {

					// Init
					temp = 0;

					// Multiplication by Convolution and Input feature map
					for (row_f = 0; row_f < CONV_1_WH; row_f++) {
						for (col_f = 0; col_f < CONV_1_WH; col_f++) {
							float in_val = input_feature[batch_cnt*INPUT_SIZE + INPUT_WH * (row + row_f) + col + col_f];
							float w_val = conv_kernel[depth_out*CONV_1_SIZE + CONV_1_WH * row_f + col_f];
							temp += in_val*w_val;
							//temp += input_feature[batch_cnt*INPUT_SIZE + INPUT_WH * (row + row_f) + col + col_f] *
							//		conv_kernel[depth_out*CONV_1_SIZE + CONV_1_WH * row_f + col_f];
						}
					}

					output_feature[(batch_cnt*CONV_1_TYPE + depth_out)*CONV_1_OUTPUT_WH*CONV_1_OUTPUT_WH +
									  CONV_1_OUTPUT_WH * row + col] = tanhf(temp+conv_bias[depth_out]);
				}
			}
		}
	}
}



// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14],  Conv_kernel[16][6][25], Bias[16], Output_feature_map[16][10x10]

void CONVOLUTION_LAYER_2_SW_debug(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature)
{
	// Connection Table for Dummy Operation



	int col, row;
	int col_f, row_f;
	int depth_in, depth_out;
	float temp = 0;
	int batch_idx;

	for(int i=0;i<1600;i++){
		output_feature[i] = 0;
	}
	for (batch_idx = 0; batch_idx < image_Batch; batch_idx++) {

		for (depth_out = 0; depth_out < CONV_2_TYPE; depth_out++) {
			for (depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
				if(!tbl[depth_in*16+depth_out])
					continue;
				for (row = 0; row < CONV_2_OUTPUT_WH; row++) {
					for (col = 0; col < CONV_2_OUTPUT_WH; col++) {

						// Init
						temp = 0;

					// Multiplication by Convolution and Input feature maps

						//printf("temp is %.6f\n", temp);
						for (row_f = 0; row_f < CONV_2_WH; row_f++) {
							for (col_f = 0; col_f < CONV_2_WH; col_f++) {
								//printf("Pixel is %.6f\n", conv_kernel[depth_out][depth_in][CONV_2_WH * row_f + col_f]);
								float in_val=input_feature[(batch_idx * CONV_1_TYPE + depth_in)*CONV_2_INPUT_SIZE + CONV_2_INPUT_WH * (row_f + row) + col + col_f];
								float w_val=conv_kernel[(depth_out*CONV_1_TYPE+depth_in)*25 + CONV_2_WH * row_f + col_f];
								temp += in_val*w_val;
								//temp += input_feature[(batch_idx * CONV_1_TYPE + depth_in)*CONV_2_INPUT_SIZE + CONV_2_INPUT_WH * (row_f + row) + col + col_f] *
								//		conv_kernel[(depth_out*CONV_1_TYPE+depth_in)*25 + CONV_2_WH * row_f + col_f];
							}
							//printf("\n");
						}
						output_feature[(batch_idx * CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_WH*CONV_2_OUTPUT_WH+
								   CONV_2_OUTPUT_WH * row + col] += temp;
						//printf("\n");
					}
					// Result of Convolution
					//output_feature[(batch_idx * CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_WH*CONV_2_OUTPUT_WH+
					//			   CONV_2_OUTPUT_WH * row + col] = tanhf(temp + conv_bias[depth_out]);
				}
			}
			for(int i=0;i<CONV_2_OUTPUT_SIZE;i++){
				float old_val = output_feature[(batch_idx * CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_SIZE+i];
				float bias = conv_bias[depth_out];
				float new_val= old_val + bias;
				output_feature[(batch_idx * CONV_2_TYPE + depth_out)*CONV_2_OUTPUT_SIZE+i] = tanhf(new_val);
			}
		}
	}
//	for(int i=0;i<1600;i++){
//		output_feature[i] = tanhf(output_feature[i]);
//	}
}
// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]

void CONVOLUTION_LAYER_3_SW_debug(float* input_feature,
							float* conv_kernel,
							float* conv_bias,
							float* output_feature)
{
	int col, row, col_f, row_f;
	int depth_in, batch_cnt, depth_out;

	float temp=0;

	for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++) {
		for (depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {

			// Init
			temp = 0;

			// Multiplication by Convolution and Input feature maps
			for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
				for (row_f = 0; row_f < CONV_3_WH; row_f++) {
					for (col_f = 0; col_f < CONV_3_WH; col_f++) {
						float in_val = input_feature[(POOL_2_TYPE * batch_cnt + depth_in)*CONV_3_WH*CONV_3_WH + CONV_3_WH * row_f + col_f];
						float w_val = conv_kernel[(depth_out*CONV_2_TYPE+depth_in)*5*5+row_f*5 + col_f];
						temp += in_val * w_val;
						//temp += input_feature[(POOL_2_TYPE * batch_cnt + depth_in)*CONV_3_WH*CONV_3_WH + CONV_3_WH * row_f + col_f] *
						//		conv_kernel[(depth_out*CONV_2_TYPE+depth_in)*5*5+row_f*5 + col_f];
					}
				}
			}
			// Result of Convolution
			float bias = conv_bias[depth_out];
			output_feature[batch_cnt * CONV_3_TYPE + depth_out] = (temp+ bias);
		}
	}
	for(int i=0;i<120;i++){
		output_feature[i] = tanhf(output_feature[i]);
	}
}



#endif

