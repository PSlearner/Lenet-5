#include "lenet5.h"
#include "./MNIST_DATA/MNIST_DATA.h"
#include <vector>
#include <numeric>
#include <ctime>
void load_model(string filename, float* weight, int size) {

	ifstream file(filename.c_str(), ios::in);
	if (file.is_open()) {
		for (int i = 0; i < size; i++) {
			float temp = 0.0;
			file >> temp;
			weight[i] = temp;
		}
	}else{
		cout<<"Loading model is failed : "<<filename<<endl;
	}
}
int compare(float* feature_1, float* feature_2, int length, int layer_name){
	printf("Length is %d ", length);
	for(int i = 0; i < length; i++){
		if(feature_1[i] != feature_2[i+length]){
			printf("%d IS ERROR\n", layer_name);
			return 0;
		}
	}
	printf("%d IS OK\n", layer_name);
	return 1;
}
int main(int argc, char *argv[])
{
	float* MNIST_IMG;
	int* MNIST_LABEL;
	if(argc==1){
		MNIST_IMG = (float*) malloc(image_Move*MNIST_PAD_SIZE*sizeof(float)); // MNIST TEST IMG
		MNIST_LABEL = (int*) malloc(image_Move*sizeof(int)); // MNIST TEST LABEL
		if(!MNIST_IMG || !MNIST_LABEL){
			cout<< "Memory allocation error(0)"<<endl;
			exit(1);
		}

		// read MNIST data & label
		READ_MNIST_DATA("MNIST_DATA/t10k-images.idx3-ubyte",MNIST_IMG,-1.0f, 1.0f, image_Move);
		READ_MNIST_LABEL("MNIST_DATA/t10k-labels.idx1-ubyte",MNIST_LABEL,image_Move,false);
	}
	float* Wconv1 = (float*)malloc(CONV_1_TYPE*CONV_1_SIZE*sizeof(float));
	float* bconv1 = (float*)malloc(CONV_1_TYPE*sizeof(float));
	float* Wconv2 = (float*)malloc(CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE*sizeof(float));
	float* bconv2 = (float*)malloc(CONV_2_TYPE*sizeof(float));
	float* Wconv3 = (float*)malloc(CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE*sizeof(float));
	float* bconv3 = (float*)malloc(CONV_3_TYPE*sizeof(float));

	float* Wpool1 = (float*) malloc(POOL_1_TYPE*4*sizeof(float));
	float* Wpool2 = (float*) malloc(POOL_2_TYPE*4*sizeof(float));
	float* bpool1 = (float*) malloc(POOL_1_TYPE*sizeof(float));
	float* bpool2 = (float*) malloc(POOL_2_TYPE*sizeof(float));

	float* Wfc1 = (float*) malloc(FILTER_NN_1_SIZE*sizeof(float));
	float* bfc1 = (float*) malloc(BIAS_NN_1_SIZE*sizeof(float));
	float* Wfc2 = (float*) malloc(FILTER_NN_2_SIZE*sizeof(float));
	float* bfc2 = (float*) malloc(BIAS_NN_2_SIZE*sizeof(float));
	
	if(!Wconv1||!Wconv2||!Wconv3||!bconv1||!bconv2||!bconv3||!Wpool1||!Wpool2||!bpool1||!bpool2||!Wfc1||!Wfc2||!bfc1||!bfc2){
		cout<<"mem alloc error(1)"<<endl;
		exit(1);
	}
	cout<<"Load models"<<endl;
	load_model("filter/Wconv1.mdl",Wconv1,CONV_1_TYPE*CONV_1_SIZE);

	load_model("filter/Wconv3_modify.mdl",Wconv2,CONV_2_TYPE*CONV_1_TYPE*CONV_2_SIZE);
	load_model("filter/Wconv5.mdl",Wconv3,CONV_3_TYPE*CONV_2_TYPE*CONV_3_SIZE);

	load_model("filter/bconv1.mdl",bconv1,CONV_1_TYPE);
	load_model("filter/bconv3.mdl",bconv2,CONV_2_TYPE);
	load_model("filter/bconv5.mdl",bconv3,CONV_3_TYPE);

	load_model("filter/Wpool1.mdl",Wpool1,POOL_1_TYPE*4);
	load_model("filter/Wpool2.mdl",Wpool2,POOL_2_TYPE*4);

	load_model("filter/bpool1.mdl",bpool1,POOL_1_TYPE);
	load_model("filter/bpool2.mdl",bpool2,POOL_2_TYPE);

	load_model("filter/Wfc1.mdl",Wfc1,FILTER_NN_1_SIZE);
	load_model("filter/Wfc2.mdl",Wfc2,FILTER_NN_2_SIZE);

	load_model("filter/bfc1.mdl",bfc1,BIAS_NN_1_SIZE);
	load_model("filter/bfc2.mdl",bfc2,BIAS_NN_2_SIZE);
	cout<<"model loaded"<<endl;
	
	// Memory allocation
	float* input_layer	= (float*) malloc(image_Batch *INPUT_WH * INPUT_WH*sizeof(float));
	float* hconv1 		= (float*) malloc(image_Batch * CONV_1_TYPE * CONV_1_OUTPUT_SIZE*sizeof(float));
	float* pool1 		= (float*) malloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE*sizeof(float));
	float* hconv2 		= (float*) malloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE*sizeof(float));
	float* pool2 		= (float*) malloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE*sizeof(float));
	float* hconv3 		= (float*) malloc(image_Batch * CONV_3_TYPE*sizeof(float));
	float* hfc1 		= (float*) malloc(image_Batch * OUTPUT_NN_1_SIZE*sizeof(float));
	float* output 		= (float*) malloc(image_Batch * OUTPUT_NN_2_SIZE*sizeof(float));
	//Debug
	float* input_layer_debug	= (float*) malloc(image_Batch *INPUT_WH * INPUT_WH*sizeof(float));
	float* hconv1_debug 		= (float*) malloc(image_Batch * CONV_1_TYPE * CONV_1_OUTPUT_SIZE*sizeof(float));
	float* pool1_debug 		= (float*) malloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE*sizeof(float));
	float* hconv2_debug 		= (float*) malloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE*sizeof(float));
	float* pool2_debug 		= (float*) malloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE*sizeof(float));
	float* hconv3_debug 		= (float*) malloc(image_Batch * CONV_3_TYPE*sizeof(float));
	float* hfc1_debug 		= (float*) malloc(image_Batch * OUTPUT_NN_1_SIZE*sizeof(float));
	float* output_debug 		= (float*) malloc(image_Batch * OUTPUT_NN_2_SIZE*sizeof(float));
	
	if(!input_layer || !hconv1 || !pool1 || !hconv2 || !pool2 || !hconv3 || !hfc1 || !output){
		cout<<"Memory allocation error(2)"<<endl;
		exit(1);
	}
	
	if(!input_layer_debug || !hconv1_debug || !pool1_debug || !hconv2_debug || !pool2_debug || !hconv3_debug || !hfc1_debug || !output_debug){
		cout<<"Memory allocation error(2_debug)"<<endl;
		exit(1);
	}
	///////////////////////////////// TEST /////////////////////////////////////////


	// cycle counters
	//perf_counter hw_ctr_tot, hw_ctr_conv1, hw_ctr_conv2, hw_ctr_conv3, hw_ctr_fc1, hw_ctr_fc2;//hw_ctr_pool1, hw_ctr_pool2,
	//perf_counter sw_ctr_tot, sw_ctr_conv1, sw_ctr_conv2, sw_ctr_conv3, sw_ctr_fc1, sw_ctr_fc2;//sw_ctr_pool1, sw_ctr_pool2,

	// test number
	int test_num = image_Move/image_Batch;
	
	vector<double> result_sw;
	double accuracy_sw;
	// SW test
	cout<< "SW test start"<<endl;
	
//	for(int i=0;i<test_num;i++){
	for(int i=0;i<1;i++){
		for(int batch=0;batch<image_Batch*INPUT_WH*INPUT_WH;batch++){
			input_layer[batch] = MNIST_IMG[1*MNIST_PAD_SIZE + batch];
		}
		CONVOLUTION_LAYER_1_SW(input_layer,Wconv1,bconv1,hconv1);

		POOLING_LAYER_1_SW(hconv1,Wpool1,bpool1,pool1);

		CONVOLUTION_LAYER_2_SW(pool1,Wconv2,bconv2,hconv2);

		POOLING_LAYER_2_SW(hconv2,Wpool2,bpool2,pool2);

		CONVOLUTION_LAYER_3_SW(pool2,Wconv3,bconv3,hconv3);

		FULLY_CONNECTED_LAYER_1_SW(hconv3,Wfc1,bfc1,hfc1);
		cout << "FULLY_CONNECTED_LAYER_1_SW :done" << endl;

		FULLY_CONNECTED_LAYER_2_SW(hfc1,Wfc2,bfc2,output);
		cout << "FULLY_CONNECTED_LAYER_2_SW :done" << endl;

		cout << "1 Predict:" << argmax(output) << " Actual:" << MNIST_LABEL[i*image_Batch+1] << endl;
	}
	
	
	for(int i=0;i<1;i++){
		for(int batch=0;batch<image_Batch*INPUT_WH*INPUT_WH;batch++){
			input_layer_debug[batch] = MNIST_IMG[batch];
		}
		CONVOLUTION_LAYER_1_SW(input_layer_debug,Wconv1,bconv1,hconv1_debug);
		if(!compare(hconv1, hconv1_debug, 4704, 1)) {
			break;
		}
		POOLING_LAYER_1_SW(hconv1_debug,Wpool1,bpool1,pool1_debug);
		if(!compare(pool1, pool1_debug, 1176, 2)) {
			break;
		}
		
		CONVOLUTION_LAYER_2_SW(pool1_debug,Wconv2,bconv2,hconv2_debug);
		if(!compare(hconv2, hconv2_debug, 1600, 3)) {
			break;
		}
		POOLING_LAYER_2_SW(hconv2_debug,Wpool2,bpool2,pool2_debug);
		if(!compare(pool2, pool2_debug, 400, 4)) {
			break;
		}
		CONVOLUTION_LAYER_3_SW(pool2_debug,Wconv3,bconv3,hconv3_debug);
		if(!compare(hconv3, hconv3_debug, 120, 5)) {
			break;
		}
		FULLY_CONNECTED_LAYER_1_SW(hconv3_debug,Wfc1,bfc1,hfc1_debug);
		if(!compare(hfc1, hfc1_debug, 84, 6)) {
			break;
		}
//		cout << "FULLY_CONNECTED_LAYER_1_SW :done" << endl;

		FULLY_CONNECTED_LAYER_2_SW(hfc1_debug,Wfc2,bfc2,output_debug);
		if(!compare(output, output_debug, 10, 7)) {
			break;
		}
//		cout << "FULLY_CONNECTED_LAYER_2_SW :done" << endl;

		cout << "2 Predict:" << argmax(output_debug) << " Actual:" << MNIST_LABEL[i*image_Batch+1] << endl;
	}
	
	
	free(input_layer);
	free(hconv1);
	free(hconv2);
	free(hconv3);
	free(pool1);
	free(pool2);
	free(hfc1);
	free(output);


	free(Wconv1);
	free(Wconv2);
	free(Wconv3);
	free(bconv1);
	free(bconv2);
	free(bconv3);
	free(Wpool1);
	free(bpool1);
	free(Wpool2);
	free(bpool2);
	free(Wfc1);
	free(bfc1);
	free(Wfc2);
	free(bfc2);

	free(MNIST_IMG);
	free(MNIST_LABEL);
}
