
#include "base.h"
#include "for_json.h"

int idx_in_weight, idx_in_bias, idx_out_weight, idx_out_bias, idx_linear_weight, idx_linear_bias;
idx_in_weight = 1; idx_in_bias = 1; idx_out_weight = 1; idx_out_bias = 1; idx_linear_weight = 1; idx_linear_bias = 1;
struct arr* q_w, * k_w, * v_w;

void test_fun(void);

struct arr* attention(struct arr* for_q, struct arr* for_k, struct arr* for_v);
struct arr* attention_masked(struct arr* for_q, struct arr* for_k, struct arr* for_v, struct arr * attn_mask, struct arr * key_padding_mask);
void before_make_qkv(struct arr* for_in_weight, int qkv);
struct arr* make_qkv(struct arr* before_qkv, int* dim_size_list);
struct arr* MatMul(struct arr* arr1, struct arr* arr2, int doit_transpose, DType scale_factor);
void Softmax(struct arr* before_arr);
struct arr* permute_2013(struct arr* before_arr);
struct arr* Linear(struct arr* input_arr, int weight_use, int bias_use);
struct arr* view_2to3(struct arr* before_arr, int ch_idx);
struct arr* Add_33(struct arr* arr1, struct arr* arr2);
struct arr* Add_44(struct arr* arr1, struct arr* arr2);
void Norm_3D(struct arr* n_arr);
int check_total_dimension(struct arr* check_arr);

struct arr* FFN(struct arr* arr1);
void Dropout(struct arr* arr1, DType p);
void ReLU(struct arr* arr1);

//int idx_yewon_check = 1;

struct arr* my_transformer(struct arr* src, struct arr* tgt) {
	struct para_config init_config; init_config.para_version = 1;

	// encoder
	struct arr* memory = NULL;
	struct arr* encoder_attn = NULL;
	struct arr* encoder_ffn = NULL;

	memory = src;
	for (int i = 0; i < DEFAULT_NUM_ENCODER_LAYERS; i++) {
		printf("Encoder Layer [%d] ===========\n", i + 1);

		char idx_e_attn[256]; char idx_e_ffn[256];

		encoder_attn = Add_33(memory, attention(memory, memory, memory));
		Norm_3D(encoder_attn);

		encoder_ffn = Add_33(encoder_attn, FFN(encoder_attn));
		Norm_3D(encoder_ffn);

		memory = encoder_ffn;
	}
	Norm_3D(memory);

	// decoder
	struct arr* transformer_output = tgt;
	struct arr* decoder_attn1 = NULL;
	struct arr* decoder_attn2 = NULL;
	struct arr* decoder_ffn = NULL;

	for (int i = 0; i < DEFAULT_NUM_DECODER_LAYERS; i++) {
		printf("Decoder Layer [%d] ===========\n", i + 1);

		char idx_d_attn1[256]; char idx_d2_attn1[256]; char idx_d_ffn[256];

		decoder_attn1 = Add_33(transformer_output, attention(transformer_output, transformer_output, transformer_output));
		Norm_3D(decoder_attn1);

		decoder_attn2 = Add_33(decoder_attn1, attention(decoder_attn1, memory, memory));
		Norm_3D(decoder_attn2);

		decoder_ffn = Add_33(decoder_attn2, FFN(decoder_attn2));
		Norm_3D(decoder_ffn);

		transformer_output = decoder_ffn;
	}
	Norm_3D(transformer_output);

	free_arr(src); free_arr(tgt);
	free_arr(encoder_attn);
	free_arr(encoder_ffn);
	free_arr(decoder_attn1);
	free_arr(decoder_attn2);

	return transformer_output;

}

void basic_masked_transformer(void) {
	struct para_config init_config; init_config.para_version = 1;
	struct arr* src = read_json("../../../data/use_data", "src");
	struct arr* tgt = read_json("../../../data/use_data", "tgt");
	struct arr* src_mask = read_json("../../../data/use_data", "src_mask");
	struct arr* tgt_mask = read_json("../../../data/use_data", "tgt_mask");
	struct arr* memory_mask = read_json("../../../data/use_data", "memory_mask");
	struct arr* src_key_padding_mask = read_json("../../../data/use_data", "src_key_padding_mask");
	struct arr* tgt_key_padding_mask = read_json("../../../data/use_data", "tgt_key_padding_mask");
	struct arr* memory_key_padding_mask = read_json("../../../data/use_data", "memory_key_padding_mask");

	print_arr(src);
	print_arr(tgt);
	print_arr(src_mask);
	print_arr(tgt_mask);
	print_arr(memory_mask);
	print_arr(src_key_padding_mask);
	print_arr(tgt_key_padding_mask);
	print_arr(memory_key_padding_mask);
	printf("\n\n");


	// encoder
	struct arr* memory = NULL;
	struct arr* encoder_attn = NULL;
	struct arr* encoder_ffn = NULL;

	memory = src;
	for (int i = 0; i < DEFAULT_NUM_ENCODER_LAYERS; i++) {
		printf("Encoder Layer [%d] ===========\n", i + 1);

		char idx_e_attn[256]; char idx_e_ffn[256];

		encoder_attn = Add_33(memory, attention_masked(memory, memory, memory, src_mask, src_key_padding_mask));
		Norm_3D(encoder_attn);
		//sprintf(idx_e_attn, "encoder_attn_%d_c", (i + 1));
		//save_json("../../../data/compare_data", &idx_e_attn, encoder_attn);

		encoder_ffn = Add_33(encoder_attn, FFN(encoder_attn));
		Norm_3D(encoder_ffn);
		//sprintf(idx_e_ffn, "encoder_ffn_%d_c", (i + 1));
		//save_json("../../../data/compare_data", &idx_e_ffn, encoder_ffn);

		memory = encoder_ffn;
	}
	Norm_3D(memory);
	//save_json("../../../data/compare_data", "encoder_output_c", memory);

	// decoder
	struct arr* transformer_output = tgt;
	struct arr* decoder_attn1 = NULL;
	struct arr* decoder_attn2 = NULL;
	struct arr* decoder_ffn = NULL;

	for (int i = 0; i < DEFAULT_NUM_DECODER_LAYERS; i++) {
		printf("Decoder Layer [%d] ===========\n", i + 1);

		char idx_d_attn1[256]; char idx_d2_attn1[256]; char idx_d_ffn[256];

		decoder_attn1 = Add_33(transformer_output, attention_masked(transformer_output, transformer_output, transformer_output, tgt_mask, tgt_key_padding_mask));
		Norm_3D(decoder_attn1);
		//sprintf(idx_d_attn1, "decoder_attn1_%d_c", (i + 1));
		//save_json("../../../data/compare_data", &idx_d_attn1, decoder_attn1);

		decoder_attn2 = Add_33(decoder_attn1, attention_masked(decoder_attn1, memory, memory, memory_mask, memory_key_padding_mask));
		Norm_3D(decoder_attn2);
		//sprintf(idx_d2_attn1, "decoder_attn2_%d_c", (i + 1));
		//save_json("../../../data/compare_data", &idx_d2_attn1, decoder_attn2);

		decoder_ffn = Add_33(decoder_attn2, FFN(decoder_attn2));
		Norm_3D(decoder_ffn);
		//sprintf(idx_d_ffn, "decoder_ffn_%d_c", (i + 1));
		//save_json("../../../data/compare_data", &idx_d_ffn, decoder_ffn);

		transformer_output = decoder_ffn;
	}
	Norm_3D(transformer_output);
	//save_json("../../../data/compare_data", "decoder_output_c", transformer_output);

	free_arr(src); free_arr(tgt);
	free_arr(src_mask); free_arr(tgt_mask); free_arr(memory_mask);
	free_arr(src_key_padding_mask); free_arr(tgt_key_padding_mask); free_arr(memory_key_padding_mask);

	free_arr(memory);
	free_arr(decoder_ffn);
	free_arr(decoder_attn2);
	free_arr(decoder_attn1);
	free_arr(encoder_attn);

}


void basic_transformer(void) {
	struct para_config init_config; init_config.para_version = 1;
	struct arr* src = read_json("../../../data/use_data", "src");
	struct arr* tgt = read_json("../../../data/use_data", "tgt");

	// encoder
	struct arr* memory = NULL;
	struct arr* encoder_attn = NULL;
	struct arr* encoder_ffn = NULL;

	memory = src;
	for (int i = 0; i < DEFAULT_NUM_ENCODER_LAYERS; i++) {
		printf("Encoder Layer [%d] ===========\n", i + 1);

		char idx_e_attn[256]; char idx_e_ffn[256];

		encoder_attn = Add_33(memory, attention(memory, memory, memory));
		Norm_3D(encoder_attn);
		sprintf(idx_e_attn, "encoder_attn_%d_c", (i + 1));
		save_json("../../../data/compare_data", &idx_e_attn, encoder_attn);

		encoder_ffn = Add_33(encoder_attn, FFN(encoder_attn));
		Norm_3D(encoder_ffn);
		sprintf(idx_e_ffn, "encoder_ffn_%d_c", (i + 1));
		save_json("../../../data/compare_data", &idx_e_ffn, encoder_ffn);

		memory = encoder_ffn;
	}
	Norm_3D(memory);
	save_json("../../../data/compare_data", "encoder_output_c", memory);

	// decoder
	struct arr* transformer_output = tgt;
	struct arr* decoder_attn1 = NULL;
	struct arr* decoder_attn2 = NULL;
	struct arr* decoder_ffn = NULL;

	for (int i = 0; i < DEFAULT_NUM_DECODER_LAYERS; i++) {
		printf("Decoder Layer [%d] ===========\n", i + 1);

		char idx_d_attn1[256]; char idx_d2_attn1[256]; char idx_d_ffn[256];

		decoder_attn1 = Add_33(transformer_output, attention(transformer_output, transformer_output, transformer_output));
		Norm_3D(decoder_attn1);
		sprintf(idx_d_attn1, "decoder_attn1_%d_c", (i + 1));
		save_json("../../../data/compare_data", &idx_d_attn1, decoder_attn1);

		decoder_attn2 = Add_33(decoder_attn1, attention(decoder_attn1, memory, memory));
		Norm_3D(decoder_attn2);
		sprintf(idx_d2_attn1, "decoder_attn2_%d_c", (i + 1));
		save_json("../../../data/compare_data", &idx_d2_attn1, decoder_attn2);

		decoder_ffn = Add_33(decoder_attn2, FFN(decoder_attn2));
		Norm_3D(decoder_ffn);
		sprintf(idx_d_ffn, "decoder_ffn_%d_c", (i + 1));
		save_json("../../../data/compare_data", &idx_d_ffn, decoder_ffn);

		transformer_output = decoder_ffn;
	}
	Norm_3D(transformer_output);
	save_json("../../../data/compare_data", "decoder_output_c", transformer_output);

	free_arr(src); free_arr(tgt);
	free_arr(encoder_attn);
	free_arr(encoder_ffn);
	free_arr(decoder_attn1);
	free_arr(decoder_attn2);
	free_arr(decoder_ffn);

}

void test_fun(void) {
	struct para_config init_config; init_config.para_version = 1;
	struct arr* t1 = init_arr(4, (int[]) { 2, 4, 3, 2 }, init_config, "test1");
	print_full_arr(t1);

	struct arr* res = permute_2013(t1);
	print_full_arr(res);

	free_arr(t1);
	free_arr(res);
}

struct arr* FFN(struct arr* arr1) {
	printf("[FFN] start ~ ");

	struct arr* res = NULL;

	res = Linear(arr1, 0, 0);
	//save_json("../../../data/compare_data", "_ff_block_linear1_c", res);

	ReLU(res);
	//save_json("../../../data/compare_data", "_ff_block_activation_c", res);


	Dropout(res, DEFAULT_DROPOUT);
	res = Linear(res, 0, 0);
	//save_json("../../../data/compare_data", "_ff_block_linear2_c", res);


	printf("[FFN] end\n");
	return res;
}

void Dropout(struct arr* arr1, DType p) {
	int i = 0;
}

void ReLU(struct arr* arr1) {
	if (arr1->dim_size == 3) {
		for (int i = 0; i < arr1->dim[0]; i++) {
			for (int j = 0; j < arr1->dim[1]; j++) {
				for (int k = 0; k < arr1->dim[2]; k++) {
					if (arr1->arr3D[i][j][k] < 0) {
						arr1->arr3D[i][j][k] = 0;
					}
				}
			}
		}
	}
}

struct arr* attention(struct arr* for_q, struct arr* for_k, struct arr* for_v) {
	printf("[attention] start ~ ");
	int for_reshape = for_q->dim[0];

	char in_weight_name[256];
	sprintf(in_weight_name, "in_proj_weight_%d", idx_in_weight);
	before_make_qkv(read_json("../../../data/use_data", in_weight_name), 1);

	struct arr* query = make_qkv(MatMul(for_q, q_w, 1, 1), (int[]) { for_q->dim[1], DEFAULT_NHEAD, for_q->dim[0], D_K });
	struct arr* key = make_qkv(MatMul(for_k, k_w, 1, 1), (int[]) { for_k->dim[1], DEFAULT_NHEAD, for_k->dim[0], D_K });
	struct arr* value = make_qkv(MatMul(for_v, v_w, 1, 1), (int[]) { for_v->dim[1], DEFAULT_NHEAD, for_v->dim[0], D_V });
	query->arr_name = "query";
	key->arr_name = "key";
	value->arr_name = "value";
	//print_arr(query); //save_json("../../../data/compare_data", "q1", query);
	//print_arr(key); //save_json("../../../data/compare_data", "k1", key);
	//print_arr(value); //save_json("../../../data/compare_data", "v1", value);

	struct arr* attn_weight = MatMul(query, key, 1, 1 / (sqrt(D_K)));
	Softmax(attn_weight);
	attn_weight = MatMul(attn_weight, value, 0, 1);
	struct arr* attn_output = permute_2013(attn_weight);
	attn_weight->arr_name = "attn_weight";
	attn_output->arr_name = "attn_output";
	//print_arr(attn_weight);save_json("../../../data/compare_data", "attn_output", attn_weight);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_output_reshape", attn_output);

	attn_output = Linear(attn_output, 1, 1);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_output_linear", attn_output);

	struct arr* attn_result = view_2to3(attn_output, for_reshape);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_result", attn_result);

	free_arr(query);
	free_arr(key);
	free_arr(value);
	free_arr(attn_weight);
	free_arr(attn_output);

	idx_in_weight++;

	printf("[attention] end\n");
	return attn_result;
}

struct arr* attention_masked(struct arr* for_q, struct arr* for_k, struct arr* for_v, struct arr* attn_mask, struct arr* key_padding_mask) {

	printf("[attention] start ~ ");
	int for_reshape = for_q->dim[0];
	struct para_config init_config; init_config.para_version = 0;

	char in_weight_name[256];
	sprintf(in_weight_name, "in_proj_weight_%d", idx_in_weight);
	before_make_qkv(read_json("../../../data/use_data", in_weight_name), 1);


	//char attn_mask_before_n[256];
	//sprintf(attn_mask_before_n, "attn_mask_before_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", attn_mask_before_n,attn_mask);

	//attn_mask.unsqueeze(0) ->(1,10,10)
	struct arr* attn_mask_unsqueeze_0 = init_arr(3, (int[]) { 1, attn_mask->dim[0], attn_mask->dim[1] }, init_config, "attn_mask_unsqueze_0");
	for (int i = 0; i < attn_mask_unsqueeze_0->dim[0]; i++) {
		for (int j = 0; j < attn_mask_unsqueeze_0->dim[1]; j++) {
			for (int k = 0; k < attn_mask_unsqueeze_0->dim[2]; k++) {
				attn_mask_unsqueeze_0->arr3D[i][j][k] = attn_mask->arr2D[j][k];
			}
		}
	}

	//char attn_mask_unsqueeze_n[256];
	//sprintf(attn_mask_unsqueeze_n, "attn_mask_unsqueeze(0)_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", attn_mask_unsqueeze_n, attn_mask_unsqueeze_0);

	//key_padding_mask (32,10) -> (256,1,10)
	/*
	key_padding_mask = (
			key_padding_mask.view(bsz, 1, 1, src_len)
			.expand(-1, num_heads, -1, -1)
			.reshape(bsz * num_heads, 1, src_len)
		)
	*/
	//char key_padding_mask_before_n[256];
	//sprintf(key_padding_mask_before_n, "key_padding_mask_before_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", key_padding_mask_before_n, key_padding_mask);

	struct arr* key_padding_mask_reshape = init_arr(3, (int[]) { key_padding_mask->dim[0] * DEFAULT_NHEAD, 1, key_padding_mask->dim[1] }, init_config, "key_padding_mask_reshape");
	for (int i = 0; i < key_padding_mask_reshape->dim[0]; i++) {
		for (int j = 0; j < key_padding_mask_reshape->dim[1]; j++) {
			for (int k = 0; k < key_padding_mask_reshape->dim[2]; k++) {
				key_padding_mask_reshape->arr3D[i][j][k] = key_padding_mask->arr2D[i / DEFAULT_NHEAD][k];
			}
		}
	}

	//char key_padding_mask_after_n[256];
	//sprintf(key_padding_mask_after_n, "key_padding_mask_after_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", key_padding_mask_after_n, key_padding_mask_reshape);


	//attn_mask(1,10,10) + key_padding_mask(256,1,10) = (256,10,10)	
	struct arr* attn_mask_res = init_arr(3, (int[]) { key_padding_mask_reshape->dim[0], attn_mask_unsqueeze_0->dim[1], attn_mask_unsqueeze_0->dim[2] }, init_config, "attn_mask_res");
	for (int i = 0; i < attn_mask_res->dim[0]; i++) {
		for (int j = 0; j < attn_mask_res->dim[1]; j++) {
			for (int k = 0; k < attn_mask_res->dim[2]; k++) {
				attn_mask_res->arr3D[i][j][k] = attn_mask_unsqueeze_0->arr3D[0][j][k] + key_padding_mask_reshape->arr3D[i][0][k];
			}
		}
	}

	//char key_padding_mask_final_n[256];
	//sprintf(key_padding_mask_final_n, "key_padding_mask_final_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", key_padding_mask_final_n, attn_mask_res);

	struct arr* query = make_qkv(MatMul(for_q, q_w, 1, 1), (int[]) { for_q->dim[1], DEFAULT_NHEAD, for_q->dim[0], D_K });
	struct arr* key = make_qkv(MatMul(for_k, k_w, 1, 1), (int[]) { for_k->dim[1], DEFAULT_NHEAD, for_k->dim[0], D_K });
	struct arr* value = make_qkv(MatMul(for_v, v_w, 1, 1), (int[]) { for_v->dim[1], DEFAULT_NHEAD, for_v->dim[0], D_V });
	query->arr_name = "query";
	key->arr_name = "key";
	value->arr_name = "value";
	//print_arr(query); //save_json("../../../data/compare_data", "q1", query);
	//print_arr(key); //save_json("../../../data/compare_data", "k1", key);
	//print_arr(value); //save_json("../../../data/compare_data", "v1", value);

	//attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)
	int idx_1, idx_2, idx_3; idx_1 = idx_2 = idx_3 = 0;
	struct arr* attn_mask_use = init_arr(4, (int[]) { query->dim[0], query->dim[1], check_total_dimension(attn_mask_res)/(query->dim[0] * query->dim[1] * key->dim[2]), key->dim[2] }, init_config, "use_attn_mask");
	for (int i = 0; i < attn_mask_use->dim[0]; i++) {
		for (int j = 0; j < attn_mask_use->dim[1]; j++) {
			for (int k = 0; k < attn_mask_use->dim[2]; k++) {
				for (int m = 0; m < attn_mask_use->dim[3]; m++) {
					if (idx_1 < attn_mask_res->dim[0] && idx_2 < attn_mask_res->dim[1] && idx_3 < attn_mask_res->dim[2]) {
						attn_mask_use->arr4D[i][j][k][m] = attn_mask_res->arr3D[idx_1][idx_2][idx_3];
					}
					else {
						printf("Index error: idx_1=%d, idx_2=%d, idx_3=%d out of bounds\n", idx_1, idx_2, idx_3);
						exit(1);
					}

					idx_3++;
					if (idx_3 >= attn_mask_res->dim[2]) {
						idx_3 = 0;
						idx_2++;
					}
					if (idx_2 >= attn_mask_res->dim[1]) {
						idx_2 = 0;
						idx_1++;
					}
					
				}
			}
		}
	}
	//char attn_mask_view_n[256];
	//sprintf(attn_mask_view_n, "attn_mask_view_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", attn_mask_view_n, attn_mask_use);

	struct arr* attn_weight = MatMul(query, key, 1, 1 / (sqrt(D_K)));
	attn_weight = Add_44(attn_mask_use, attn_weight);

	Softmax(attn_weight);
	attn_weight = MatMul(attn_weight, value, 0, 1);

	//char attn_output_n[256];
	//sprintf(attn_output_n, "attn_output_%d_c", idx_yewon_check);
	//save_json("../../../data/check_mask", attn_output_n, attn_weight);
	//idx_yewon_check++;


	struct arr* attn_output = permute_2013(attn_weight);
	attn_weight->arr_name = "attn_weight";
	attn_output->arr_name = "attn_output";
	//print_arr(attn_weight);save_json("../../../data/compare_data", "attn_output", attn_weight);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_output_reshape", attn_output);

	attn_output = Linear(attn_output, 1, 1);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_output_linear", attn_output);

	struct arr* attn_result = view_2to3(attn_output, for_reshape);
	//print_arr(attn_output);save_json("../../../data/compare_data", "attn_result", attn_result);

	free_arr(query);
	free_arr(key);
	free_arr(value);
	free_arr(attn_weight);
	free_arr(attn_output);
	free_arr(attn_mask_unsqueeze_0);
	free_arr(key_padding_mask_reshape);
	free_arr(attn_mask_res);
	idx_in_weight++;

	printf("[attention] end\n");
	return attn_result;
}

int check_total_dimension(struct arr* check_arr) {
	int res = 1;
	for (int i = 0; i < check_arr->dim_size; i++) {
		res *= check_arr->dim[i];
	}
	return res;
}
struct arr* permute_2013(struct arr* before_arr) {
	struct para_config init_config; init_config.para_version = 0;
	struct arr* res_arr = init_arr(2, (int[]) { before_arr->dim[0] * before_arr->dim[2], before_arr->dim[1] * before_arr->dim[3] }, init_config, NULL);

	int idx_x, idx_y; idx_x = idx_y = 0;
	for (int l = 0; l < before_arr->dim[0]; l++) {
		for (int i = 0; i < before_arr->dim[1]; i++) {
			for (int j = 0; j < before_arr->dim[2]; j++) {
				for (int k = 0; k < before_arr->dim[3]; k++) {
					idx_x = (l)+(j * before_arr->dim[0]);
					idx_y = (i * before_arr->dim[3]) + k;

					res_arr->arr2D[idx_x][idx_y] = before_arr->arr4D[l][i][j][k];
					//printf("> [%d][%d][%d][%d] ->[%d][%d]\n", l, i, j, k, idx_x, idx_y);
				}
			}
		}
	}
	return res_arr;
}

void Softmax(struct arr* before_arr) {
	if (before_arr->dim_size == 4) {
		for (int l = 0; l < before_arr->dim[0]; l++) {
			for (int i = 0; i < before_arr->dim[1]; i++) {
				for (int j = 0; j < before_arr->dim[2]; j++) {

					DType sum = 0.0;
					for (int k = 0; k < before_arr->dim[3]; k++) {
						sum += exp(before_arr->arr4D[l][i][j][k]);
					}
					for (int k = 0; k < before_arr->dim[3]; k++) {
						before_arr->arr4D[l][i][j][k] = exp(before_arr->arr4D[l][i][j][k]) / sum;
					}
				}
			}
		}
	}
}

struct arr* make_qkv(struct arr* before_qkv, int* dim_size_list) {
	struct para_config init_config; init_config.para_version = 0;
	struct arr* qkv_res = init_arr(4, dim_size_list, init_config, NULL);

	for (int l = 0; l < dim_size_list[0]; l++) {
		for (int i = 0; i < dim_size_list[1]; i++) {
			for (int j = 0; j < dim_size_list[2]; j++) {
				for (int k = 0; k < dim_size_list[3]; k++) {
					qkv_res->arr4D[l][i][j][k] = before_qkv->arr3D[j][l][k + (i * (before_qkv->dim[2] / DEFAULT_NHEAD))];
					//printf("qkv[%d][%d][%d][%d] = before[%d][%d][%d+(%d*%d)]  (%f)\n", l, i, j, k, j, l, k , i, (before_qkv->dim[2] / DEFAULT_NHEAD), qkv_res->arr4D[l][i][j][k]);
				}
			}
		}
	}
	return qkv_res;
}

void before_make_qkv(struct arr* for_in_weight, int qkv) {
	struct para_config init_config; init_config.para_version = 0; init_config.a = 0;

	q_w = init_arr(2, (int[]) { DEFAULT_D_MODEL / DEFAULT_NHEAD * DEFAULT_NHEAD, DEFAULT_D_MODEL }, init_config, "q");
	k_w = init_arr(2, (int[]) { DEFAULT_D_MODEL / DEFAULT_NHEAD * DEFAULT_NHEAD, DEFAULT_D_MODEL }, init_config, "k");
	v_w = init_arr(2, (int[]) { DEFAULT_D_MODEL / DEFAULT_NHEAD * DEFAULT_NHEAD, DEFAULT_D_MODEL }, init_config, "v");

	int ii = 0; int jj = 0;
	for (int i = 0; i < DEFAULT_D_MODEL; i++) {
		for (int j = 0; j < DEFAULT_D_MODEL; j++) {
			q_w->arr2D[i][j] = for_in_weight->arr2D[ii][jj];
			k_w->arr2D[i][j] = for_in_weight->arr2D[ii + (DEFAULT_D_MODEL)][jj];
			v_w->arr2D[i][j] = for_in_weight->arr2D[ii + (DEFAULT_D_MODEL * 2)][jj];

			jj++;
			if (jj == for_in_weight->dim[1]) {
				jj = 0;
				ii++;
			}
		}
	}
}

struct arr* Linear(struct arr* input_arr, int weight_use, int bias_use) {

	struct arr* weight_arr = NULL;
	struct arr* bias_arr = NULL;
	struct arr* res_arr = NULL;

	if (weight_use == 0) {
		char linear_weight_name[256];
		sprintf(linear_weight_name, "linear_weight_%d", idx_linear_weight);
		weight_arr = read_json("../../../data/use_data", linear_weight_name);
		//printf("(use)lienar_weight_%d ", idx_linear_weight);
	}
	else if (weight_use == 1) {
		char out_weight_name[256];
		sprintf(out_weight_name, "out_proj_weight_%d", idx_out_weight);
		weight_arr = read_json("../../../data/use_data", out_weight_name);
		//printf("(use)out_proj_weight_%d ", idx_out_weight);
	}

	if (bias_use == 0) {
		char linear_bias_name[256];
		sprintf(linear_bias_name, "linear_bias_%d", idx_linear_bias);
		bias_arr = read_json("../../../data/use_data", linear_bias_name);
		//printf("  linear_bias_%d\n", idx_linear_bias);
	}
	else if (bias_use == 1) {
		char out_bias_name[256];
		sprintf(out_bias_name, "out_proj_bias_%d", idx_out_bias);
		bias_arr = read_json("../../../data/use_data", out_bias_name);
		//printf("  out_proj_bias_%d\n", idx_out_bias);
	}

	res_arr = MatMul(input_arr, weight_arr, 1, 1);

	if (bias_arr->dim_size == 1) {
		if (res_arr->dim_size == 2) {
			for (int i = 0; i < res_arr->dim[0]; i++) {
				for (int j = 0; j < res_arr->dim[1]; j++) {
					res_arr->arr2D[i][j] += bias_arr->arr1D[j];
				}
			}
		}
		else if (res_arr->dim_size == 3) {
			for (int i = 0; i < res_arr->dim[0]; i++) {
				for (int j = 0; j < res_arr->dim[1]; j++) {
					for (int k = 0; k < res_arr->dim[2]; k++) {
						res_arr->arr3D[i][j][k] += bias_arr->arr1D[k];
					}
				}
			}
		}
	}

	free_arr(weight_arr);
	free_arr(bias_arr);


	if (weight_use == 0) {
		idx_linear_weight++;
	}
	else if (weight_use == 1) {
		idx_out_weight++;
	}

	if (bias_use == 0) {
		idx_linear_bias++;
	}
	else if (bias_use == 1) {
		idx_out_bias++;
	}
	return res_arr;
}

struct arr* MatMul(struct arr* arr1, struct arr* arr2, int doit_transpose, DType scale_factor) {
	int matmul_case = 0;
	struct arr* matmul_res = NULL;
	struct para_config init_config; init_config.para_version = 0; init_config.a = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////
	if ((arr1->dim_size == 2) && (arr2->dim_size == 2)) {
		if (doit_transpose == 0) {
			matmul_res = init_arr(2, (int[]) { arr1->dim[0], arr2->dim[1] }, init_config, NULL);
			for (int i = 0; i < arr1->dim[0]; i++) {
				for (int j = 0; j < arr2->dim[1]; j++) {

					for (int k = 0; k < arr1->dim[1]; k++) {
						matmul_res->arr2D[i][j] += arr1->arr2D[i][k] * arr2->arr2D[k][j];
					}
					matmul_res->arr2D[i][j] *= (scale_factor);
				}
			}
		}
		else {
			matmul_res = init_arr(2, (int[]) { arr1->dim[0], arr2->dim[0] }, init_config, NULL);

			for (int i = 0; i < arr1->dim[0]; i++) {
				for (int j = 0; j < arr2->dim[0]; j++) {

					for (int k = 0; k < arr1->dim[1]; k++) {
						matmul_res->arr2D[i][j] += arr1->arr2D[i][k] * arr2->arr2D[j][k];
					}
					matmul_res->arr2D[i][j] *= (scale_factor);
				}
			}
		}
	}

	else if ((arr1->dim_size == 3) && (arr2->dim_size == 2)) {
		if (doit_transpose == 0) {
			matmul_res = init_arr(3, (int[]) { arr1->dim[0], arr1->dim[1], arr2->dim[1] }, init_config, NULL);
			for (int l = 0; l < arr1->dim[0]; l++) {
				for (int i = 0; i < arr1->dim[1]; i++) {
					for (int j = 0; j < arr2->dim[1]; j++) {

						for (int k = 0; k < arr1->dim[2]; k++) {
							matmul_res->arr3D[l][i][j] += arr1->arr3D[l][i][k] * arr2->arr2D[k][j];
						}
						matmul_res->arr3D[l][i][j] *= (scale_factor);
					}
				}
			}
		}
		else {
			matmul_res = init_arr(3, (int[]) { arr1->dim[0], arr1->dim[1], arr2->dim[0] }, init_config, NULL);
			for (int l = 0; l < arr1->dim[0]; l++) {
				for (int i = 0; i < arr1->dim[1]; i++) {
					for (int j = 0; j < arr2->dim[0]; j++) {

						for (int k = 0; k < arr1->dim[2]; k++) {
							matmul_res->arr3D[l][i][j] += arr1->arr3D[l][i][k] * arr2->arr2D[j][k];
						}
						matmul_res->arr3D[l][i][j] *= (scale_factor);
					}
				}
			}
		}
	}
	else if ((arr1->dim_size == 4) && (arr2->dim_size == 4)) {
		if (doit_transpose == 0) {
			if (arr1->dim[3] != arr2->dim[2]) {
				printf("error) arr1.dim[2](%d) != arr2.dim[3](%d)\n", arr1->dim[2], arr2->dim[3]);
				return NULL;
			}
			matmul_res = init_arr(4, (int[]) { arr1->dim[0], arr1->dim[1], arr1->dim[2], arr2->dim[3] }, init_config, NULL);
			for (int l = 0; l < arr1->dim[0]; l++) {
				for (int i = 0; i < arr1->dim[1]; i++) {
					for (int j = 0; j < arr1->dim[2]; j++) {
						for (int k = 0; k < arr2->dim[3]; k++) {

							for (int m = 0; m < arr1->dim[3]; m++) {
								matmul_res->arr4D[l][i][j][k] += arr1->arr4D[l][i][j][m] * arr2->arr4D[l][i][m][k];
							}
							matmul_res->arr4D[l][i][j][k] *= (scale_factor);
						}
					}
				}
			}
		}
		else if (doit_transpose == 1) {
			if (arr1->dim[3] != arr2->dim[3]) {
				printf("error) arr1.dim[2](%d) != arr2.dim[3](%d)\n", arr1->dim[2], arr2->dim[3]);
				return NULL;
			}
			matmul_res = init_arr(4, (int[]) { arr1->dim[0], arr1->dim[1], arr1->dim[2], arr2->dim[2] }, init_config, NULL);
			for (int l = 0; l < arr1->dim[0]; l++) {
				for (int i = 0; i < arr1->dim[1]; i++) {
					for (int j = 0; j < arr1->dim[2]; j++) {
						for (int k = 0; k < arr2->dim[2]; k++) {

							for (int m = 0; m < arr1->dim[3]; m++) {
								matmul_res->arr4D[l][i][j][k] += arr1->arr4D[l][i][j][m] * arr2->arr4D[l][i][k][m];
							}
							matmul_res->arr4D[l][i][j][k] *= (scale_factor);
						}
					}
				}
			}
		}

	}
	////////////////////////////////////////////////////////////////////////////////////////////////

	return matmul_res;
}

struct arr* view_2to3(struct arr* before_arr, int ch_idx) {

	struct para_config init_config; init_config.para_version = 0;
	struct arr* res = init_arr(3, (int[]) { ch_idx, before_arr->dim[0] / ch_idx, before_arr->dim[1] }, init_config, NULL);

	for (int i = 0; i < res->dim[0]; i++) {
		for (int j = 0; j < res->dim[1]; j++) {
			for (int k = 0; k < res->dim[2]; k++) {
				res->arr3D[i][j][k] = before_arr->arr2D[j + (i * before_arr->dim[0] / ch_idx)][k];
			}
		}
	}
	return res;
}

struct arr* Add_33(struct arr* arr1, struct arr* arr2) {
	if (arr1->dim_size != arr2->dim_size) {
		printf("error) arr1.dim_size != arr2.dim_size\n");
		return NULL;
	}
	int check = 0;
	for (int i = 0; i < arr1->dim_size; i++) {
		if (arr1->dim[i] != arr2->dim[i]) {
			printf("error) arr1.dim[%d] != arr2.dim[%d]\n");
			check = 1;
		}
	}
	if (check == 1) {
		printf("error) arr1.dim_size != arr2.dim_size\n");
		return NULL;
	}

	struct para_config init_config; init_config.para_version = 0;
	struct arr* res = init_arr(arr1->dim_size, arr1->dim, init_config, NULL);

	for (int i = 0; i < res->dim[0]; i++) {
		for (int j = 0; j < res->dim[1]; j++) {
			for (int k = 0; k < res->dim[2]; k++) {
				res->arr3D[i][j][k] = arr1->arr3D[i][j][k] + arr2->arr3D[i][j][k];
			}
		}
	}

	return res;
}

struct arr* Add_44(struct arr* arr1, struct arr* arr2) {
	if (arr1->dim_size != arr2->dim_size) {
		printf("error) arr1.dim_size != arr2.dim_size\n");
		return NULL;
	}
	int check = 0;
	for (int i = 0; i < arr1->dim_size; i++) {
		if (arr1->dim[i] != arr2->dim[i]) {
			printf("error) arr1.dim[%d] != arr2.dim[%d]\n");
			check = 1;
		}
	}
	if (check == 1) {
		printf("error) arr1.dim_size != arr2.dim_size\n");
		return NULL;
	}

	struct para_config init_config; init_config.para_version = 0;
	struct arr* res = init_arr(arr1->dim_size, arr1->dim, init_config, NULL);

	for (int i = 0; i < res->dim[0]; i++) {
		for (int j = 0; j < res->dim[1]; j++) {
			for (int k = 0; k < res->dim[2]; k++) {
				for (int m = 0; m < res->dim[3]; m++) {
					res->arr4D[i][j][k][m] = arr1->arr4D[i][j][k] [m] + arr2->arr4D[i][j][k][m];
				}
			}
		}
	}

	return res;
}

void Norm_3D(struct arr* n_arr) {
	for (int i = 0; i < n_arr->dim[0]; i++) {
		for (int j = 0; j < n_arr->dim[1]; j++) {
			DType var = 0.0;
			DType avg = 0.0;
			DType sum = 0.0;

			for (int k = 0; k < n_arr->dim[2]; k++) {
				sum += n_arr->arr3D[i][j][k];
			}
			avg = sum / n_arr->dim[2];

			sum = 0.0;
			for (int k = 0; k < n_arr->dim[2]; k++) {
				sum += (n_arr->arr3D[i][j][k] - avg) * (n_arr->arr3D[i][j][k] - avg);
			}
			var = sum / n_arr->dim[2];

			for (int k = 0; k < n_arr->dim[2]; k++) {
				n_arr->arr3D[i][j][k] = (n_arr->arr3D[i][j][k] - avg) / (sqrt(var + DEFAULT_LAYER_NORM_EPS));
			}
		}
	}

}
