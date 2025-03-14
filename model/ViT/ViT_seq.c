#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "Network.h"
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6
#define M_PI 3.14159265358979323846
Weight *networks;

/* Conv2d : stride 및 kernel size가 patch_size와 같으므로 패치 임베딩 역할 */
// 입력 : input, shape = (in_chans, img_size, img_size)
// 출력 : output, shape = (emb_dim, img_size/patch_size, img_size/patch_size)
void Conv2d(float *input, float *output){ 
    int output_size = img_size / patch_size;

    for(int oc=0; oc<embed_dim; ++oc){
        for(int oh=0; oh<output_size; ++oh){
            for(int ow=0; ow<output_size; ++ow){
                float sum = networks[2].data[oc];

                for(int ic=0; ic<in_chans; ++ic){
                    for(int kh=0; kh<patch_size; ++kh){
                        for(int kw=0; kw<patch_size; ++kw){
                            int ih = oh * patch_size + kh;
                            int iw = ow * patch_size + kw;
                            int input_idx = (ic*img_size + ih) * img_size + iw;
                            int kernel_idx = ((oc*in_chans + ic) * patch_size + kh) * patch_size + kw;

                            sum += input[input_idx] * networks[1].data[kernel_idx];
                        }
                    }
                }

                output[(oc * output_size + oh) * output_size + ow] = sum;
                //printf("%.4f ",sum);
            }
        }
        //printf("\n");
    }
}

/* flatten_transpose : Conv2d의 출력을 flatten하고 transpose하는 과정 */
// 출력: output, shape: (num_patches, embed_dim)
// 여기서 output_size = img_size/patch_size, num_patches = output_size * output_size
void flatten_transpose(float *input, float *output){
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 각 공간 위치(oh, ow)를 하나의 패치로 취급하여 patch index 계산
    for(int oh = 0; oh < output_size; oh++){
        for(int ow = 0; ow<output_size; ow++){
            int patch_idx = oh * output_size + ow;
            for(int oc = 0; oc<embed_dim; oc++){
                // 기존 입력은 (oc, oh, ow)
                int idx_input = (oc*output_size + oh) * output_size + ow;
                // 원하는 출력은 (patch_idx, oc)
                int idx_output = patch_idx * embed_dim + oc;
                output[idx_output] = input[idx_input];
                //printf("%f ",output[idx_output]);
            }
        }
    }
}

void class_token(float *patch_tokens, float *final_tokens) {
    // 이미지의 패치 수 계산: output_size = img_size / patch_size, num_patches = output_size^2
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    
    // 1. 첫 번째 토큰에 class token 복사 (networks[0].data에 저장됨, embed_dim 길이)
    for (int j = 0; j < embed_dim; j++){
        final_tokens[j] = networks[0].data[j];
    }
    
    // 2. 이후 patch_tokens를 이어붙임
    // final_tokens의 인덱스 embed_dim부터, patch_tokens 전체(embed_dim * num_patches) 복사
    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);

    int total_tokens = num_patches + 1; // class token + patch tokens
    for (int i = 0; i < total_tokens * embed_dim; i++){
        //("%f ", final_tokens[i]);
    }
    //printf("\n");
}

void pos_emb(float *input, float *output) {
    // output_size: 한 변의 패치 수, num_patches: 전체 패치 수, total_tokens: class token + patch tokens
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * embed_dim;
    
    // networks[3].data에 positional embedding이 저장되어 있다고 가정 (flatten된 배열, 길이 = total_elements)
    // 각 원소별로 input과 positional embedding을 더함
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + networks[3].data[i];
    }
}

void layer_norm(float *input, float *output, float *weight, float *bias){
    int token = ((img_size/patch_size) * (img_size/patch_size)) + 1;
    
    for(int t=0; t<token; t++){
        float sum = 0.0, sum_sq = 0.0;
        for(int i=0; i<embed_dim; i++){
            float val = input[t*embed_dim + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / embed_dim;
        float var = sum_sq / embed_dim - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for(int i=0; i<embed_dim; i++){
            int idx = t * embed_dim + i;
            output[idx] = (input[idx] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

void multihead_attn(float *input, float *output, float *in_weight, float *in_bias, float *out_weight, float *out_bias){
    int head_dim = embed_dim / num_heads, total_dim = embed_dim, tokens = ((img_size/patch_size) * (img_size/patch_size)) + 1;

    /*Allocate Q, K, V : tokens * dim*/
    float *Q = (float*)malloc(sizeof(float)* tokens * total_dim);
    float *K = (float*)malloc(sizeof(float)* tokens * total_dim);
    float *V = (float*)malloc(sizeof(float)* tokens * total_dim);

    /*Q, K, V 구하기*/
    for(int t=0; t<tokens; t++){
        float sum_q, sum_k, sum_v;
        for(int i=0; i<total_dim; i++){
            sum_q = in_bias[i], sum_k = in_bias[total_dim + i], sum_v = in_bias[total_dim*2 + i];
            for(int j=0; j<total_dim; j++){
                sum_q += input[t*total_dim + j] * in_weight[i*total_dim + j];
                sum_k += input[t*total_dim + j] * in_weight[(i+total_dim)*total_dim + j];
                sum_v += input[t*total_dim + j] * in_weight[(i+total_dim*2)*total_dim + j];
            }
            Q[t * total_dim + i] = sum_q;
            K[t * total_dim + i] = sum_k;
            V[t * total_dim + i] = sum_v;
        }
    }

    /*Attn 결과를 저장할 버퍼*/
    float *attn_output = (float*)malloc(sizeof(float)* tokens * total_dim);
    for (int i = 0; i < tokens * total_dim; i++) attn_output[i] = 0.0f;

    /*head별로 attn 수행*/
    for(int h=0; h<num_heads; h++){
        int head_offset = h * head_dim;

        // attn_score 저장 공간
        float *scores = (float*)malloc(sizeof(float)* tokens * tokens);

        // 각 head에 대해 scaled-dot attn
        for(int i=0; i<tokens; i++){
            for(int j=0; j<tokens; j++){
                float score = 0.0f;
                for(int d=0; d<head_dim; d++){
                    float q = Q[i * total_dim + head_offset + d];
                    float k = K[j * total_dim + head_offset + d];
                    score += q * k;                    
                }
                scores[i * tokens + j] = score / sqrtf((float)head_dim);
            }
        }

        // softmax 적용
        for (int i = 0; i < tokens; i++){
            float max_val = scores[i * tokens];
            for (int j = 1; j < tokens; j++){
                if (scores[i * tokens + j] > max_val) max_val = scores[i * tokens + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < tokens; j++){
                scores[i * tokens + j] = expf(scores[i * tokens + j] - max_val);
                sum_exp += scores[i * tokens + j];
            }
            for (int j = 0; j < tokens; j++){
                scores[i * tokens + j] /= sum_exp;
            }
        }
        // scores와 V를 곱해 head output 계산
        float *head_out = (float*)malloc(sizeof(float) * tokens * head_dim);
        for (int i = 0; i < tokens; i++){
            for (int d = 0; d < head_dim; d++){
                float sum = 0.0f;
                for (int j = 0; j < tokens; j++){
                    sum += scores[i * tokens + j] * V[j * total_dim + head_offset + d];
                }
                head_out[i * head_dim + d] = sum;
            }
        }
        // head_out를 attn_output의 해당 부분에 복사
        for (int i = 0; i < tokens; i++){
            for (int d = 0; d < head_dim; d++){
                attn_output[i * total_dim + head_offset + d] = head_out[i * head_dim + d];
            }
        }
        free(scores);
        free(head_out);
    }

    free(Q); free(K); free(V);

    // 최종 선형 프로젝션
    for (int t = 0; t < tokens; t++){
        for (int i = 0; i < total_dim; i++){
            float sum = out_bias[i];
            for (int j = 0; j < total_dim; j++){
                sum += attn_output[t * total_dim + j] * out_weight[i * total_dim + j];
            }
            output[t * total_dim + i] = sum;
        }
    }
    free(attn_output);    
}

// GELU 활성화 함수 (근사식)
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
}
void gelu_activation(float *input, float *output, int size) {
    for (int i = 0; i < size; i++){
        output[i] = gelu(input[i]);
    }
}

void linear_layer(float *input, float *output, int tokens, int in_features, int out_features, float *weight, float *bias) {
    for (int t = 0; t < tokens; t++){
        for (int o = 0; o < out_features; o++){
            float sum = bias[o];
            for (int i = 0; i < in_features; i++){
                sum += input[t * in_features + i] * weight[i * out_features + o];
            }
            output[t * out_features + o] = sum;
        }
    }
}

void mlp_block(float *input, float *output,  float *fc1_weight, float *fc1_bias, float *fc2_weight, float *fc2_bias){
    int tokens = ((img_size/patch_size) * (img_size/patch_size)) + 1, hidden_dim = ((int)(embed_dim*mlp_ratio));
    float *fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);

    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    // GELU 활성화
    for (int i = 0; i < tokens * hidden_dim; i++){
        fc1_out[i] = gelu(fc1_out[i]);
    }
    // fc2: (tokens, in_dim)
    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
    free(fc1_out);
}

void Encoder(float *input, float *output, int idx){
    int tokens = ((img_size/patch_size)*(img_size/patch_size)) + 1;
    
    if(idx < depth){
        int base = 4 + idx * 12;

        // 임시 버퍼 할당
        float *ln1_out   = (float*)malloc(sizeof(float) * tokens * embed_dim);
        float *attn_out  = (float*)malloc(sizeof(float) * tokens * embed_dim);
        float *residual  = (float*)malloc(sizeof(float) * tokens * embed_dim);
        float *ln2_out   = (float*)malloc(sizeof(float) * tokens * embed_dim);
        float *mlp_out   = (float*)malloc(sizeof(float) * tokens * embed_dim);

        // LN1 : 입력에 대해 LayerNorm 적용
        layer_norm(input, ln1_out, networks[base].data, networks[base+1].data);

        // MHA
        multihead_attn(ln1_out, attn_out, networks[base+2].data, networks[base+3].data, networks[base+4].data, networks[base+5].data);

        // Residual1
        for(int i=0; i<tokens * embed_dim; i++){
            residual[i] = input[i] + attn_out[i];
        }

        // LN2
        layer_norm(residual, ln2_out, networks[base+6].data, networks[base+7].data);

        // MLP
        mlp_block(ln2_out, mlp_out, networks[base+8].data, networks[base+9].data, networks[base+10].data, networks[base+11].data);

        // Residual2
        for (int i = 0; i < tokens * embed_dim; i++){
            output[i] = residual[i] + mlp_out[i];
        }
        free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);    
    }
    

}

////////////////////////////////////// layer별 size //////////////////////////////////////
const int size[] = {
    embed_dim * (img_size/patch_size) * (img_size/patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size)+1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size)+1) // position embedding
};

const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size)+1);

void ViT_seq(ImageData *image, Weight *network){
    networks = network;
    int tokens = ((img_size / patch_size) * (img_size / patch_size)+1);
    float *layer[4];
    float *enc_layer[12];
    float *enc_output;

    for(int i=0; i<4; i++){
        layer[i] = (float*)malloc(sizeof(float)*size[i]);
    }
    for(int i=0; i<12; i++){
        enc_layer[i] = (float*)malloc(sizeof(float)*enc_size);
    }
    enc_output = (float*)malloc(sizeof(float)*enc_size);

    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////// Model Architecture //////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    for(int i=0; i<image->n; i++){
        
        /*patch embedding*/
        Conv2d(image[i].data, layer[0]);
        
        /*flatten and transpose*/
        flatten_transpose(layer[0], layer[1]);

        /*prepend class token*/
        class_token(layer[1], layer[2]);

        /*position embedding*/
        pos_emb(layer[2], layer[3]);

        /*Encoder*/
        Encoder(layer[3], enc_layer[0], 0);
        Encoder(enc_layer[0], enc_layer[1], 1);
        Encoder(enc_layer[1], enc_layer[2], 2);
        Encoder(enc_layer[2], enc_layer[3], 3);
        Encoder(enc_layer[3], enc_layer[4], 4);
        Encoder(enc_layer[4], enc_layer[5], 5);
        Encoder(enc_layer[5], enc_layer[6], 6);
        Encoder(enc_layer[6], enc_layer[7], 7);
        Encoder(enc_layer[7], enc_layer[8], 8);
        Encoder(enc_layer[8], enc_layer[9], 9);
        Encoder(enc_layer[9], enc_layer[10], 10);
        Encoder(enc_layer[10], enc_layer[11], 11);
        layer_norm(enc_layer[11], enc_output, networks[148].data, networks[149].data);
        for(int i=0; enc_size; i++) printf("%f ", enc_output[i]);

        /* Extract Class Token (첫 번째 토큰만 사용) */
        float *cls_token = (float*)malloc(sizeof(float) * embed_dim);
        float *cls_output = (float*)malloc(sizeof(float) * num_classes);

        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        /* Classification Head (heads.head)
           networks[150].data: linear layer weight, networks[151].data: linear layer bias */
        //linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150].data, networks[151].data);

        /* 결과 출력 (classification head의 출력) */
        // for (int j = 0; j < num_classes; j++){
        //     printf("%f ", cls_output[j]);
        // }
    }
}