#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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
                printf("%.4f ",sum);
            }
        }
        printf("\n");
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
                printf("%f ",output[idx_output]);
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
        printf("%f ", final_tokens[i]);
    }
    printf("\n");
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
    
    // 덧셈 결과를 출력 (검증용)
    for (int i = 0; i < total_elements; i++){
        printf("%f ", output[i]);
    }
    printf("\n");
}

////////////////////////////////////// layer별 size //////////////////////////////////////
const int size[] = {
    embed_dim * (img_size/patch_size) * (img_size/patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size)+1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size)+1) // position embedding
};

void ViT_seq(ImageData *image, Weight *network){
    networks = network;

    float *layer[4];

    for(int i=0; i<4; i++){
        layer[i] = (float*)malloc(sizeof(float)*size[i]);
    }

    
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
    }
}