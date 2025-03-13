#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "Network.h"
#include "ViT_seq.h"

int main(){
    // 출력 리디렉션: 모든 stdout 출력이 "./Data/Model_Resultlog_C.txt" 파일로 기록됩니다.
    FILE *fp = freopen("./Log/ViT_print_C.txt", "w", stdout);
    if(fp == NULL) {
        fprintf(stderr, "stdout 리디렉션 실패\n");
        return 1;
    }
    
    ////////////////////////////////////// Input load //////////////////////////////////////
    const char *img_filename = "./Data/input.bin";
    ImageData *images = load_image_data(img_filename);
    if(images == NULL) return 1;
    
    int image_size = images->c * images->h * images->w; 

    for(int i = 0; i < images->n; i++){
        for(int j = 0; j < image_size; j++){
            printf("%.4f ", images[i].data[j]);
        }
        printf("\n");
    }

    ////////////////////////////////////// Weight load //////////////////////////////////////
    const char *weights_filename = "./Data/all_weights.bin";
    int num_weights = 0;
    Weight *weights_array = load_all_weights(weights_filename, &num_weights);
    if (weights_array == NULL) {
        return 1;
    }    
    
    ////////////////////////////////////// Model //////////////////////////////////////
    ViT_seq(images, weights_array);

    return 0;
}
