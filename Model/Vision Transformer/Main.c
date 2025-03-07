#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "Network.h"
#include "ViT_seq.h"

int main() {
    // 이미지 데이터 로드 (input.bin)
    const char *img_filename = "input.bin";
    ImageData *images = load_image_data(img_filename);
    if (images == NULL) {
        return 1;
    }
    int image_size = images->c * images->h * images->w; 

    
    // 모든 가중치 로드 (all_weights.bin)
    const char *weights_filename = "all_weights.bin";
    int num_weights = 0;
    Weight *weights_array = load_all_weights(weights_filename, &num_weights);
    if (weights_array == NULL) {
        return 1;
    }

    ViT_seq(images, weights_array);
    
    // 로드한 weight 정보 출력: 각 weight의 이름, 차원, shape, 첫 5개 값
    /*
        for (int i = 0; i < num_weights; i++) {
        printf("Weight %d: %s\n", i, weights_array[i].key);
        printf("  Dimensions: %d, Shape: [", weights_array[i].num_dims);
        for (int j = 0; j < weights_array[i].num_dims; j++) {
            printf("%d", weights_array[i].shape[j]);
            if (j < weights_array[i].num_dims - 1)
                printf(", ");
        }
        printf("]\n");
        printf("  First 5 values: ");
        int print_count = weights_array[i].num_elements < 5 ? weights_array[i].num_elements : 5;
        for (int j = 0; j < print_count; j++) {
            printf("%f ", weights_array[i].data[j]);
        }
        printf("\n");
    }
    */
    
    // 가중치 메모리 해제
    free_all_weights(weights_array, num_weights);
    
    return 0;
}
