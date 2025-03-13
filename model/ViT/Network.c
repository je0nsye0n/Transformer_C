#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 이미지 데이터 정보를 담을 구조체 정의
typedef struct {
    int n;      // 이미지 개수
    int c;      // 채널 수
    int h;      // 높이
    int w;      // 너비
    float *data; // 모든 이미지 데이터를 연속된 메모리 공간에 저장 (N x C x H x W)
} ImageData;

// Weight 정보를 담을 구조체 정의
typedef struct {
    char *key;       // 파라미터 이름 (null-terminated)
    int num_dims;    // 차원 수
    int *shape;      // 각 차원의 크기 (배열, 길이 num_dims)
    int num_elements;// 총 원소 개수
    float *data;     // float 데이터 (num_elements 크기)
} Weight;

// input.bin 파일의 헤더는 4개의 int32: (n, c, h, w)
// 이후 float32 데이터가 연속해서 저장되어 있다고 가정합니다.
ImageData *load_image_data(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        perror("파일 열기 실패");
        return NULL;
    }
    
    // 헤더 읽기: n, c, h, w
    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) {
        perror("헤더 읽기 실패");
        fclose(f);
        return NULL;
    }
    
    int n = header[0];
    int c = header[1];
    int h = header[2];
    int w = header[3];
    int image_size = c * h * w;
    int total_elements = n * image_size;
    
    // 모든 이미지 데이터를 한 번에 읽어 들일 임시 버퍼 할당
    float *all_data = (float *)malloc(total_elements * sizeof(float));
    if (all_data == NULL) {
        perror("전체 데이터 버퍼 메모리 할당 실패");
        fclose(f);
        return NULL;
    }
    if (fread(all_data, sizeof(float), total_elements, f) != total_elements) {
        perror("이미지 데이터 읽기 실패");
        free(all_data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    
    // 이미지 개수만큼의 ImageData 배열 할당
    ImageData *images = (ImageData *)malloc(n * sizeof(ImageData));
    if (images == NULL) {
        perror("ImageData 배열 메모리 할당 실패");
        free(all_data);
        return NULL;
    }
    
    // 각 이미지별로 메타정보를 설정하고, 데이터는 별도의 메모리 영역에 복사
    for (int i = 0; i < n; i++) {
        images[i].n = n;  // 전체 이미지 개수를 저장 (편의를 위해 각 구조체에 동일하게 저장)
        images[i].c = c;
        images[i].h = h;
        images[i].w = w;
        images[i].data = (float *)malloc(image_size * sizeof(float));
        if (images[i].data == NULL) {
            perror("개별 이미지 데이터 메모리 할당 실패");
            // 에러 발생 시 이전에 할당한 메모리 해제
            for (int j = 0; j < i; j++) {
                free(images[j].data);
            }
            free(images);
            free(all_data);
            return NULL;
        }
        // 모든 이미지 데이터가 연속으로 저장되어 있으므로, i번째 이미지 데이터를 복사
        memcpy(images[i].data, all_data + i * image_size, image_size * sizeof(float));
    }
    
    free(all_data);
    return images;
}

// 모든 가중치 파일(all_weights.bin)을 읽어 Weight 구조체 배열로 반환하는 함수
Weight* load_all_weights(const char *filename, int *num_weights) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open weight file %s\n", filename);
        return NULL;
    }
    // 전체 항목 수 읽기 (int32)
    int total_entries;
    if (fread(&total_entries, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read total entries\n");
        fclose(fp);
        return NULL;
    }
    
    // Weight 구조체 배열 할당
    Weight *weights = (Weight*)malloc(total_entries * sizeof(Weight));
    if (!weights) {
        fprintf(stderr, "Error: Memory allocation failed for weights array\n");
        fclose(fp);
        return NULL;
    }
    
    for (int i = 0; i < total_entries; i++) {
        // 키 길이 (int32)
        int key_length;
        if (fread(&key_length, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read key length for weight %d\n", i);
            // Free already allocated weights
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        // 키 문자열 읽기
        weights[i].key = (char*)malloc(key_length + 1);
        if (!weights[i].key) {
            fprintf(stderr, "Error: Memory allocation failed for weight key\n");
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        if (fread(weights[i].key, sizeof(char), key_length, fp) != (size_t)key_length) {
            fprintf(stderr, "Error: Failed to read weight key string\n");
            free(weights[i].key);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        weights[i].key[key_length] = '\0'; // null-terminate
        
        // 차원 수 (int32)
        if (fread(&weights[i].num_dims, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read num_dims for weight %s\n", weights[i].key);
            free(weights[i].key);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        // shape 배열 할당 및 읽기
        weights[i].shape = (int*)malloc(weights[i].num_dims * sizeof(int));
        if (!weights[i].shape) {
            fprintf(stderr, "Error: Memory allocation failed for shape of weight %s\n", weights[i].key);
            free(weights[i].key);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        for (int j = 0; j < weights[i].num_dims; j++) {
            if (fread(&weights[i].shape[j], sizeof(int), 1, fp) != 1) {
                fprintf(stderr, "Error: Failed to read shape dimension for weight %s\n", weights[i].key);
                free(weights[i].key);
                free(weights[i].shape);
                for (int k = 0; k < i; k++) {
                    free(weights[k].key);
                    free(weights[k].shape);
                    free(weights[k].data);
                }
                free(weights);
                fclose(fp);
                return NULL;
            }
        }
        // 원소 개수 (int32)
        if (fread(&weights[i].num_elements, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read num_elements for weight %s\n", weights[i].key);
            free(weights[i].key);
            free(weights[i].shape);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        // 데이터 배열 할당 및 읽기 (float32)
        weights[i].data = (float*)malloc(weights[i].num_elements * sizeof(float));
        if (!weights[i].data) {
            fprintf(stderr, "Error: Memory allocation failed for data of weight %s\n", weights[i].key);
            free(weights[i].key);
            free(weights[i].shape);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
        if (fread(weights[i].data, sizeof(float), weights[i].num_elements, fp) != (size_t)weights[i].num_elements) {
            fprintf(stderr, "Error: Failed to read float data for weight %s\n", weights[i].key);
            free(weights[i].key);
            free(weights[i].shape);
            free(weights[i].data);
            for (int j = 0; j < i; j++) {
                free(weights[j].key);
                free(weights[j].shape);
                free(weights[j].data);
            }
            free(weights);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);
    *num_weights = total_entries;
    return weights;
}

// 메모리 해제 함수
void free_all_weights(Weight *weights, int num_weights) {
    if (!weights) return;
    for (int i = 0; i < num_weights; i++) {
        free(weights[i].key);
        free(weights[i].shape);
        free(weights[i].data);
    }
    free(weights);
}