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

// input.bin 파일을 읽어 ImageData 구조체를 반환하는 함수
ImageData* load_image_data(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return NULL;
    }
    
    // 헤더 읽기: 이미지 개수, 채널, 높이, 너비 (4개의 int 값)
    int header[4];
    size_t n_read = fread(header, sizeof(int), 4, fp);
    if (n_read != 4) {
        fprintf(stderr, "Error: Failed to read header from %s\n", filename);
        fclose(fp);
        return NULL;
    }
    
    int n = header[0];
    int c = header[1];
    int h = header[2];
    int w = header[3];
    printf("Loaded image shape: [%d, %d, %d, %d]\n", n, c, h, w);
    
    // 전체 요소 개수 계산: N x C x H x W
    long long total = (long long)n * c * h * w;
    
    // 모든 float 데이터를 저장할 메모리 할당
    float *data = (float*)malloc(total * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    n_read = fread(data, sizeof(float), total, fp);
    if (n_read != total) {
        fprintf(stderr, "Warning: Expected to read %lld floats but read %zu floats\n", total, n_read);
    }
    fclose(fp);
    
    // ImageData 구조체 할당 및 초기화
    ImageData *imgData = (ImageData*)malloc(sizeof(ImageData));
    if (imgData == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for ImageData struct\n");
        free(data);
        return NULL;
    }
    imgData->n = n;
    imgData->c = c;
    imgData->h = h;
    imgData->w = w;
    imgData->data = data;
    
    return imgData;
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