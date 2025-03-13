#ifndef _Network_H
#define _Network_H

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

ImageData* load_image_data(const char *filename);
Weight* load_all_weights(const char *filename, int *num_weights);
void free_all_weights(Weight *weights, int num_weights);

#endif