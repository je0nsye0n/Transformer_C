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

ImageData* load_image_data(const char *filename);

// Network 로드에 대한 로직
typedef struct{
    float *data;
    size_t size;
} Network;

void load_weights(const char *directory, Network network[], int count);

#endif