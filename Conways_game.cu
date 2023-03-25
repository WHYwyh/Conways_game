﻿#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 游戏宽度和高度
#define WIDTH 1200
#define HEIGHT 900
// 颜色色域256
#define COLOR_RANGE 256
#define HZ 144
#define USE_GPU 1
#define TILE_SIZE 32

// 细胞状态，0 表示死亡，1 表示存活
int h_cellls[2][HEIGHT][WIDTH] = { 0 };
int now = 0; //当前帧对应的数组id，上一帧对应的id为pre = now ^ 1;
int* d_cells[2];


// 初始化细胞状态
void init_cells() {
    // 随机生成细胞状态
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            h_cellls[now][i][j] = (rand() % 20) == 0;
        }
    }
}

// 绘制界面
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    // 蓝色
    //glColor3f(1.0 / COLOR_RANGE, 149.0 / COLOR_RANGE, 249.0 / COLOR_RANGE);
    // 绿色
    glColor3f(0.0 / COLOR_RANGE, 197.0 / COLOR_RANGE, 50.0 / COLOR_RANGE);
    glPointSize(1.0);
    glBegin(GL_POINTS);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (h_cellls[now][i][j] == 1) {
                glVertex2i(j, i);
            }
        }
    }
    glEnd();
    glutSwapBuffers();
}

// 更新细胞状态
void update_cells() {
    int pre = now ^ 1;
    memset(h_cellls[now], 0, sizeof(h_cellls[now]));
    // 统计每个细胞周围的存活细胞数量
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            for (int y = i - 1; y <= i + 1; y++) {
                for (int x = j - 1; x <= j + 1; x++) {
                    if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT && (x != j || y != i)) {
                        h_cellls[now][i][j] += h_cellls[pre][y][x];
                    }
                }
            }
        }
    } 
    // 根据细胞状态和周围存活细胞数量更新细胞状态
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            h_cellls[now][i][j] = (h_cellls[now][i][j] >= 3 && h_cellls[now][i][j] <= 7);
        }
    }
}

__global__ void update(int* d_old, int* d_new, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width) return;

    __shared__ int shared_input[TILE_SIZE + 2][TILE_SIZE + 2]; 

    int shared_x = threadIdx.x + 1;
    int shared_y = threadIdx.y + 1;
    int index = y * width + x;
    int shared_index = shared_y * (TILE_SIZE + 2) + shared_x;

    shared_input[shared_y][shared_x] = d_old[index]; //每个线程先把自身对应位置的数据写入共享内存

    //位于线程块边界的线程，把与自身相邻的边界数据写入共享内存

    if (threadIdx.x == 0) {
        if (x > 0) {
            shared_input[shared_y][0] = d_old[index - 1]; //(x,y)的左边(x-1,y)
        }
        else {
            shared_input[shared_y][0] = 0;
        }
    }

    if (threadIdx.x == blockDim.x - 1) {
        if (x < width - 1) {
            shared_input[shared_y][TILE_SIZE + 1] = d_old[index + 1];//(x,y)的右边(x+1,y)
        }
        else {
            shared_input[shared_y][TILE_SIZE + 1] = 0;
        }
    }


    if (threadIdx.y == 0) {
        if (y > 0) {
            shared_input[0][shared_x] = d_old[index - width];//(x,y)的上边(x,y-1)
        }
        else {
            shared_input[0][shared_x] = 0;
        }
    }

    if (threadIdx.y == blockDim.y - 1) {
        if (y < height - 1) {
            shared_input[TILE_SIZE + 1][shared_x] = d_old[index + width];//(x,y)的下边(x,y+1)
        }
        else {
            shared_input[TILE_SIZE + 1][shared_x] = 0;
        }
    }


    //等待同一个block的所有线程把数据都写入共享内存
    __syncthreads();

    int count = 0;
    for (int i =  - 1; i <=  + 1; i++) {
        for (int j =  - 1; j <=  + 1; j++) {
            if (!(i == 0 && j == 0)) {
                count += shared_input[shared_x + i][shared_y + j];
            }
        }
    }

    if (3 <= count && count <=7) {
        d_new[index] = 1;
    }
    else {
        d_new[index] = 0;
    }
}
dim3 dimBlock(TILE_SIZE, TILE_SIZE);
dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (HEIGHT + dimBlock.y - 1) / dimBlock.y);

// 计时器函数，每隔一定时间更新一次细胞状态，设置为144hz
void timer(int value) {
    now ^= 1; //now在0 1 之间不断转换
    int pre = now ^ 1;
    if (USE_GPU) {
        update <<< dimGrid, dimBlock >>> (d_cells[pre], d_cells[now], WIDTH, HEIGHT);
        cudaMemcpy(h_cellls[now], d_cells[now], WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    }
    else {
        update_cells();
    }
    glutPostRedisplay();
    glutTimerFunc(1000.0 / HZ, timer, 0);
}

void init_window() {
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Conway's Game of Life");
    glClearColor(255, 255, 255, 0.0);
    gluOrtho2D(0, WIDTH, 0, HEIGHT);
}

int main(int argc, char** argv) {

    // 初始化 屏幕
    glutInit(&argc, argv);
    init_window();

    // 初始化h_cellls
    srand(time(NULL));
    init_cells();

    if (USE_GPU) {
        cudaMalloc((void**)&d_cells[0], WIDTH * HEIGHT * sizeof(int));
        cudaMalloc((void**)&d_cells[1], WIDTH * HEIGHT * sizeof(int));
        cudaMemcpy(d_cells[now], h_cellls[now], WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    }

    // 注册回调函数
    glutDisplayFunc(display);

    glutTimerFunc(1000.0 / HZ, timer, 0);

    // 进入循环
    glutMainLoop();

    if (USE_GPU) {
        cudaFree(d_cells[0]);
        cudaFree(d_cells[1]);
    }
    return 0;
}