#include <stdio.h>
#include <string.h>
#include <ftw.h>
#include <pthread.h>
#include <time.h>


// 重定义数据类型
typedef signed   int        INT32;
typedef unsigned int        UINT32;
typedef unsigned char       UINT8;

// 宏定义
#define     MAX_QUEUE      10000          // 最大队列元素个数

// 结构体变量
typedef struct
{
    UINT32 iID;             // 编号
    UINT8  szInfo[100];     // 描述
    image 
} T_StructInfo;

// 全局变量定义
T_StructInfo g_tQueue[MAX_QUEUE] = {0};      // 队列结构体
UINT32 g_iQueueHead = 0;                     // 队列头部索引
UINT32 g_iQueueTail = 0;                     // 队列尾部索引
pthread_mutex_t     g_mutex_queue_cs;        // 互斥信号量
pthread_cond_t      queue_cv;
pthread_mutexattr_t g_MutexAttr;

// 函数声明
void PutDataIntoQueue(void);
void GetDataFromQueue(void);
INT32 EnQueue(T_StructInfo tQueueData);
INT32 DeQueue(T_StructInfo *ptStructData);
void Sleep(UINT32 iCountMs);



INT32 main(void)
{
    pthread_mutex_init(&g_mutex_queue_cs, &g_MutexAttr);
    pthread_cond_init(&queue_cv, NULL);


    // 在循环中执行入队和出队操作
    while (1)
    {
        PutDataIntoQueue();  // 数据入队


        Sleep(5 * 1000);     // 间隔5秒


        GetDataFromQueue();  // 数据出队


        Sleep(60 * 1000);    // 每一分钟执行一次出队和入队
    }


    return 0;
}


void PutDataIntoQueue(void)
{
    T_StructInfo tQueueData = {0};
    static UINT32 iCountNum = 0;


    // 对结构体的变量进行赋值
    tQueueData.iID = iCountNum;
    snprintf(tQueueData.szInfo, sizeof(tQueueData.szInfo) - 1, "zhou%d", iCountNum);


    // 计数值累加
    iCountNum ++;
    if (iCountNum >= MAX_QUEUE-1)
    {
        iCountNum = 0;
    }


    // 将数据加入队列(一直等到加入成功之后才退出)
    while (EnQueue(tQueueData) == -1)
    {
        Sleep(1000);       // 加入失败,1秒后重试
    }


    // 打印加入的数据
    printf("PutDataIntoQueue: ID=%d, Info=%s\n", tQueueData.iID, tQueueData.szInfo);
}




void GetDataFromQueue(void)
{
    T_StructInfo tQueueData = {0};


    if (DeQueue(&tQueueData) == -1)
    {
        return;
    }


    // 打印取出的数据
    printf("GetDataFromQueue: ID=%d, Info=%s\n", tQueueData.iID, tQueueData.szInfo);
}



INT32 EnQueue(T_StructInfo tQueueData)
{
    INT32  iRetVal  = 0;
    UINT32 iNextPos = 0;


    pthread_mutex_lock(&g_mutex_queue_cs);
    iNextPos = g_iQueueTail + 1;


    if (iNextPos >= MAX_QUEUE)
    {
        iNextPos = 0;
    }


    if (iNextPos == g_iQueueHead)
    {
        iRetVal = -1;   // 已达到队列的最大长度
    }
    else
    {
        // 入队列
        memset(&g_tQueue[g_iQueueTail], 0x00,  sizeof(T_StructInfo));
        memcpy(&g_tQueue[g_iQueueTail], &tQueueData, sizeof(T_StructInfo));


        g_iQueueTail = iNextPos;
    }


    pthread_cond_signal(&queue_cv);
    pthread_mutex_unlock(&g_mutex_queue_cs);


    return iRetVal;
}


INT32 DeQueue(T_StructInfo *ptStructData)
{
    T_StructInfo tQueueData = {0};


    if (ptStructData == NULL)
    {
        return -1;
    }


    pthread_mutex_lock(&g_mutex_queue_cs);


    while (g_iQueueHead == g_iQueueTail)
    {
        pthread_cond_wait(&queue_cv, &g_mutex_queue_cs);
    }


    memset(&tQueueData, 0x00, sizeof(T_StructInfo));
    memcpy(&tQueueData, &g_tQueue[g_iQueueHead], sizeof(T_StructInfo));
    g_iQueueHead ++;


    if (g_iQueueHead >= MAX_QUEUE)
    {
        g_iQueueHead = 0;
    }


    pthread_mutex_unlock(&g_mutex_queue_cs);
    memcpy(ptStructData, &tQueueData, sizeof(T_StructInfo));


    return 0;
}


void Sleep(UINT32 iCountMs)
{
    struct timeval t_timeout = {0};


    if (iCountMs < 1000)
    {
        t_timeout.tv_sec  = 0;
        t_timeout.tv_usec = iCountMs * 1000;
    }
    else
    {
        t_timeout.tv_sec  = iCountMs / 1000;
        t_timeout.tv_usec = (iCountMs % 1000) * 1000;
    }
    select(0, NULL, NULL, NULL, &t_timeout);    // 调用select函数阻塞程序
}