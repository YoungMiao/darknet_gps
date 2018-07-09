#include <stdio.h>
#include <string.h>
#include <ftw.h>
#include <pthread.h>
#include <time.h>


// �ض�����������
typedef signed   int        INT32;
typedef unsigned int        UINT32;
typedef unsigned char       UINT8;

// �궨��
#define     MAX_QUEUE      10000          // ������Ԫ�ظ���

// �ṹ�����
typedef struct
{
    UINT32 iID;             // ���
    UINT8  szInfo[100];     // ����
    image 
} T_StructInfo;

// ȫ�ֱ�������
T_StructInfo g_tQueue[MAX_QUEUE] = {0};      // ���нṹ��
UINT32 g_iQueueHead = 0;                     // ����ͷ������
UINT32 g_iQueueTail = 0;                     // ����β������
pthread_mutex_t     g_mutex_queue_cs;        // �����ź���
pthread_cond_t      queue_cv;
pthread_mutexattr_t g_MutexAttr;

// ��������
void PutDataIntoQueue(void);
void GetDataFromQueue(void);
INT32 EnQueue(T_StructInfo tQueueData);
INT32 DeQueue(T_StructInfo *ptStructData);
void Sleep(UINT32 iCountMs);



INT32 main(void)
{
    pthread_mutex_init(&g_mutex_queue_cs, &g_MutexAttr);
    pthread_cond_init(&queue_cv, NULL);


    // ��ѭ����ִ����Ӻͳ��Ӳ���
    while (1)
    {
        PutDataIntoQueue();  // �������


        Sleep(5 * 1000);     // ���5��


        GetDataFromQueue();  // ���ݳ���


        Sleep(60 * 1000);    // ÿһ����ִ��һ�γ��Ӻ����
    }


    return 0;
}


void PutDataIntoQueue(void)
{
    T_StructInfo tQueueData = {0};
    static UINT32 iCountNum = 0;


    // �Խṹ��ı������и�ֵ
    tQueueData.iID = iCountNum;
    snprintf(tQueueData.szInfo, sizeof(tQueueData.szInfo) - 1, "zhou%d", iCountNum);


    // ����ֵ�ۼ�
    iCountNum ++;
    if (iCountNum >= MAX_QUEUE-1)
    {
        iCountNum = 0;
    }


    // �����ݼ������(һֱ�ȵ�����ɹ�֮����˳�)
    while (EnQueue(tQueueData) == -1)
    {
        Sleep(1000);       // ����ʧ��,1�������
    }


    // ��ӡ���������
    printf("PutDataIntoQueue: ID=%d, Info=%s\n", tQueueData.iID, tQueueData.szInfo);
}




void GetDataFromQueue(void)
{
    T_StructInfo tQueueData = {0};


    if (DeQueue(&tQueueData) == -1)
    {
        return;
    }


    // ��ӡȡ��������
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
        iRetVal = -1;   // �Ѵﵽ���е���󳤶�
    }
    else
    {
        // �����
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
    select(0, NULL, NULL, NULL, &t_timeout);    // ����select������������
}