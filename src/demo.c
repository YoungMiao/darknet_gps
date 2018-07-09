#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "get_gps.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;
int fd;

float g_latitude=0.0;
float g_longitude=0.0;
char save_iamge_name[256]={0};
//image saveimage = make_empty_image(0,0,0);
image saveimage;
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);



/*************add save image thread *************************/
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
    char imageName[256];     // image name
    image saveImage;      // 保存图片
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
// 将数据加入队列
void PutDataIntoQueue(void)
{
    T_StructInfo tQueueData = {0};
    static UINT32 iCountNum = 0;

    if(strlen(save_iamge_name) != 0){
        // 对结构体的变量进行赋值
        tQueueData.iID = iCountNum;
        //tQueueData.imageName = save_iamge_name;
        tQueueData.saveImage = saveimage;
        snprintf(tQueueData.imageName, sizeof(tQueueData.imageName) - 1, save_iamge_name, iCountNum);
        printf("tQueueData.imageName:",tQueueData.imageName);
        //save_image(tQueueData.saveImage,tQueueData.imageName);


        // 计数值累加
        iCountNum ++;
        if (iCountNum >= MAX_QUEUE-1)
        {
            iCountNum = 0;
        }


        // 将数据加入队列(一直等到加入成功之后才退出)
        while (EnQueue(tQueueData) == -1)
        {
            Sleep(10);       // 加入失败,1秒后重试
        }
    }
}
// 从队列取出数据
void GetDataFromQueue(void)
{
    T_StructInfo tQueueData = {0};


    if (DeQueue(&tQueueData) == -1)
    {
        return;
    }

    save_image(tQueueData.saveImage,tQueueData.imageName);

}
// 数据入队列操作
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
//数据出队列操作
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
/******************************************************************/

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);


    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    //printf("save_iamge_name:%s\n",save_iamge_name);
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, g_latitude, g_longitude, &save_iamge_name, &saveimage);
    PutDataIntoQueue();
    free_detections(dets, nboxes);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *nodeal_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
   int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *getgps_in_thread(void *ptr)
{
    read_data(fd,&g_latitude,&g_longitude);
    //printf("gpsgpsgpsgspsgspsgpsgpsgpsgps");
    return 0;
}


void *save_image_data(void *ptr)
{
    while(1){
        GetDataFromQueue();  // 数据出队
    }
}


void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    fd=0;        
    int HOST_COM_PORT=1;       
    fd=open_port(HOST_COM_PORT);  
    if(fd<0)   
    {  
        perror("open fail!");  
    }  
    printf("open sucess!\n");  
    if((set_com_config(fd,9600,8,'N',1))<0)    
    {  
        perror("set_com_config fail!\n");  
    }  
    printf("The received worlds are:\n");  
    //printf("111111111111111111111");
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t getgps_thread;
    pthread_t fetch_thread;
    pthread_t nodeal_thread;
    if(pthread_create(&getgps_thread, 0, getgps_in_thread, 0)) error("Thread creation failed");
    //pthread_join(getgps_thread, 0);
    pthread_mutex_init(&g_mutex_queue_cs, &g_MutexAttr);
    pthread_cond_init(&queue_cv, NULL);

    pthread_t saveimage_thread;
    if(pthread_create(&saveimage_thread, 0, save_image_data, 0)) error("Thread creation failed");
    

    srand(2222222);
    //printf("22222222222222222222222");
    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));
    //printf("3333333333333333333333");
    if(filename){
        printf("video file: %s\n", filename);
        //cap = cvCaptureFromFile(filename);
	cap = cvCreateFileCapture(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);
    //printf("44444444444444444444444444444");
    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1280, 720);
        }
    }
    demo_time = what_time_is_it_now();
    //printf("555555555555555555");
    while(!demo_done){
	buff_index = (buff_index + 1) %3;
	
	if(count%3==0){
        	
		if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
		if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		if(!prefix){
		   fps = 1./(what_time_is_it_now() - demo_time);
		   demo_time = what_time_is_it_now();
		   display_in_thread(0);
		}else{
		  char name[256];
		  sprintf(name, "%s_%08d", prefix, count);
		  save_image(buff[(buff_index + 1)%3], name);
		}
		pthread_join(fetch_thread, 0);
		pthread_join(detect_thread, 0);	
	}else{
		if(pthread_create(&nodeal_thread, 0, nodeal_in_thread, 0)) error("Thread creation failed");
		if(!prefix){
		   fps = 1./(what_time_is_it_now() - demo_time);
		   demo_time = what_time_is_it_now();
		   display_in_thread(0);
		}else{
		  char name[256];
		  sprintf(name, "%s_%08d", prefix, count);
		  save_image(buff[(buff_index + 1)%3], name);
		}
		pthread_join(nodeal_thread, 0);
	}
	//pthread_join(fetch_thread, 0);
	//pthread_join(detect_thread, 0);
        ++count;
    }
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

