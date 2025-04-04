
//-------------------------------------TIPS FOR SETTING UP CODE TO RUN:----------------------

//			1. ***Make sure to include cufft lib files in Debug > Properties > Linker > input > Additional Dependencies (Add "cufft.lib")***
//			2. Include paths (Debug > Properties -> VC++): C:\Users\alekt_000\source\repos\OnetimeFFTGPU\OnetimeFFTGPU; ***MAKE SURE TO CHANGE USER***
//			3. Lib paths (Properties -> VC ++): C:\Users\alekt_000\anaconda3\libs;
//			4. Use an inplace FFT transform (i.e. input data variable is where the output data will be stored. In our case this variable is d_data). Use cufftplanmany().
//			*5. Make sure that NX, BATCH, THREADS, BLOCKS are adjusted to DATASIZE! 
//			6. Make sure using x64 system.
//			7. Run in release mode instead of debug mode if you are not debugging since it will run faster.


//---------------------------------------GNUPLOT TIPS:----------------------------------------

//			1. To change to specific directory, invoke File -> Change Directory, and then navigate to said directory.
//			2. To plot GNUPlot go to gnuplot command window  cd 'C:\Users\alekt_000\source\repos\OnetimeFFTGPU\OnetimeFFTGPU'  (NEXT LINE)    gnuplot > plot "FFT_Output.dat" with linespoints

/* --------------------------------------***TO DO***------------------------------------------
                        1. Use OpenGL to visualize the Data

                        2. Consider using R2C (Real to Complex) instead of C2C (Complex to Complex) for the FFT to speed it up even more.
                        *For R2C you might not be able to use the same variable (i.e. cannot do inplace transform).

                        3. Consider downloading and using CUDA 11.0 since it says it has faster cufft routines.

                        4. Once OpenGL Visualization is working, incorporate slice slider for viewing layers of the data.

                        5. Have cufftplan() before while loop and cufftdestroyplan() after while loop, since you can use the plan multiple times.

                        6. Malloc all memory before while loop and free all memory after while loop, since we can reuse the memory allocations.

*/

#pragma warning(disable : 4996)

//Example Opengl:
#include <thread>
#include "helper_gl.h"
#include "freeglut.h"

//CUDA Headers:
//#include "cuda_runtime.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include "helper_functions.h" 
#include "helper_cuda.h"
//#include <device_launch_parameters.h>
#include "./cuda_kernel.cuh"
#include "common.h"

//#define SIZE  1024
#define REAL     0
#define IMAG     1

//CHANGEABLE FFT SETTINGS: (***DATA SET CANNOT BE TOO LARGE OR ELSE BATCH WILL FAIL AND NOT DO EVERY SIGNAL IN TRANSFORM***)
#define postTriggerSamples 2000// 1000//24960; // for sure a ot less
#define recordsPerBuffer  100// 160; // A-scan per B-scan //40000
#define buffersPerAcquisition  100//63; // B x C scans //10000 => 200 after interleaving

//int postTriggerSamples;
//int recordsPerBuffer;
//int buffersPerAcquisition;

#define DATASIZE postTriggerSamples * recordsPerBuffer *  buffersPerAcquisition //6021972 //5929741 //XLENGTH * YLENGTH * ZLENGTH    // Amount of data points in each data set. // RECORD LENGTH
#define NX   (int) (postTriggerSamples/2.0)//410 //150						// SHOULD BE THE AMOUNT OF POINTS FOR ONE SIGNAL. (LENGTH OF SIGNAL IN X DIMENSION ARRAY) // Maybe might be 10000 as it gives different kind of output plot that might be correct.
#define BATCH    recordsPerBuffer *  buffersPerAcquisition //floor(DATASIZE/NX) //14687 //(DATASIZE/NX)					// BATCH is amount of mini transforms to do.
#define FFT_DATA_MAXIMUM_VALUE 3000000//1000 //40000 //4086 //14000	//80000		//Represents the max value that is possible after FFT the data.
#define DC_CUT_OFFSET 180 //12 chnaged to zero bc photoreceiver removes envelop
float UPPER_NORMALIZATION_THRESHOLD = 10000.0f;//1.0f
float LOWER_NORMALIZATION_THRESHOLD = 0.1f;//0.7f

//Cube Coordinate Data Macros:
#define XLENGTH recordsPerBuffer	//80 //198 //181 //X
#define YLENGTH  ((floor(NX/2.0)) - DC_CUT_OFFSET) //212 //183 //181 // this is the length of pulses //Y
//#define YLENGTH ((floor(NX/2)))
#define ZLENGTH	buffersPerAcquisition/2 //(floor(DATASIZE/(XLENGTH * YLENGTH))) //(floor(DATASIZE/(XLENGTH * YLENGTH))) //100	//80 //150 //80  //181 //Z
//#define FRAMES	ZLENGTH	//150 //80  //181 
#define TOTAL  XLENGTH * YLENGTH * ZLENGTH * 3
#define COLORTOTAL XLENGTH * YLENGTH * ZLENGTH * 4

//CHANGEABLE MAGNITUDE SETTINGS: //use these links for help in detemining proper values (https://en.wikipedia.org/wiki/CUDA) (https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid)
#define THREADS 1024
#define BLOCKS  recordsPerBuffer *  buffersPerAcquisition // (round((DATASIZE/THREADS)) + 100) //iNCORPORATE THE ROUNDING FUNCTION

//Namespaces:
using namespace std;

//Dummy Data source:
string dataFile = "OCTDATA.dat"; //use it to test the work

//-----------------------------PERIOD GUESS DATA--------------------------
//Editables:
#define SIZE DATASIZE //3500 //take about 7 periods
double periodGuess = postTriggerSamples; //Initial guess
double interp_factor = 1.0;//1.0f;
double max_tolerance = 0.2;//0.2f;
double min_tolerance = 0.05;//0.05f;
#define VISA_BUFF_SIZE DATASIZE
#define VISA_BUFF_OFFSET 10 //10 OR 11 // OFFSET IS 2 (START BYTE AND BYTE LENGTH CHARACTERS) + X (# OF CHARACTERS TO REPRESENT VISA_BUFF_SIZE)(E.X. VISA_BUFF_SIZE = 2000000 -> X = 7) + 1 (END BYTE) 
float test_val = 0;
//Declarations:
double determinedPeriod;
double period = periodGuess;
int endbound;
double myMax;
double myMin;
//int span;
double span;
vector<double> maxes;
vector<double> mins;
double initialMax;
double ffirstMin;
double firstMax;
int firstMinEnd;
double lfirstMin;
double fnextMin;
double nextMax;
int nextMinEnd;
double lnextMin;
int firstMin;
double nextMin;
int length;
int n = 2;
bool jumping = true;
int jump;
int areaStart;
int areaEnd;
vector<double> area;
vector<double> areaMins;
int predictedPeriod = periodGuess;
double* ahead;
//float* waveform = new float[SIZE];

int N = 1;

//-----------------------------CUDA Constants-----------------------------
int nn[1] = { NX };
int inembed[] = { 0 };
int onembed[] = { 0 };

//Declare vector:
vector<double> names(DATASIZE);

//Declare double:
double input;

//Memory variables:
//int mem_size = sizeof(cufftComplex) * (DATASIZE + (DATASIZE / 150));
int mem_size = sizeof(cufftComplex) * (DATASIZE);
int mem_size1 = sizeof(cufftComplex) * ((int)(DATASIZE / 2));
//int floatmem_size = sizeof(float) * (DATASIZE + (DATASIZE / 150));
int floatmem_size = sizeof(float) * (DATASIZE);
int floatmem_size1 = sizeof(float) * ((int)(DATASIZE / 2));
int uint16_tmem_size = sizeof(uint16_t) * (DATASIZE);

//TEST:
float* waveform = (float*)malloc(floatmem_size);

//Allocate host memory for result signal:
//cufftComplex* h_calcdata = (cufftComplex*)malloc(mem_size);

// Allocate host memory for the signal:
cufftComplex* h_data = (cufftComplex*)malloc(mem_size);

//Prepared for pinned memory:
//cufftComplex* pinned_data; //Prepared for pinned memory.

//Allocate host memory for result data:
float* h_sharedresult = (float*)malloc(floatmem_size);

// Allocate device memory for signal after transform:
cufftComplex* d_data;

// Allocate device pointer for pinned memory:
float* d_pinned_data;

// Allocate device memory for result data:
float* d_sharedresult;

// Allocate memory for finding maximum:
float* h_normalize = (float*)malloc(floatmem_size);
float* d_normalize;
float normalize_max;

//TEST:
float* d_test;
float* d_test1;

cufftComplex* d_frame;
float* d_waveform;
float* d_waveform1;
float* h_test = (float*)malloc(floatmem_size * 2);
float* h_test1 = (float*)malloc(floatmem_size * 2);
float* d_interleaved;
float* h_test2 = (float*)malloc(sizeof(float) * DATASIZE);

// Allocate host memory for output
//float* h_interleaved = new float[sizeof(float) * 2 * DATASIZE];
float* h_interleaved = (float*)malloc(sizeof(float) * 2 * DATASIZE);

// Kernel setup
int threadsPerBlock = 256;
int blocksPerGrid = (2 * DATASIZE + threadsPerBlock - 1) / threadsPerBlock;

// Interpolation logic (global)
__device__ int d_flag = 0;  // Flag stored in device memory
__device__ int h_flag = 0;

//Allocate host memory for result data:
//float3* d_3D_data;

// Allocate device memory for signal before transform:
//cufftComplex* d_data_old;

//Establish plan handle:
cufftHandle plan;



//-----------------------------OpenGL Constants---------------------------
#define REFRESH_DELAY     0 //ms

// map OpenGL buffer object for writing from CUDA
//float* dptr = new float[TOTAL];
float4* dptr;
float4* dptr1;
float* colors = new float[COLORTOTAL];
// Z-Plane:
double eqn[] = { 1.0, 0.0, 0.0, -0.5 }; //cross section: change last argument (value without f, can be from 0 to 1) set other three arguments one at a time equal to 1 to choose which axis you want to be slicing
double eqn2[] = { -1.0, 0.0, 0.0, 0.6 };
bool z_plane = true;
// Y-Plane:
double eqn3[] = { 0.0, 1.0, 0.0, -0.5 };
double eqn4[] = { 0.0, -1.0, 0.0, 0.6 };
bool y_plane = false;
// X-Plane:
double eqn5[] = { 0.0, 0.0, 1.0, -0.5 };
double eqn6[] = { 0.0, 0.0, -1.0, 0.6 };
bool x_plane = false;

const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 256; //determined by size of data
const unsigned int mesh_height = 256; //determined by size of data

// vbo variables
GLuint vbo;
GLuint vbo1;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

float g_fAnim = 0.0;

//FOR TIMING
long drawendTime = 0;
long drawstartTime = 0;

//FOR 3D CUBE:
unsigned int vbopositions;
unsigned int vbocolors;
float* positions = new float[TOTAL];
GLsizei draw_number = (XLENGTH * YLENGTH * ZLENGTH);//DATASIZE;
//unsigned int width = LENGTH * 1.5;
//unsigned int height = LENGTH;

// For slider:
int current_key;
float dx, dy;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 30.0, rotate_y = -30.0;
float translate_z = -5.0; //-3.0 controls zoom in and out


/*

----------------------------PRE DEFINED FUNCTIONS--------------------------------

*/






// MAGNITUDE KERNEL FUNCTION:
__global__ void d_magnitude(cufftComplex* data, float4* result, int numElements, float* testdata, float* normalize_result, int CONTRAST, int N, float UPPER_NORMALIZATION_THRESHOLD, float LOWER_NORMALIZATION_THRESHOLD, int N_AVE) {

    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = (NX * blockIdx.x) + (DC_CUT_OFFSET)+threadIdx.x; //where the first term "410" is the period of each fft and third term "11" is how much to offset - 1 so in this case the offset is actually 12 to get rid of DC portion of FFT.
    //int i = (NX * blockIdx.x) + threadIdx.x;

    if (i < numElements) {
        //result[i] = 0;
        //result[i + 1] = 0;
        //printf("%f\n", data[0].x);
        //printf("%f\n", (((hypotf(data[0].x, data[0].y)) / FFT_DATA_MAXIMUM_VALUE) * 1.0f));
        //printf("%f\n", (((hypotf(data[1].x, data[1].y)))));
        //enter max function

        //result[i] = make_float4(0.0f, 0.0f, 1.0f,(((hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE) * 1.0f));

        //result[i + 3] = (((hypotf(data[i].x, data[i].y))));
        //printf("%f\n", result[50].z);
        //Maybe just write the sqrt(x^2 + y^2) to speed it up

        //if ((((hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE) * 1.0f) <= 0.1f) {
        //result[j] = make_float4(0.5f, 0.5f, 0.5f, (((hypotf(data[i].x, data[i].y)) / (FFT_DATA_MAXIMUM_VALUE)) * 1.0f)); //GRAY

        // Correct Line for no normalization:
        float FACTOR = 0.00001 / 100.0 * ((float)CONTRAST + 0.01);
        float test1 = (hypotf(data[i].x, data[i].y)) * FACTOR;
        //printf("%f , ", testdata[j]);
        if (N_AVE == 1) {
            result[j] = make_float4(0.7f, 0.498039f, 0.196078f, test1); //GOLD
        }
        else if (N < N_AVE) {
            testdata[j] = (testdata[j] + test1);
            //printf("%f. \n", testdata[j]);
            result[j] = make_float4(0.7f, 0.498039f, 0.196078f, normalize_result[j]); //GOLD result[j] = normalize_result[j];
        }
        else {
            result[j] = make_float4(0.7f, 0.498039f, 0.196078f, testdata[j] / N_AVE); //GOLD
            normalize_result[j] = result[j].w;
            testdata[j] = 0;
        }






        //result[j] = make_float4(0.7f, 0.498039f, 0.196078f, (((hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE))); //GOLD
        //result[j] = make_float4(0.7f, 0.498039f, 0.196078f, 1.0f); //GOLD
        //result[j] = make_float4(0.7f, 0.498039f, 0.196078f, (1/(((hypotf(data[i].x, data[i].y)) / (FFT_DATA_MAXIMUM_VALUE))) * 1.0f));

        if (1) {
            if (result[j].w > UPPER_NORMALIZATION_THRESHOLD) {
                result[j].w = 1.0f;
            }
            else if (result[j].w <= LOWER_NORMALIZATION_THRESHOLD) {
                result[j].w = 0.0f;
            }

        }

        //normalize_result[j] = (hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE;																												   //result[j] = make_float4(0.858824f, 0.858824f, 0.439216f, (((hypotf(data[i].x, data[i].y)) / (FFT_DATA_MAXIMUM_VALUE)) * 1.0f)); //LIGHT GOLD
        //testdata[j] = ((hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE);

        //testdata[j] = data[i].x;
    //}
    //else if ((((hypotf(data[i].x, data[i].y)) / FFT_DATA_MAXIMUM_VALUE) * 1.0f) <= 0.7f) {
        //result[i] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    //}
    //else {
        //result[i] = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
    //}
    }

}

//FRAMING KERNEL FUNCTION:
int framing_elements;
//int kNum = (std::floor(n * determinedPeriod) + std::floor(determinedPeriod));
int framing_threads;
int framing_blocks;
__global__ void d_framing(double determinedPeriod, float* waveform, cufftComplex* h_data, int numElements) {

    int n = blockIdx.x;
    int q = blockDim.x * blockIdx.x + threadIdx.x;
    int k = floor(n * determinedPeriod) + (threadIdx.x);
    //printf("TST! = %f\n",(float)waveform[500]);

    h_data[q].x = waveform[q];
    h_data[q].y = 0;
    // printf("%f\n", (float)waveform[500]);
    if (0) {//(n < numElements) {

        //for (int k = floor(n * determinedPeriod); k < (floor(n * determinedPeriod) + floor(determinedPeriod)); k++) {

        //h_data[q].x = (float)(waveform[k + (VISA_BUFF_OFFSET - 1)] * -1); //Add n to get all elements.
        h_data[q].x = (float)(waveform[k - 1]); //Add n to get all elements.
        h_data[q].y = 0;

        //printf("%f\n", (float)waveform[500]);
        //}
    }

}

__global__ void d_framing1(float* waveform, float* h_data) {

    int n = blockIdx.x;
    int q = blockDim.x * blockIdx.x + threadIdx.x;
    //int k = floor(n * determinedPeriod) + (threadIdx.x);
    //printf(" value[%d] = %f\n",q, waveform[q]);
    //printf(" threadIdx.x =  %d\n", threadIdx.x);
    //printf("q = %d\n", q);

    h_data[q] = waveform[q];
    //h_data[q].y = 0;
    if (0) {//(n < numElements) {

        //for (int k = floor(n * determinedPeriod); k < (floor(n * determinedPeriod) + floor(determinedPeriod)); k++) {

        //h_data[q].x = (float)(waveform[k + (VISA_BUFF_OFFSET - 1)] * -1); //Add n to get all elements.
        //h_data[q].x = 10.0;// (float)(waveform[k - 1] * -1); //Add n to get all elements.
        //h_data[q].y = 0;

        //printf("%f\n", (float)waveform[500]);
        //}
    }

}

__global__ void d_framing2(double determinedPeriod, float* waveform, cufftComplex* h_data, int numElements, float* waveform1) {

    int n = blockIdx.x;
    int q = blockDim.x * blockIdx.x + threadIdx.x;
    int k = floor(n * determinedPeriod) + (threadIdx.x);
    //printf("TST! = %f\n",(float)waveform[500]);

    if (q % 2 == 0) {
        if (q == 0) {
            h_data[(int)(q / 2)].x = waveform[q];
            h_data[(int)(q / 2)].y = 0;
            waveform1[(int)(q / 2)] = waveform[q];
        }
        else {
            h_data[(int)(q / 2)].x = (waveform[q] + waveform[q - 1]) / 2.0;
            h_data[(int)(q / 2)].y = 0;
            waveform1[(int)(q / 2)] = (waveform[q] + waveform[q - 1]) / 2.0;
        }
    }
    // printf("%f\n", (float)waveform[500]);
    if (0) {//(n < numElements) {

        //for (int k = floor(n * determinedPeriod); k < (floor(n * determinedPeriod) + floor(determinedPeriod)); k++) {

        //h_data[q].x = (float)(waveform[k + (VISA_BUFF_OFFSET - 1)] * -1); //Add n to get all elements.
        h_data[q].x = (float)(waveform[k - 1]); //Add n to get all elements.
        h_data[q].y = 0;

        //printf("%f\n", (float)waveform[500]);
        //}
    }

}


// DEBUG CUDA PRINT FUNCTION:
__global__ void print_kernel(float* waveform) {
    printf("GPU: %f\n", (float)waveform[500]);
}

__global__ void print_kernel_complex(cufftComplex* waveform) {
    printf("GPU complex: %f\n", (float)waveform[500].x);
}

__global__ void print_kernel_float(cufftComplex* waveform) {
    printf("Length: %d\n", (sizeof(waveform) / sizeof(waveform[0])));
}

// Threads:
//void requestData();

// GL Functionality:
bool initGL(int* argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags);
bool runTest(int argc, char** argv, char* ref_file);

// Rendering Callbacks:
void kernel(int argc, char** argv);
void display();
void display1();

void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda Functionality:
void runCuda(struct cudaGraphicsResource** vbo_resource);

// Period Estimation Code:
double periodEstimation(float* waveform);



bool runTest(int argc, char** argv, char* ref_file)
{

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);


    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    //3D CUBE STUFF:
        // Create buffer object

    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);

    // Create VBO


    // register callbacks
   //glutDisplayFunc(display1);
    //glutKeyboardFunc(keyboard);
    //glutMouseFunc(mouse);
    //glutMotionFunc(motion);





    //long FFTstart = clock();

    // run the cuda part
    //runCuda(&cuda_vbo_resource);

    //long FFTend = clock();

    //cout << "Transform took " << (FFTend - FFTstart) << " millis" << endl;




    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Kernel code to interleave the arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void interleaveKernel(float* d1, float* d2, float* di, int sz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) {
        di[2 * idx] = d1[idx];      // Even indices
        di[2 * idx + 1] = d2[idx];  // Odd indices
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Kernel code to set flags
////////////////////////////////////////////////////////////////////////////////
__global__ void setFlagtoOne() {
    d_flag = 1;
}

__global__ void setFlagtoZero() {
    d_flag = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource** vbo_resource)
{
    // execute the FFT
    long startTime = clock();
    // map OpenGL buffer object for writing from CUDA
    //float4* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);

    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
        *vbo_resource);
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // Copy host memory to device:
    //cudaMemcpy(d_data, pinned_data, mem_size, cudaMemcpyHostToDevice); //LOOK INTO BEING ABLE TO GET RID OF THIS IN FINAL PRODUCT BY STREAMING DATA DIRECTLY TO GPU.

    //test_val = (float)sampleValues[500];
    //printf("sampleValues length (GPU code) = %d\n", (int) (sizeof(sampleValues) / sizeof(float)) );
    //printf("sampleValues = %f\n", sampleValues_copy[500]);

    //DEBUG MODE:
    if (0) {
        ofstream myout;

        myout.open("Output1_3.dat");

        for (uint32_t i = 0; i < DATASIZE; i++) {

            //Print data to dat file:
            myout << i << " " << (float)sampleValues_copy[i] << endl;
            //myout << i << " " << i] << endl;
            //printf("i = %d\n", i);

        }

        myout.close();
        //Display finish message:
        cout << "Finished!" << endl;

    }

    //cout << test_val << endl;
    //cout << sizeof(sampleValues)/sizeof(float) << endl;

    cudaMemcpy(d_waveform, sampleValues_copy, sizeof(float) * (DATASIZE), cudaMemcpyHostToDevice); // straight from the buffer

    // Interpolation code here

    // Copy flag from host to device
    cudaMemcpyToSymbol(d_flag, &h_flag, sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

    // If flag == 0 set flag to 1, copy the waveform into memory, then capture next sample
    if (h_flag == 0) {

        // Modify d_flag on the device via kernel
        setFlagtoOne << <1, 1 >> > ();
        cudaDeviceSynchronize();

        // Copy flag back from device to host
        cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

        // Copy interleaved array to h_test2 to view on MATLAB
        cudaMemcpy(h_test2, d_waveform, floatmem_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Copy first waveform to d_waveform1
        cudaMemcpy(d_waveform1, d_waveform, sizeof(float) * DATASIZE, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // Test on MATLAB to view raw data
        if (1) {
            ofstream myout;
            myout.open("Output_no_interleave.dat");

            for (uint32_t i = 0; i < DATASIZE; i++) {
                // Print data to a .dat file
                myout << i << " " << (float)h_test2[i] << endl;
            }

            myout.close();
            // Display finish message
            cout << "Finished!" << endl;
        }

        return;
    }

    // Interleave if flag == 1
    if (h_flag == 1) {
        // Reset the flag back to 0
        setFlagtoZero << <1, 1 >> > (); // Set flag on GPU
        cudaDeviceSynchronize();

        // Copy flag back from device to host
        cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(int), 0, cudaMemcpyDeviceToHost);

        // Interleave the two arrays (d_waveform1 has the previous sample & d_waveform has the current sample)
        interleaveKernel << <blocksPerGrid, threadsPerBlock >> > (d_waveform1, d_waveform, d_interleaved, DATASIZE);
        cudaDeviceSynchronize();

        // Copy interleaved array to h_interleaved to view on MATLAB
        cudaMemcpy(h_interleaved, d_interleaved, 2 * floatmem_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Copy the interleaved result back to d_waveform
        //cudaMemcpy(d_waveform, d_interleaved, 2 * floatmem_size, cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();

        // Test on MATLAB to view raw data
        if (0) {
            ofstream myout;

            myout.open("Output_interleaved.dat");

            for (uint32_t i = 0; i < 2 * DATASIZE; i++) {

                //Print data to dat file:
                myout << i << " " << (float)h_interleaved[i] << endl;
                //myout << i << " " << i] << endl;
                //printf("i = %d\n", i);

            }

            myout.close();
            //Display finish message:
            cout << "Finished!" << endl;

        }

    }

    //cudaMemcpy(d_waveform, next_frame, floatmem_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(h_test1, d_waveform, floatmem_size, cudaMemcpyDeviceToHost);
    //printf("h_test 1 = %.20f\n", h_test1[500]);

    //print_kernel << <1, 1 >> > (d_waveform);

    //cudaMemcpy(d_waveform, d_pinned_data, floatmem_size, cudaMemcpyHostToDevice); //pinned data form

    if (0) {
        d_framing1 << <framing_blocks, framing_threads >> > (d_waveform, d_waveform1);
        cudaDeviceSynchronize();
        cudaMemcpy(h_test1, d_waveform1, floatmem_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("h_test 1 = %f\n", h_test1[500]);
    }



    if (determinedPeriod > 1000) // i.e. determinedPeriod = 2000
        d_framing2 << < (int)framing_elements * 2, (int)determinedPeriod / 2 >> > (determinedPeriod, d_waveform, (cufftComplex*)d_frame, framing_elements, d_test);
    else
        d_framing << <framing_blocks, framing_threads >> > (determinedPeriod, d_waveform, (cufftComplex*)d_frame, framing_elements);

    cudaMemcpy(h_test, d_test, floatmem_size, cudaMemcpyDeviceToHost);

    if (0) {
        ofstream myout;

        myout.open("FFT_Output2.dat");

        for (uint32_t i = 0; i < DATASIZE / 2; i++) {

            //Print data to dat file:
            myout << i << " " << (float)h_test[i] << endl;
            //myout << i << " " << i] << endl;
            //printf("i = %d\n", i);

        }

        myout.close();
        //Display finish message:
        cout << "Finished!" << endl;

    }
    //d_framing << <1, 1 >> > (determinedPeriod, d_waveform, (cufftComplex*)d_frame, framing_elements);

    //print_kernel_complex << <1, 1 >> > (d_frame);
    //cudaMemcpy(h_test, d_waveform1, floatmem_size, cudaMemcpyDeviceToHost);
    //printf("h_test 1 = %.20f\n", h_test[500]);

    long transformStartTime = clock();

    //Execute plan:
    //cufftExecC2C(plan, (cufftComplex*)d_pinned_data, (cufftComplex*)d_data, CUFFT_FORWARD);

    cufftExecC2C(plan, (cufftComplex*)d_frame, (cufftComplex*)d_data, CUFFT_FORWARD);

    //print_kernel_complex << <1, 1 >> > (d_data);
    //TEST
    //cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost);

    long transformFinishTime = clock();

    //GPU Calculate Magnitude:
    //d_magnitude << < BLOCKS, THREADS >> > (d_data, dptr, DATASIZE);
    //print_kernel_float << <1, 1 >> > (d_data);

    //d_magnitude << < (XLENGTH * YLENGTH), ZLENGTH >> > (d_data, dptr, DATASIZE, d_test);
    //d_magnitude << < (XLENGTH * YLENGTH), ZLENGTH >> > (d_data, dptr, DATASIZE, d_test, d_normalize);

    if (0) {
        printf("XLENGTH %d\n", XLENGTH);
        printf("YLENGTH %d\n", YLENGTH);
        printf("ZLENGTH %d\n", ZLENGTH);
        printf("DATASIZE %d\n", DATASIZE);
        printf("BATCH %d\n", BATCH);


        print_kernel_float << <1, 1 >> > (d_data);
    }

    if (0) {
        char buffer[50];
        sprintf(buffer, "%f", h_test[100]);
        MessageBox(0, buffer, "Title", MB_OK);
    }

    cudaMemcpy(d_test, h_test, floatmem_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normalize, h_test1, floatmem_size1, cudaMemcpyHostToDevice);
    //d_test = d_test1;
    if (determinedPeriod > 1000) // i.e. determinedPeriod = 2000
        d_magnitude << < (XLENGTH * ZLENGTH), YLENGTH >> > (d_data, dptr, (int)(DATASIZE / 2), d_test, d_normalize, CONTRAST, N, UPPER_NORMALIZATION_THRESHOLD, LOWER_NORMALIZATION_THRESHOLD, N_AVE);
    else
        d_magnitude << < (XLENGTH * ZLENGTH), YLENGTH >> > (d_data, dptr, DATASIZE, d_test, d_normalize, CONTRAST, N, UPPER_NORMALIZATION_THRESHOLD, LOWER_NORMALIZATION_THRESHOLD, N_AVE);

    if (N == N_AVE) { N = 1; }
    else { N = N + 1; }

    //d_test1 = d_test;
    cudaMemcpy(h_test, d_test, floatmem_size1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test1, d_normalize, floatmem_size1, cudaMemcpyDeviceToHost);

    //printf("%f", );
    if (0) {
        char buffer[50];
        sprintf(buffer, "%f", h_test[100]);
        MessageBox(0, buffer, "Title", MB_OK);
    }


    if (0) {
        ofstream myout;

        myout.open("FFT_Output3.dat");

        for (uint32_t i = 0; i < DATASIZE; i++) {

            //Print data to dat file:
            myout << i << " " << (float)h_test[i] << endl;
            //myout << i << " " << i] << endl;
            //printf("i = %d\n", i);

        }

        myout.close();
        //Display finish message:
        cout << "Finished!" << endl;

    }

    if (0) {
        //print_kernel << <1, 1 >> > (d_test);
    }
    //glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), dptr, GL_STREAM_DRAW);

    /*cudaMemcpy(h_normalize, d_normalize, floatmem_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_normalize, d_data, floatmem_size, cudaMemcpyDeviceToHost);

    normalize_max = *max_element(h_normalize, h_normalize + (DATASIZE - 1));

    cout << "Maximum FFT value: " << normalize_max << endl;

    d_normalization << < (XLENGTH * YLENGTH), ZLENGTH >> > (h_normalize, dptr, DATASIZE, normalize_max, d_test);*/

    //TEST

    //printf("h_test 2= %.20f\n", h_test[500]);
    //cudaMemcpy(h_test, (float*)d_frame, floatmem_size*2, cudaMemcpyDeviceToHost);

    //DEBUG MODE:


    long magnitudeFinishTime = clock();

    //Tell CPU to wait for GPU to finish:
    cudaDeviceSynchronize();

    //Copy device memory to host
    //cudaMemcpy(h_test, d_test, floatmem_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_sharedresult, dptr, floatmem_size, cudaMemcpyDeviceToHost);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);

    long processFinishTime = clock();

    //for (int i = 0; i < 1; i++) { //TOTAL
     //   cout << h_test[500] << endl;
    //}

    //Timing report:
    //cout << "Transform took " << (transformFinishTime - transformStartTime) << " millis" << endl;
    //cout << "Transform + magnitude calculation took " << (magnitudeFinishTime - transformStartTime) << " millis" << endl;
    //cout << "Total process calculation took " << (processFinishTime - startTime) << " millis" << endl;

}

void init() {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // register callbacks
    if (0) {
        glutIgnoreKeyRepeat(1);
        //glutKeyboardFunc(processNormalKeys);
        //glutSpecialFunc(pressKey);
        //glutSpecialUpFunc(releaseKey);
       // glutMouseFunc(mouseButton);
      //  glutMotionFunc(mouseMove);
    }
}

GLint WindowID0, WindowID1, WindowID2, WindowID3, WindowID4;

bool initGL(int* argc, char** argv)
{


    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width * 3, window_height * 3);
    glutInitWindowPosition(OCT_COMNTROL_width, 0);
    WindowID0 = glutCreateWindow("OCT 3D view");

    if (0) {
        glutSetWindow(WindowID1);
        glutDisplayFunc(display1);
        //glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        //glutKeyboardFunc(keyboard);
        //glutMotionFunc(motion);

       // WindowID3 = glutCreateSubWindow(WindowID1, window_width * 2, window_height * 2, window_width, window_height);
        //glutSetWindow(WindowID3);
        //glutDisplayFunc(display1);

        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

        // initialize necessary OpenGL extensions
        if (!isGLVersionSupported(2, 0))
        {
            fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            fflush(stderr);
            return false;
        }


        glGenBuffers(1, &vbopositions);
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_STATIC_DRAW);

        glGenBuffers(1, &vbocolors);
        glBindBuffer(GL_ARRAY_BUFFER, vbocolors);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_DYNAMIC_DRAW);

        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // Tell code which buffer represent positions and which represent color:
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glColorPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

        if (0) {
            glClipPlane(GL_CLIP_PLANE0, eqn);
            glEnable(GL_CLIP_PLANE0);
            glClipPlane(GL_CLIP_PLANE1, eqn2);
            glEnable(GL_CLIP_PLANE1);
            glClipPlane(GL_CLIP_PLANE2, eqn3);
            //glEnable(GL_CLIP_PLANE2);
            glClipPlane(GL_CLIP_PLANE3, eqn4);
            //glEnable(GL_CLIP_PLANE3);
            glClipPlane(GL_CLIP_PLANE4, eqn5);
            //glEnable(GL_CLIP_PLANE4);
            glClipPlane(GL_CLIP_PLANE5, eqn6);
            //glEnable(GL_CLIP_PLANE5);
        }
    }
    //glutKeyboardFunc(keyboard);
    //glutMouseFunc(mouse);
    //glutMotionFunc(motion);

    if (0) {
        //glutInit(argc, argv);
        //glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(window_width, window_height);
        glutInitWindowPosition(OCT_COMNTROL_width + window_width, 0);
        WindowID2 = glutCreateWindow("OCT X view");
        glutSetWindow(WindowID2);
        glutDisplayFunc(display);

        glutInitWindowSize(window_width, window_height);
        glutInitWindowPosition(OCT_COMNTROL_width, window_height + 30);
        WindowID3 = glutCreateWindow("OCT Y view");
        //glutDisplayFunc(display1);
    }

    if (1) {
        WindowID2 = glutCreateSubWindow(WindowID0, 2 * window_width, 0, window_width, window_height);// window_width * 2, window_height * 2, window_width, window_height);
        //glOrtho(0.0f, window_height, window_height, 0.0f, 0.0f, 1.0f);
        //glutSetWindow(WindowID2);
        glutDisplayFunc(display1);
        //glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        //glutKeyboardFunc(keyboard);
        //glutMotionFunc(motion);

       // WindowID3 = glutCreateSubWindow(WindowID1, window_width * 2, window_height * 2, window_width, window_height);
        //glutSetWindow(WindowID3);
        //glutDisplayFunc(display1);

        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

        // initialize necessary OpenGL extensions
        if (!isGLVersionSupported(2, 0))
        {
            fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            fflush(stderr);
            return false;
        }



        glGenBuffers(1, &vbopositions);
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_STATIC_DRAW);

        glGenBuffers(1, &vbocolors);
        glBindBuffer(GL_ARRAY_BUFFER, vbocolors);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_STREAM_DRAW);

        createVBO(&vbo1, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // Tell code which buffer represent positions and which represent color:
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vbo1);
        glColorPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

        if (0) {
            glClipPlane(GL_CLIP_PLANE0, eqn);
            glEnable(GL_CLIP_PLANE0);
            glClipPlane(GL_CLIP_PLANE1, eqn2);
            glEnable(GL_CLIP_PLANE1);
            glClipPlane(GL_CLIP_PLANE2, eqn3);
            //glEnable(GL_CLIP_PLANE2);
            glClipPlane(GL_CLIP_PLANE3, eqn4);
            //glEnable(GL_CLIP_PLANE3);
            glClipPlane(GL_CLIP_PLANE4, eqn5);
            //glEnable(GL_CLIP_PLANE4);
            glClipPlane(GL_CLIP_PLANE5, eqn6);
            //glEnable(GL_CLIP_PLANE5);
        }
        //*/
    }


    if (1) {
        //glutInitWindowSize(window_width, window_height);
        //glutInitWindowPosition(OCT_COMNTROL_width + window_width, window_height + 30);
        WindowID1 = glutCreateSubWindow(WindowID0, 0, 0, window_width * 2, window_height * 2);
        //WindowID4 = glutCreateWindow("OCT Z view");
        //glutSetWindow(WindowID2);
        glutDisplayFunc(display);

        //glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        //glutKeyboardFunc(keyboard);
        //glutMotionFunc(motion);


        //glutSetWindow(WindowID3);


        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

        // initialize necessary OpenGL extensions
        if (!isGLVersionSupported(2, 0))
        {
            fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            fflush(stderr);
            return false;
        }


        glGenBuffers(1, &vbopositions);
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_STATIC_DRAW);

        glGenBuffers(1, &vbocolors);
        glBindBuffer(GL_ARRAY_BUFFER, vbocolors);
        glBufferData(GL_ARRAY_BUFFER, TOTAL * sizeof(float), positions, GL_STREAM_DRAW);

        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // Tell code which buffer represent positions and which represent color:
        glBindBuffer(GL_ARRAY_BUFFER, vbopositions);
        glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glColorPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

        if (0) {
            glClipPlane(GL_CLIP_PLANE0, eqn);
            glEnable(GL_CLIP_PLANE0);
            glClipPlane(GL_CLIP_PLANE1, eqn2);
            glEnable(GL_CLIP_PLANE1);
            glClipPlane(GL_CLIP_PLANE2, eqn3);
            //glEnable(GL_CLIP_PLANE2);
            glClipPlane(GL_CLIP_PLANE3, eqn4);
            //glEnable(GL_CLIP_PLANE3);
            glClipPlane(GL_CLIP_PLANE4, eqn5);
            //glEnable(GL_CLIP_PLANE4);
            glClipPlane(GL_CLIP_PLANE5, eqn6);
            //glEnable(GL_CLIP_PLANE5);
        }


    }


    // start rendering mainloop
    glutMainLoop();

    return true;

}

void display()
{


    glutSetWindow(WindowID1);
    //TEST:
    //Print data:
        /*for (int i = 0; i < (20500); i++) {
            //Must change to magnitude:
            cout << h_data[i].x << " + " << h_data[i].y << endl;
            cout << hypotf(h_data[i].x, h_data[i].y) << endl;
            cout << h_test[i] << endl;
        }*/


        // For debugging purposes only:
    if (false) {
        ofstream myout1;

        //Plot first pulse:
        myout1.open("CUFFT_data_trimmed.dat");

        for (int i = 0; i < ((floor(framing_threads / 2)) * framing_elements); i++) {
            //for (int i = 0; i < (((floor(framing_threads / 2)) * framing_elements)*2); i++) {

                //cout << (sizeof(h_test) / sizeof(h_test[1])) << endl;

                //Magnitude:
                //myout1 << " " << hypot(h_data[i].x, h_data[i].y);
                //myout1 << i << " " << h_data[i].x;
            myout1 << h_test[i] << endl;

            //Imaginary:
            //cout << h_calcdata[i].x << " + " << h_calcdata[i].y << endl;
        }

        myout1.close();
    }

    //Timing:
    //cudaDeviceSynchronize();
    //drawendTime = clock();
    //cout << "Total process took " << (drawendTime - drawstartTime) << " millis" << endl;
    //drawstartTime = clock();


    drawendTime = clock();
    //test_val = (float)sampleValues[500];
    //cout << test_val << endl;
    //cout << "Frame took " << (drawendTime - drawstartTime) << " millis" << endl;
    printf("%.1f FPS\n", 1.0 / (float)((drawendTime - drawstartTime) / 1000.0));
    drawstartTime = clock();


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 0.5, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.5, 0.0);
    glRotatef(90.0, 0.0, 0.0, 0.5);



    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //glColor3f(0.0, 0.0, 0.1);
    //glDrawArrays(GL_POINTS, 0, draw_number);
    ///glDisableClientState(GL_VERTEX_ARRAY);
    //glDisableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, draw_number);

    if (0) {
        //glDisable(GL_CLIP_PLANE0);
        //eqn[3] = -dx / 1000;
        glClipPlane(GL_CLIP_PLANE0, eqn);
        //glEnable(GL_CLIP_PLANE0);
        glClipPlane(GL_CLIP_PLANE1, eqn2);
        //glEnable(GL_CLIP_PLANE1);
        glClipPlane(GL_CLIP_PLANE2, eqn3);
        //glEnable(GL_CLIP_PLANE2);
        glClipPlane(GL_CLIP_PLANE3, eqn4);
        //glEnable(GL_CLIP_PLANE3);
        glClipPlane(GL_CLIP_PLANE4, eqn5);
        //glEnable(GL_CLIP_PLANE4);
        glClipPlane(GL_CLIP_PLANE5, eqn6);
        //glEnable(GL_CLIP_PLANE5);
    }

    // default initialization
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(40, (GLfloat)window_width / (GLfloat)window_height, 0, 100.0); //Determines the clipping range for how much you can see.
    //glRotatef(10.0, 0.5, 0.0, 0.0);
    //glRotatef(10.0, 0.0, 0.5, 0.0);

    glutSwapBuffers();

    // Grab next dataset from oscilloscope:
    //status = viRead(vi, (ViBuf)buffer, (VISA_BUFF_SIZE + VISA_BUFF_OFFSET), &retCnt);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), h_test, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, COLORTOTAL * sizeof(float), h_test);
    //printf("color = %f\n",(float) colors[500]);



    glFlush();

    //glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), NULL, GL_DYNAMIC_COPY);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);


    //Use to prove that data can be updated frame by frame:
    //for (int i = 0; i < DATASIZE; i++) {
    //	h_data[i].x = 0; //Add n to get all elements.
    //	h_data[i].y = 0;
    //}

}


void display1()
{
    glutSetWindow(WindowID2);
    //TEST:
    //Print data:
        /*for (int i = 0; i < (20500); i++) {
            //Must change to magnitude:
            cout << h_data[i].x << " + " << h_data[i].y << endl;
            cout << hypotf(h_data[i].x, h_data[i].y) << endl;
            cout << h_test[i] << endl;
        }*/


        // For debugging purposes only:
    if (false) {
        ofstream myout1;

        //Plot first pulse:
        myout1.open("CUFFT_data_trimmed.dat");

        for (int i = 0; i < ((floor(framing_threads / 2)) * framing_elements); i++) {
            //for (int i = 0; i < (((floor(framing_threads / 2)) * framing_elements)*2); i++) {

                //cout << (sizeof(h_test) / sizeof(h_test[1])) << endl;

                //Magnitude:
                //myout1 << " " << hypot(h_data[i].x, h_data[i].y);
                //myout1 << i << " " << h_data[i].x;
            myout1 << h_test[i] << endl;

            //Imaginary:
            //cout << h_calcdata[i].x << " + " << h_calcdata[i].y << endl;
        }

        myout1.close();
    }

    //Timing:
    //cudaDeviceSynchronize();
    //drawendTime = clock();
    //cout << "Total process took " << (drawendTime - drawstartTime) << " millis" << endl;
    //drawstartTime = clock();


    drawendTime = clock();
    //test_val = (float)sampleValues[500];
    //cout << test_val << endl;
    //cout << "Frame took " << (drawendTime - drawstartTime) << " millis" << endl;
    printf("%.1f FPS\n", 1.0 / (float)((drawendTime - drawstartTime) / 1000.0));
    drawstartTime = clock();


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, translate_z);
    glRotatef(rotate_x, 0.5, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.5, 0.0);
    glRotatef(90.0, 0.0, 0.0, 0.5);



    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //glColor3f(0.0, 0.0, 0.1);
    //glDrawArrays(GL_POINTS, 0, draw_number);
    ///glDisableClientState(GL_VERTEX_ARRAY);
    //glDisableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, draw_number);

    if (0) {
        //glDisable(GL_CLIP_PLANE0);
        //eqn[3] = -dx / 1000;
        glClipPlane(GL_CLIP_PLANE0, eqn);
        //glEnable(GL_CLIP_PLANE0);
        glClipPlane(GL_CLIP_PLANE1, eqn2);
        //glEnable(GL_CLIP_PLANE1);
        glClipPlane(GL_CLIP_PLANE2, eqn3);
        //glEnable(GL_CLIP_PLANE2);
        glClipPlane(GL_CLIP_PLANE3, eqn4);
        //glEnable(GL_CLIP_PLANE3);
        glClipPlane(GL_CLIP_PLANE4, eqn5);
        //glEnable(GL_CLIP_PLANE4);
        glClipPlane(GL_CLIP_PLANE5, eqn6);
        //glEnable(GL_CLIP_PLANE5);
    }

    // default initialization
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(40, (GLfloat)window_width / (GLfloat)window_height, 0.0, 100.0); //Determines the clipping range for how much you can see.
    //glRotatef(10.0, 0.5, 0.0, 0.0);
    //glRotatef(10.0, 0.0, 0.5, 0.0);

    glutSwapBuffers();

    // Grab next dataset from oscilloscope:
    //status = viRead(vi, (ViBuf)buffer, (VISA_BUFF_SIZE + VISA_BUFF_OFFSET), &retCnt);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), h_test, GL_STATIC_DRAW);

    //printf("color = %f\n",(float) colors[500]);



    glFlush();

    //glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), NULL, GL_DYNAMIC_COPY);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);


    //Use to prove that data can be updated frame by frame:
    //for (int i = 0; i < DATASIZE; i++) {
    //	h_data[i].x = 0; //Add n to get all elements.
    //	h_data[i].y = 0;
    //}

}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler (MAKE SURE CAPS LOCK IS OFF IN ORDER TO TOGGLE BUTTONS)
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        /*case (38):
            cout << "hi" << draw_number << endl;
            if (draw_number <= LENGTH * LENGTH * LENGTH) {
                draw_number = draw_number + (LENGTH * LENGTH);
        }
            else {
                ;
            }

        case (32):
            if (draw_number >= 0) {
            draw_number = draw_number - (LENGTH * LENGTH);
            }
            else {
                ;
            }*/
    case (122): // if z key is pressed
        current_key = 122; // 
        return;
    case (107): // if k key is pressed
        if (z_plane) {
            glDisable(GL_CLIP_PLANE0);
            glDisable(GL_CLIP_PLANE1);
            z_plane = false;
        }
        else {
            glEnable(GL_CLIP_PLANE0);
            glEnable(GL_CLIP_PLANE1);
            z_plane = true;
        }
        return;

    case (121): // if y key is pressed
        current_key = 121; // 
        return;
    case (106): // if j key is pressed
        if (y_plane) {
            glDisable(GL_CLIP_PLANE2);
            glDisable(GL_CLIP_PLANE3);
            y_plane = false;
        }
        else {
            glEnable(GL_CLIP_PLANE2);
            glEnable(GL_CLIP_PLANE3);
            y_plane = true;
        }
        return;

    case (120): // if x key is pressed
        current_key = 120; // 
        return;
    case (105): // if i key is pressed
        if (x_plane) {
            glDisable(GL_CLIP_PLANE4);
            glDisable(GL_CLIP_PLANE5);
            x_plane = false;
        }
        else {
            glEnable(GL_CLIP_PLANE4);
            glEnable(GL_CLIP_PLANE5);
            x_plane = true;
        }
        return;
    case (27): // if esc key is pressed
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());

        //Destroy plans:
        cufftDestroy(plan);

        //Free data:
        //cudaFree(d_data_old);
        cudaFree(d_data);
        cudaFree(d_sharedresult);
        cudaFree(d_normalize);
        //free(h_calcdata);
        free(h_sharedresult);
        //free(h_data);
        cudaFreeHost(h_data);
        //worker.join();


        delete[] positions;
        delete[] colors;
        delete[] ahead;
        delete[] waveform;


        cout << "---------------------------------VISUALIZATION HAS ENDED----------------------------------" << endl;

        return;
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
        current_key = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    //float dx, dy;
    if (current_key == 122) { // KEY = 'z'
        //cout << "We made it here" << endl;
        dx = (float)(x - mouse_old_x);
        //dy = (float)(y - mouse_old_y);
        cout << "Dx:                                             " << dx << endl;
        eqn[3] = -dx / 1000;
        eqn2[3] = (dx / 1000) + 0.1; // for one layer use 0.01
    }
    else if (current_key == 121) { // KEY = 'y'
        //cout << "We made it here" << endl;
        //dx = (float)(x - mouse_old_x);
        dy = (float)(y - mouse_old_y);
        cout << "Dy:                                             " << dy << endl;
        eqn3[3] = dy / 1000;
        eqn4[3] = (-dy / 1000) + 0.1; // for one layer use 0.01
    }
    else if (current_key == 120) { // KEY = 'x'
        //cout << "We made it here" << endl;
        dx = (float)(x - mouse_old_x);
        //dy = (float)(y - mouse_old_y);
        cout << "Dx:                                             " << dx << endl;
        eqn5[3] = -dx / 1000;
        eqn6[3] = (dx / 1000) + 0.1; // for one layer use 0.01
    }
    else {
        dx = (float)(x - mouse_old_x);
        dy = (float)(y - mouse_old_y);

        if (mouse_buttons & 1)
        {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        }
        else if (mouse_buttons & 4)
        {
            translate_z += dy * 0.01f;
        }

        mouse_old_x = x;
        mouse_old_y = y;
    }
}

void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    //unsigned int size = mesh_width * mesh_height * sizeof(float);
    //glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), colors, GL_DYNAMIC_COPY);
    glBufferData(GL_ARRAY_BUFFER, COLORTOTAL * sizeof(float), colors, GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();

        //Reset timer:
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


void kernel(int argc, char* argv[]) {
    // Allocate Pinned Memory for buffer variable:
    cudaHostAlloc((void**)&h_data, mem_size, cudaHostAllocMapped); //can be commented out
    //cudaHostAlloc((void**)&pSamples, sizeof(uint16_t) * (DATASIZE), cudaHostAllocMapped);
    //cudaHostGetDevicePointer(&d_pinned_data, sampleValues_copy, 0);


    //3D example:
    float x = 1;
    float y = 1;
    float z = 1;
    float xx = YLENGTH;
    float yy = ZLENGTH; //ZLENGTH AND XLENGTH MAY NEED TO BE SWITCHED
    float zz = XLENGTH;

    for (int i = 0; i < TOTAL; i += 3)
    {
        positions[i] = ((x / xx) * 2.0f - 1.0f) + .25f;//.25f;
        positions[i + 1] = ((y / yy) * 2.0f - 1.0f) + .25f;//.25f;
        positions[i + 2] = ((z / zz) * 2.0f - 1.0f);//;

        //cout << x << ", " << y << ", " << z << endl;

        x++;

        if (x == YLENGTH) //XLENGTH) //z
        {
            x = 0;
        }
        if (x == 0)
        {
            y++;
        }
        if (y == ZLENGTH) //YLENGTH)
        {
            y = 0;
        }
        if (y == 0 && x == 0) //x
        {
            z++;
        }
    }

    //INITIALIZE COLORS:
    if (0) {
        for (int i = 0; i < COLORTOTAL; i++) {
            colors[i] = 0.5f;// 0.0f;
        }
    }


    /*

    -----------------------------------SETUP PHASE-----------------------------------------

    */

    //***.dat file must be arranged where column data is going from left to right***

    //int THREADS = 128;
    //int BLOCKS = (round((DATASIZE / THREADS)) + 100);


    //SET TO TRUE FOR DEBUG MODE (creates initial data before transform for plotting):
    bool DEBUG = false;
    bool JUST_THIS = true;     //When true, code will only put pre FFT data into dat file and will not continue to evaluate FFT.
    int START_SAMPLE_SIZE = 0; //Start point range
    int END_SAMPLE_SIZE = 40000000; //End point range.


    //Override for debugging:
    determinedPeriod = postTriggerSamples;

    /*

    -------------------------MEMORY ALLOCATION AND PLANNING PHASE---------------------

    */

    // Allocate device memory for signal after transform:

    cudaMalloc((void**)&d_data, mem_size);

    // Allocate device memory for result data:
    cudaMalloc((void**)&d_sharedresult, floatmem_size);

    // Allocate device memory for normalize data:
    cudaMalloc((void**)&d_normalize, floatmem_size1);

    //TEST:
    cudaMalloc((void**)&d_test, floatmem_size1);
    cudaMalloc((void**)&d_test1, floatmem_size1);
    cudaMalloc((void**)&d_frame, mem_size);
    cudaMalloc((void**)&d_waveform, floatmem_size);
    cudaMalloc((void**)&d_waveform1, floatmem_size);

    cudaMalloc(&d_interleaved, 2 * floatmem_size);
    //cudaMemcpy(d_waveform, waveform, floatmem_size, cudaMemcpyHostToDevice);

    //Allocate host memory for result data:
    //cudaMalloc((void**)&d_3D_data, floatmem_size);

    // Allocate device memory for signal before transform:
    //cudaMalloc((void**)&d_data_old, mem_size);

    //CUFFT1D plan:
    //cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH); // ARG 2 AND 4 MUST EQUATE TO TOTAL NUMBER OF DATA POINTS IN D_DATA (ARG2 * ARG4 = LENGTH(D_DATA))

    //CUFFT1Dmany plan (supports multiple batch):
    if (0) {
        printf("%d\n", nn[0]);
        printf("%d\n", nn[1]);
        printf("%d\n", NX);
        printf("%d\n", BATCH);
        printf("%d\n", inembed[0]);
    }

    cufftPlanMany(&plan, 1, nn, inembed, 1, NX, onembed, 1, NX, CUFFT_C2C, (int)BATCH);

    /*

    ---------------------------------------PROCESSING AND VISUALIZATION PHASE (WHILE LOOP)---------------------------------

    */

    //BEGIN TRANSFORM SETUP:
    //long si = clock();
        //Place data in Complex:
    /*for (int i = 0; i < DATASIZE; i++) {
        h_data[i].x = names[i]; //Add n to get all elements.
        h_data[i].y = 0;
    }*/
    int k = 0;
    /*for (int i = 0; i < DATASIZE; i = std::round(i + determinedPeriod)) {
        for (int j = i; j < std::round(i + determinedPeriod); j++) {
            h_data[k].x = names[k]; //Add n to get all elements.
            h_data[k].y = 0;
            k++;
        }
        cout << k << endl;
        //cout << i << " to " << std::round(i + determinedPeriod) << endl;
    }*/

    //determinedPeriod = 410.799836266885;

    // GPU Version: // NEED 2D BLOCKS AND THREADS FOR THE GPU FUNCTION
    framing_elements = std::floor(recordsPerBuffer * buffersPerAcquisition);// (std::floor(DATASIZE / determinedPeriod) - 1);
    //int kNum = (std::floor(n * determinedPeriod) + std::floor(determinedPeriod));
    framing_threads = std::floor(determinedPeriod);
    framing_blocks = framing_elements;
    //dim3 grid(BATCH, BATCH);
    //dim3 block(framing_threads, framing_threads);
    //cudaMemcpy(d_frametest, h_data, mem_size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();

    //cout << sampleValues << endl;
    //cudaMemcpy(d_waveform, (float*) sampleValues, sizeof(float) * (DATASIZE), cudaMemcpyHostToDevice); // straight from the buffer //NEED TO MAKE D_WAVEFORM A VIInt8Buf data type 
    //cudaMemcpy(d_waveform, d_pinned_data, floatmem_size, cudaMemcpyHostToDevice); // pinned data form
    //cudaMemcpy(d_waveform, next_frame, floatmem_size, cudaMemcpyHostToDevice);
    //d_framing << <framing_blocks, framing_threads >> > (determinedPeriod, d_waveform, (cufftComplex*)d_frame, framing_elements); //framing_elements
    //d_framing1 << <framing_blocks, framing_threads >> > (d_waveform, d_waveform1);

    //cudaDeviceSynchronize();


    //DEBUG MODE:
    if (DEBUG == true) {
        ofstream myout;

        myout.open("FFT_Output.dat");

        for (int i = START_SAMPLE_SIZE; i < END_SAMPLE_SIZE; i++) {

            //Print data to dat file:
            myout << i << " " << h_data[i].x << endl;

        }

        myout.close();

        //if (JUST_THIS == true) {
            //return 0;
        //}

        //Display finish message:
        cout << "Finished!" << endl;

    }

    // Start glut loop:
    runTest(argc, argv, NULL);

    //********Nothing after this gets evaluated, must put any ending calls in the esc button event*******
    /*

    ----------------------------------------DEBUG AND CHECK OPTION SECTION-------------------------------------

    */

    //Print data:
    cout << "FFT: " << endl;
    for (int i = 0; i < 10; i++) {

        //Magnitude:
        cout << h_sharedresult[i] << endl;

        //Imaginary:
        //cout << h_calcdata[i].x << " + " << h_calcdata[i].y << endl;
    }

    /*

    ----------------------------------------------CLEAN/FREE SECTION--------------------------------------------

    */

    //Destroy plans:
    cufftDestroy(plan);

    //Free data:
    //cudaFree(d_data_old);
    cudaFree(d_data);
    cudaFree(d_sharedresult);
    cudaFree(d_interleaved);

    //free(h_calcdata);
    free(h_sharedresult);
    free(h_data);

    delete[] positions;
    delete[] colors;

    cout << "---------------------------------VISUALIZATION HAS ENDED----------------------------------" << endl;


}

