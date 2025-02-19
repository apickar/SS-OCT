// This is a part of the Microsoft Foundation Classes C++ library.
// Copyright (C) Microsoft Corporation
// All rights reserved.
//
// This source code is only intended as a supplement to the
// Microsoft Foundation Classes Reference and related
// electronic documentation provided with the library.
// See these sources for detailed information regarding the
// Microsoft Foundation Classes product.

#include "stdafx.h"
#include "Slider.h"

#include "MainFrm.h"
#include "SliderDoc.h"
#include "SliderView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


#include <thread>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "AlazarError.h"
#include "AlazarApi.h"
#include "AlazarCmd.h"
#include <chrono>
#include "cuda_kernel.cuh"
#include "common.h"
using namespace std;



#ifdef _WIN32
#include <conio.h>
#else // ifndef _WIN32
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define TRUE  1
#define FALSE 0

#define _snprintf snprintf


//#ifdef __cplusplus extern "C"  #endif 


inline U32 GetTickCount(void);
inline void Sleep(U32 dwTime_ms);
inline int _kbhit(void);
inline int GetLastError();
#endif // ifndef _WIN32
#define BYTE U8
// TODO: Select the number of DMA buffers to allocate.
#define BUFFER_COUNT 4 // DMA buffers in rotation
HANDLE boardHandle = AlazarGetBoardBySystemID(1, 1);
// Globals variables
U16* BufferArray[BUFFER_COUNT] = { NULL };
double samplesPerSec = 0.0;
// There are no pre-trigger samples in NPT mode
const U32 preTriggerSamples = 0;
// TODO: Select the number of post-trigger samples per record
extern int postTriggerSamples = 2000;// 1000;//24960; // for sure a ot less
// TODO: Specify the number of records per DMA buffer
extern int recordsPerBuffer = 100;// 160; // A-scan per B-scan //40000
// TODO: Specify the total number of buffers to capture
extern int buffersPerAcquisition = 100;//63; // B x C scans //10000
int OCT_COMNTROL_width;
int VOLTAGE_LEVEL_idx1 = 7; //400 mV
int VOLTAGE_LEVEL_idx = 7; //400 mV
const U32  samplesPerBuffer = postTriggerSamples * recordsPerBuffer;
const double samplesPerAcquasition = postTriggerSamples * recordsPerBuffer * buffersPerAcquisition;
float* sampleValues = new float[samplesPerAcquasition];
float* sampleValues_copy = new float[samplesPerAcquasition];
U16* pBuffer = BufferArray[0];
int* pSamples = (int*)pBuffer;
int ACQUIRE_RUN = 1;

int N_AVE = 1;

thread worker;

// Forward declarations
BOOL ConfigureBoard(HANDLE boardHandle, int VOLTAGE_LEVEL_idx1);
void AcquireData();


double SampleToVoltsU16(U16 sampleValue, double inputRange_volts)
{ // AlazarTech digitizers are calibrated as follows 
	double codeZero = (double)USHRT_MAX / 2;
	double codeRange = (double)USHRT_MAX / 2;
	// Convert sample code to volts 
	double sampleVolts = inputRange_volts * ((double)(sampleValue - codeZero) / codeRange);
	return sampleVolts;
}



//-------------------------------------------------------------------------------------------------
//
// Function    :  ConfigureBoard
//
// Description :  Configure sample rate, input, and trigger settings
//
//-------------------------------------------------------------------------------------------------

BOOL ConfigureBoard(HANDLE boardHandle, int VOLTAGE_LEVEL_idx1)
{
    RETURN_CODE retCode;

    // TODO: Specify the sample rate (see sample rate id below)



    // TODO: Select clock parameters as required to generate this sample rate.
    //
    // For example: if samplesPerSec is 100.e6 (100 MS/s), then:
    // - select clock source INTERNAL_CLOCK and sample rate SAMPLE_RATE_100MSPS
    // - select clock source FAST_EXTERNAL_CLOCK, sample rate SAMPLE_RATE_USER_DEF, and connect a
    //   100 MHz signal to the EXT CLK BNC connector.
    bool kClock = true;
    if (kClock) {
        samplesPerSec = 250000000.0;
        retCode = AlazarSetCaptureClock(boardHandle,
            FAST_EXTERNAL_CLOCK, // EXTERNAL_CLOCK  FAST_EXTERNAL_CLOCK
            SAMPLE_RATE_USER_DEF, //SAMPLE_RATE_USER_DEF,   SAMPLE_RATE_100MSPS
            CLOCK_EDGE_RISING,
            0);
        if (retCode != ApiSuccess)
        {
            printf("Error: AlazarSetCaptureClock failed -- %s\n", AlazarErrorToText(retCode));
            return FALSE;
        }
    }
    else {
        samplesPerSec = 100000000.0;
        retCode = AlazarSetCaptureClock(boardHandle,
            INTERNAL_CLOCK,
            SAMPLE_RATE_100MSPS,
            CLOCK_EDGE_RISING,
            0);
        if (retCode != ApiSuccess)
        {
            printf("Error: AlazarSetCaptureClock failed -- %s\n", AlazarErrorToText(retCode));
            return FALSE;
        }
    }



    // TODO: Select channel A input parameters as required

    //INPUT_RANGE_PM_20_MV; INPUT_RANGE_PM_40_MV; INPUT_RANGE_PM_50_MV; INPUT_RANGE_PM_80_MV; INPUT_RANGE_PM_100_MV; INPUT_RANGE_PM_200_MV; INPUT_RANGE_PM_400_MV; INPUT_RANGE_PM_500_MV; INPUT_RANGE_PM_800_MV; INPUT_RANGE_PM_1_V
   
    //char INPUT_RANGE = "INPUT_RANGE_PM_400_MV";
    retCode = AlazarInputControlEx(boardHandle,
        CHANNEL_A,
        DC_COUPLING,
        VOLTAGE_LEVEL_idx1,
        IMPEDANCE_50_OHM);

    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarInputControlEx failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }


    // TODO: Select channel B input parameters as required
    retCode = AlazarInputControlEx(boardHandle,
        CHANNEL_B,
        DC_COUPLING,
        INPUT_RANGE_PM_400_MV,
        IMPEDANCE_50_OHM);
    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarInputControlEx failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }

    // TODO: Select trigger inputs and levels as required
    retCode = AlazarSetTriggerOperation(boardHandle,
        TRIG_ENGINE_OP_J,
        TRIG_ENGINE_J,
        TRIG_EXTERNAL,
        TRIGGER_SLOPE_POSITIVE,
        128,
        TRIG_ENGINE_K,
        TRIG_DISABLE,
        TRIGGER_SLOPE_POSITIVE,
        128); //128 128
    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarSetTriggerOperation failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }

    // TODO: Select external trigger parameters as required

    retCode = AlazarSetExternalTrigger(boardHandle,
        DC_COUPLING,
        ETR_TTL);

    // TODO: Set trigger delay as required.

    double triggerDelay_sec = 0;
    U32 triggerDelay_samples = (U32)(3087);// (triggerDelay_sec * samplesPerSec + 0.5);   with kclk 1000 points: 167, without kclk 1000: 696, with kclk 2000 points: 3087
    retCode = AlazarSetTriggerDelay(boardHandle, triggerDelay_samples);
    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarSetTriggerDelay failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }

    // TODO: Set trigger timeout as required.

    // NOTE:
    //
    // The board will wait for a for this amount of time for a trigger event. If a trigger event
    // does not arrive, then the board will automatically trigger. Set the trigger timeout value to
    // 0 to force the board to wait forever for a trigger event.
    //
    // IMPORTANT:
    //
    // The trigger timeout value should be set to zero after appropriate trigger parameters have
    // been determined, otherwise the board may trigger if the timeout interval expires before a
    // hardware trigger event arrives.

    retCode = AlazarSetTriggerTimeOut(boardHandle, 0);
    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarSetTriggerTimeOut failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }

    // TODO: Configure AUX I/O connector as required

    retCode = AlazarConfigureAuxIO(boardHandle, AUX_IN_TRIGGER_ENABLE, TRIGGER_SLOPE_POSITIVE); //AUX_IN_AUXILIARY  , GET_AUX_INPUT_LEVEL ); // 1 ?  

   //SET_BUFFERS_PER_TRIGGER_ENABLE


    if (retCode != ApiSuccess)
    {
        printf("Error: AlazarConfigureAuxIO failed -- %s\n", AlazarErrorToText(retCode));
        return FALSE;
    }
    /*
    AUX_IN_TRIGGER_ENABLE edge of a pulse to the AUX I/O connector as an AutoDMA trigger enable signal.

    retCode = AlazarConfigureAuxIO(boardHandle,  AUX_IN_TRIGGER_ENABLE,  TRIGGER_SLOPE_POSITIVE);

    */

    return TRUE;
}



//-------------------------------------------------------------------------------------------------
//
// Function    :  AcquireData
//
// Description :  Perform an acquisition, optionally saving data to file.
//
//-------------------------------------------------------------------------------------------------

void AcquireData()
{

    while (ACQUIRE_RUN)
    {
        try
        {
            //cout << "Next round" << endl;
        //vector<double> vv2(samplesPerBuffer);
        // TODO: Select which channels to capture (A, B, or both)
            U32 channelMask = CHANNEL_A;

            // TODO: Select if you wish to save the sample data to a file
            BOOL saveData = true; // false;

            // Calculate the number of enabled channels from the channel mask
            int channelCount = 0;
            int channelsPerBoard = 2;
            for (int channel = 0; channel < channelsPerBoard; channel++)
            {
                U32 channelId = 1U << channel;
                if (channelMask & channelId)
                    channelCount++;
            }
            channelCount = 1;

            // Get the sample size in bits, and the on-board memory size in samples per channel
            U8 bitsPerSample;
            U32 maxSamplesPerChannel;
            RETURN_CODE retCode = AlazarGetChannelInfo(boardHandle, &maxSamplesPerChannel, &bitsPerSample);
            if (retCode != ApiSuccess)
            {
                printf("Error: AlazarGetChannelInfo failed -- %s\n", AlazarErrorToText(retCode));
                //return FALSE;
            }

            // Calculate the size of each DMA buffer in bytes
            float bytesPerSample = (float)((bitsPerSample + 7) / 8);
            U32 samplesPerRecord = preTriggerSamples + postTriggerSamples;
            U32 bytesPerRecord = (U32)(bytesPerSample * samplesPerRecord +
                0.5); // 0.5 compensates for double to integer conversion 
            U32 bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount;

            //Create our buffer

            //U16* our_buffer[buffersPerAcquisition][samplesPerBuffer] = { NULL };
            //std::vector<double*> our_buffer(samplesPerAcquasition,0);
            //our_buffer.prototype.fill(0);
            vector<double> our_buffer;
            //our_buffer.reserve(samplesPerAcquasition);
            // Create a data file if required
            FILE* fpData = NULL;



            if (saveData)
            {
                fpData = fopen("data.bin", "wb");//, std::fstream::trunc | std::fstream::out | std::fstream::app);
                if (fpData == NULL)
                {
                    printf("Error: Unable to create data file -- %u\n", GetLastError());
                    //return FALSE;
                }
            }





            // Allocate memory for DMA buffers
            BOOL success = TRUE;

            U32 bufferIndex;
            for (bufferIndex = 0; (bufferIndex < BUFFER_COUNT) && success; bufferIndex++)
            {
                // Allocate page aligned memory
                BufferArray[bufferIndex] =
                    (U16*)AlazarAllocBufferU16(boardHandle, bytesPerBuffer);
                if (BufferArray[bufferIndex] == NULL)
                {
                    printf("Error: Alloc %u bytes failed\n", bytesPerBuffer);
                    success = FALSE;
                }
            }


            // Configure the record size
            if (success)
            {
                retCode = AlazarSetRecordSize(boardHandle, preTriggerSamples, postTriggerSamples);
                if (retCode != ApiSuccess)
                {
                    printf("Error: AlazarSetRecordSize failed -- %s\n", AlazarErrorToText(retCode));
                    success = FALSE;
                }
            }

            // Configure the board to make an NPT AutoDMA acquisition
            if (success)
            {
                U32 recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition;

                U32 admaFlags = ADMA_EXTERNAL_STARTCAPTURE | ADMA_NPT | ADMA_FIFO_ONLY_STREAMING;

                retCode = AlazarBeforeAsyncRead(boardHandle, channelMask, -(long)preTriggerSamples,
                    samplesPerRecord, recordsPerBuffer, recordsPerAcquisition,
                    admaFlags);
                if (retCode != ApiSuccess)
                {
                    printf("Error: AlazarBeforeAsyncRead failed -- %s\n", AlazarErrorToText(retCode));
                    success = FALSE;
                }
            }

            U32 buffers_per = (U32)buffersPerAcquisition;
            retCode = AlazarSetParameterUL(boardHandle, CHANNEL_ALL, SET_BUFFERS_PER_TRIGGER_ENABLE, buffers_per);
            if (retCode != ApiSuccess)
            {
                printf("Error: SET_BUFFERS_PER_TRIGGER_ENABLE failed -- %s\n", AlazarErrorToText(retCode));
                //return FALSE;
            }

            // Add the buffers to a list of buffers available to be filled by the board

            for (bufferIndex = 0; (bufferIndex < BUFFER_COUNT) && success; bufferIndex++)
            {
                U16* pBuffer = BufferArray[bufferIndex];
                retCode = AlazarPostAsyncBuffer(boardHandle, pBuffer, bytesPerBuffer);
                if (retCode != ApiSuccess)
                {
                    printf("Error: AlazarPostAsyncBuffer %u failed -- %s\n", bufferIndex,
                        AlazarErrorToText(retCode));
                    success = FALSE;
                }
            }



            // Arm the board system to wait for a trigger event to begin the acquisition
            if (success)
            {
                retCode = AlazarStartCapture(boardHandle);
                if (retCode != ApiSuccess)
                {
                    printf("Error: AlazarStartCapture failed -- %s\n", AlazarErrorToText(retCode));
                    success = FALSE;
                }
            }


            if (success)
            {
                //printf("Capturing %d buffers ... press any key to abort\n", buffersPerAcquisition);

                U32 startTickCount = GetTickCount();
                U32 buffersCompleted = 0;
                INT64 bytesTransferred = 0;

                //U32 start_clock = GetTickCount();

                while (buffersCompleted < buffersPerAcquisition)
                {
                    //U32 save_start = GetTickCount();
                    //retCode = AlazarSetParameter(boardHandle, CHANNEL_B, GET_AUX_INPUT_LEVEL, aux_level??);
                   // printf("aux value --%ld\n", retCode);

                    //printf("aux value --%lu -- %ld\n",aux, aux_level);

                    // TODO: Set a buffer timeout that is longer than the time
                    //       required to capture all the records in one buffer.
                    U32 timeout_ms = 5000;

                    // Wait for the buffer at the head of the list of available buffers
                    // to be filled by the board.
                    bufferIndex = buffersCompleted % BUFFER_COUNT;
                    U16* pBuffer = BufferArray[bufferIndex];


                    retCode = AlazarWaitAsyncBufferComplete(boardHandle, pBuffer, timeout_ms);
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                    if (retCode != ApiSuccess)
                    {
                        printf("Error: AlazarWaitAsyncBufferComplete failed -- %s\n",
                            AlazarErrorToText(retCode));
                        success = FALSE;
                    }
                    U32 startTickCount = GetTickCount();

                    if (success)
                    {
                        // The buffer is full and has been removed from the list
                        // of buffers available for the board

                        buffersCompleted++;
                        bytesTransferred += bytesPerBuffer;

                        //vector<U16> vals(pBuffer, pBuffer + samplesPerBuffer);


                                        //double* vv2 = new double[tt];
                        //const void* pSamples = (void*)pBuffer;


                        if (1)
                        {
                            U16* pSamples = (U16*)pBuffer;

                            //U16* vv2 = new U16[tt];
                            for (U32 sample = (int)(0 + samplesPerBuffer * (buffersCompleted - 1)); sample < (samplesPerBuffer + samplesPerBuffer * (buffersCompleted - 1)); sample++) {
                                //U16 sampleValue = (U16) *pSamples++;
                                sampleValues[sample] = (float)*pSamples++; // ((U16)*pSamples++ - 28500) / 40;
                                //our_buffer.push_back(sampleValue);
                                //printf("sample value = %d ,  %X\n", sample, sampleValue);

                                //vv2[sample] = SampleToVoltsU16(sampleValue, 400.0);
                            }

                        }



                        // TODO: Process sample data in this buffer.

                        // NOTE:
                        //
                        // While you are processing this buffer, the board is already filling the next
                        // available buffer(s).
                        //
                        // You MUST finish processing this buffer and post it back to the board before
                        // the board fills all of its available DMA buffers and on-board memory.
                        //
                        // Samples are arranged in the buffer as follows: S0A, S0B, ..., S1A, S1B, ...
                        // with SXY the sample number X of channel Y.
                        //
                        // A 12-bit sample code is stored in the most significant bits of in each 16-bit
                        // sample value. 
                        // Sample codes are unsigned by default. As a result:
                        // - a sample code of 0x0000 represents a negative full scale input signal.
                        // - a sample code of 0x8000 represents a ~0V signal.
                        // - a sample code of 0xFFFF represents a positive full scale input signal.  

                        // -- SEND DATA TO GPU ---
                        //copy(vv2,vv2+(samplesPerBuffer*(buffersCompleted+1)), (double) (samplesPerBuffer * buffersCompleted));
                        //U32 save_start = GetTickCount();
                        //copy(vv2.begin(), vv2.end(), back_inserter(our_buffer));
                        //U32 save_end = GetTickCount();


                        //printf("Save Time -- %u ms\n", save_end - save_start);

                        //.copy(0,samplesPerRecord,buffersCompleted*samplesPerRecord);
                        //our_buffer[ii] = vv2;



                        if (0)
                        {
                            // Write record to file


                            size_t bytesWritten = fwrite(pSamples, sizeof(U8), bytesPerBuffer, fpData);
                            if (bytesWritten != bytesPerBuffer)
                            {
                                printf("Error: Write buffer %u failed -- %u\n", buffersCompleted,
                                    GetLastError());
                                success = FALSE;
                            }
                        }

                        //U32 end_clock = GetTickCount();
                        //double save_time = (end_clock - start_clock) * 1000;
                        //printf("SAVE TIME -- %u\n", save_time);

                        //DEBUG MODE:
                        if (0) {
                            ofstream myout;

                            myout.open("FFT_Output.dat");

                            for (U32 i = 0; i < samplesPerBuffer; i++) {

                                //Print data to dat file:
                                //myout << i << " " << (float)sampleValues[i] << endl;
                                myout << i << " " << pSamples[i] << endl;

                            }

                            myout.close();
                            //Display finish message:
                            cout << "Finished!" << endl;

                        }

                    }

                    // Add the buffer to the end of the list of available buffers.
                    if (success)
                    {
                        retCode = AlazarPostAsyncBuffer(boardHandle, pBuffer, bytesPerBuffer);
                        if (retCode != ApiSuccess)
                        {
                            printf("Error: AlazarPostAsyncBuffer failed -- %s\n",
                                AlazarErrorToText(retCode));
                            // success = FALSE;
                        }
                    }

                    // If the acquisition failed, exit the acquisition loop
                    if (!success)
                        break;

                    // If a key was pressed, exit the acquisition loop
                    if (_kbhit())
                    {
                        printf("Aborted...\n");
                        break;
                    }

                    // Display progress
                    //printf("Completed %u buffers\r", buffersCompleted);
                    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    //std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

                }

                //int arrSize = (sampleValues + 1) - &sampleValues;
                //printf("sampleValues length (CPU code) = %d\n", (int)(sizeof(sampleValues) / sizeof(float)));
                //printf("sampleValues length (CPU code) = %d\n", arrSize);

                sampleValues_copy = sampleValues;

                //DEBUG MODE:
                if (0) {
                    ofstream myout;

                    myout.open("FFT_Output.dat");

                    for (U32 i = 0; i < samplesPerAcquasition; i++) {

                        //Print data to dat file:
                        myout << i << " " << (float)sampleValues[i] << endl;
                        //myout << i << " " << i] << endl;

                    }

                    myout.close();
                    //Display finish message:
                    cout << "Finished!" << endl;

                }

                // Display results
                double transferTime_sec = (GetTickCount() - startTickCount) / 1000.;
                //printf("Capture completed in %.2lf sec\n", transferTime_sec);

                double buffersPerSec;
                double bytesPerSec;
                double recordsPerSec;
                U32 recordsTransferred = recordsPerBuffer * buffersCompleted;

                if (transferTime_sec > 0.)
                {
                    buffersPerSec = buffersCompleted / transferTime_sec;
                    bytesPerSec = bytesTransferred / transferTime_sec;
                    recordsPerSec = recordsTransferred / transferTime_sec;
                }
                else
                {
                    buffersPerSec = 0.;
                    bytesPerSec = 0.;
                    recordsPerSec = 0.;
                }

                //printf("Captured %u buffers (%.4g buffers per sec)\n", buffersCompleted, buffersPerSec);
                //printf("Captured %u records (%.4g records per sec)\n", recordsTransferred, recordsPerSec);
                //printf("Transferred %I64d bytes (%.4g bytes per sec)\n", bytesTransferred, bytesPerSec);
            }

            // Abort the acquisition
            retCode = AlazarAbortAsyncRead(boardHandle);
            if (retCode != ApiSuccess)
            {
                printf("Error: AlazarAbortAsyncRead failed -- %s\n", AlazarErrorToText(retCode));
                success = FALSE;
            }

            // Free all memory allocated
            for (bufferIndex = 0; bufferIndex < BUFFER_COUNT; bufferIndex++)
            {
                if (BufferArray[bufferIndex] != NULL)
                {
                    AlazarFreeBufferU16(boardHandle, BufferArray[bufferIndex]);

                }
            }

            // Close the data file

            if (fpData != NULL)
                fclose(fpData);

        }
        catch (int e)
        {
            cout << "An exception occurred. Exception Nr. " << e << '\n';
        }
    }
    //return sampleValues;
}


#ifndef WIN32
inline U32 GetTickCount(void)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0)
        return 0;
    return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

inline void Sleep(U32 dwTime_ms)
{
    usleep(dwTime_ms * 1000);
}

inline int _kbhit(void)
{
    struct timeval tv;
    fd_set rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO(&rdfs);
    FD_SET(STDIN_FILENO, &rdfs);

    select(STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &rdfs);
}

inline int GetLastError()
{
    return errno;
}
#endif

/////////////////////////////////////////////////////////////////////////////
// CSliderApp

BEGIN_MESSAGE_MAP(CSliderApp, CWinApp)
	ON_COMMAND(ID_APP_ABOUT, OnAppAbout)
	// Standard file based document commands
	//ON_COMMAND(ID_FILE_NEW, CWinApp::OnFileNew)
	//ON_COMMAND(ID_FILE_OPEN, CWinApp::OnFileOpen)
	// Standard print setup command
	//ON_COMMAND(ID_FILE_PRINT_SETUP, CWinApp::OnFilePrintSetup)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSliderApp construction

CSliderApp::CSliderApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

/////////////////////////////////////////////////////////////////////////////
// The one and only CSliderApp object

CSliderApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CSliderApp initialization

BOOL CSliderApp::InitInstance()
{
	// Initialize OLE libraries
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}

	AfxEnableControlContainer();

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	//  of your final executable, you should remove from the following
	//  the specific initialization routines you do not need.

	// Change the registry key under which our settings are stored.
	// TODO: You should modify this string to be something appropriate
	// such as the name of your company or organization.
	SetRegistryKey(_T("Microsoft\\MFC\\Samples"));

	LoadStdProfileSettings();  // Load standard INI file options (including MRU)

	SetRegistryBase (_T("Settings"));

	// Initialize all Managers for usage. They are automatically constructed
	// if not yet present
	// Register the application's document templates.  Document templates
	//  serve as the connection between documents, frame windows and views.

	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CSliderDoc),
		RUNTIME_CLASS(CMainFrame),       // main SDI frame window
		RUNTIME_CLASS(CSliderView));
	AddDocTemplate(pDocTemplate);

	// Parse command line for standard shell commands, DDE, file open
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);

    

	// Dispatch commands specified on the command line
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// The one and only window has been initialized, so show and update it.
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();

    m_pMainWnd->MoveWindow(0, 0, 200, 60);  // add this line for fixing the default size of mainWindow
    m_pMainWnd->ShowWindow(SW_SHOW);
    m_pMainWnd->UpdateWindow();

	return TRUE;
}

/////////////////////////////////////////////////////////////////////////////
// CSliderApp message handlers

int CSliderApp::ExitInstance() 
{
	return CWinAppEx::ExitInstance();
}

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
		// No message handlers
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnBnClickedOk();
    afx_msg void OnNMCustomdrawSlider1(NMHDR* pNMHDR, LRESULT* pResult);
    afx_msg void OnNMCustomdrawSlider2(NMHDR* pNMHDR, LRESULT* pResult);

    afx_msg void OnNMCustomdrawSlider3(NMHDR* pNMHDR, LRESULT* pResult);

    CSliderCtrl SLIDER_CONTRAST;
    //CScomboCtr combobox1;
    
    afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);

    CSliderCtrl DATA_MIN_SLIDER;
    CSliderCtrl DATA_MAX_SLIDER;
    CEdit N_AVE_BOX;
    afx_msg void OnBnClickedButton1();
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{

   
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
    //{{AFX_DATA_MAP(CAboutDlg)
    //}}AFX_DATA_MAP
    DDX_Control(pDX, IDC_SLIDER_CONTRAST, SLIDER_CONTRAST);
    DDX_Control(pDX, IDC_SLIDER_CONTRAST12, DATA_MIN_SLIDER);
    DDX_Control(pDX, IDC_SLIDER_CONTRAST13, DATA_MAX_SLIDER);
    DDX_Control(pDX, IDC_EDIT1, N_AVE_BOX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
    ON_BN_CLICKED(IDOK, &CAboutDlg::OnBnClickedOk)
    ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_CONTRAST, &CAboutDlg::OnNMCustomdrawSlider1)
    ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_CONTRAST12, &CAboutDlg::OnNMCustomdrawSlider2)
    ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_CONTRAST13, &CAboutDlg::OnNMCustomdrawSlider3)
    ON_WM_HSCROLL()
    ON_BN_CLICKED(IDC_BUTTON1, &CAboutDlg::OnBnClickedButton1)
END_MESSAGE_MAP()

// App command to run the dialog
void CSliderApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

/////////////////////////////////////////////////////////////////////////////
// CSliderApp message handlers


void CAboutDlg::OnBnClickedOk()
{   
    CRect rect;
    CAboutDlg::GetWindowRect(rect);
    OCT_COMNTROL_width = rect.Width();
    int dy = rect.Height();
    GetWindowRect(rect);
    rect.left = 0;// rect.left - 200;
    rect.right = rect.left+ OCT_COMNTROL_width;
    rect.top = 0;// rect.bottom - 200;
    rect.bottom = rect.top +dy;
    CAboutDlg::MoveWindow(rect);
    CAboutDlg::ShowWindow(SW_SHOW);

    //DATA_MAX_SLIDER.SetRange(0, 1, TRUE);
    SLIDER_CONTRAST.SetPos(50);
    DATA_MAX_SLIDER.SetPos(50);
    //DATA_MIN_SLIDER.SetRange(0, 1, TRUE);
    DATA_MIN_SLIDER.SetPos(30);

    N_AVE_BOX.SetWindowText("1");

    // TODO: Add your control notification handler code here
    if (1) {
        U32 systemId = 1;
        U32 boardId = 1;

        // Get a handle to the board

        HANDLE boardHandle = AlazarGetBoardBySystemID(systemId, boardId);
        if (boardHandle == NULL)
        {
            printf("Error: Unable to open board system Id %u board Id %u\n", systemId, boardId);
            //return 1;
        }

        // Configure the board's sample rate, input, and trigger settings

        if (!ConfigureBoard(boardHandle, VOLTAGE_LEVEL_idx1))
        {
            printf("Error: Configure board failed\n");
            //return 1;
        }

        //thread worker(AcquireData);
        
       // std::terminate(worker);
        
       // AlazarClose(boardHandle);


        thread worker(AcquireData);

        if (0) {
            if (!ConfigureBoard(boardHandle, 5))
            {
                printf("Error: Configure board failed\n");
                //return 1;
            }
            VOLTAGE_LEVEL_idx1 = VOLTAGE_LEVEL_idx;
        }
        
        // Make an acquisition, optionally saving sample data to a file
        
        //sampleValues = AcquireData(boardHandle);
        /* if (res)
        {
            printf("Error: Acquisition failed\n");
            return 1;
        }
        */

        // _bstr_t b(argv[]);
         //char* argv1 = b;
        //TCHAR* pCommandLine = ::GetCommandLine();
        //int nArgc = 0;
        //LPWSTR* pArgv = ::CommandLineToArgvW(pCommandLine, &nArgc);


        //printf("%d", __argc);
        kernel(__argc, __argv);

        

        //kernel(__argc, __argv);
        //int d = 1;
    }
    CDialog::OnOK();
}


void CAboutDlg::OnNMCustomdrawSlider1(NMHDR* pNMHDR, LRESULT* pResult)
{
    LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
    // TODO: Add your control notification handler code here
   // CONTRAST = CSliderCtrl::GetPos();
    CONTRAST = SLIDER_CONTRAST.GetPos();

    *pResult = 0;
}

void CAboutDlg::OnNMCustomdrawSlider2(NMHDR* pNMHDR, LRESULT* pResult)
{
    LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
    // TODO: Add your control notification handler code here
   // CONTRAST = CSliderCtrl::GetPos();
    LOWER_NORMALIZATION_THRESHOLD = (float) DATA_MIN_SLIDER.GetPos() /200.0;

    *pResult = 0;
}

void CAboutDlg::OnNMCustomdrawSlider3(NMHDR* pNMHDR, LRESULT* pResult)
{
    LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
    // TODO: Add your control notification handler code here
   // CONTRAST = CSliderCtrl::GetPos();
    UPPER_NORMALIZATION_THRESHOLD = (float) DATA_MAX_SLIDER.GetPos() / 200.0;

    *pResult = 0;
}

void CAboutDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // TODO: Add your message handler code here and/or call default

    CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
    CONTRAST = SLIDER_CONTRAST.GetPos();
}




void CAboutDlg::OnBnClickedButton1()
{
    CString N_AVE_str = _T("");
    N_AVE_BOX.GetWindowText(N_AVE_str);

    char* p;
    N_AVE = strtol(N_AVE_str, &p, 10);
    N = 1;
    if (*p != 0) {
        MessageBox(0, "Average number must be integer - N_AVE will be set to 1.",  MB_OK);
        N_AVE = 1;
        N = 1;
    }

   
    // TODO: Add your control notification handler code here
}
