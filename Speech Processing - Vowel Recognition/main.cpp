// SP_AS03.cpp : Defines the entry point for the console application.
/******************************************************************************************************************************************
	Speech Processing CS 566: Assignment 03
	Roll No: 214101058 MTech CSE'23 IITG
	Input: 
		*	Vowel Data Files '.txt' in format 214XXXXXX_%c_%d.txt (c={a,e,i,o,u} d={1,2,3 ..., 20} 100 files)
		*	Kept inside the path provided by array filePathVowels[] = "input_vowels_data/". (Line #89)
		*	Roll Num can be changed by variable rollNum[]="214XXXXXX" (Line #55)
	Output:  
		*	R1:Final Reference Files for each vowel by analysing the first 10 files of each vowel.
				Folder: fileOutputRef[] = "output_reference_file/";
				Format: {214XXXXXX} + _ + {a} + _avg_ci_reference.txt
		*	R2:Collective Analysed Files Data for each vowel by analysing the first 10 files in single file for each vowel.
				Folder:  fileOutputRef[] = "output_reference_file/";	
				Format: {214XXXXXX}+ _ + {a} + _final_analysis.txt
		*	T1:Individual Test File Result after comparing with each reference file of vowel.
				Folder: fileOutputTest[] = "output_test_result/";
				Format: {214XXXXXX} + _ + {a} + {1} + _test_result.txt
		*	T2:Collective Test File Result of all the test files (last 10 files of each vowel). 
				Folder: fileOutputTest[] = "output_test_result/";
				Format:  _ + {214XXXXXX} +_aeiou_TEST_RESULT_OUTPUT.txt
		*   (optional when var segregate_speech=TRUE)=>	Segregated Speech data of each file analysed for testing/debugging purpose. 
				Folder:  fileOutputRecording[] = "output_recordings/";
				Format: {214XXXXXX} + _ + {a} + _ + {1} + _normalized_segregated_recording.txt
*********************************************************************************************************************************************/

#include "stdafx.h"
#pragma warning (disable : 4996)		//to disable fopen() warning
#include <stdio.h>
#include <stdlib.h>		//atoi, atof
#include <string.h>		//strcat, strcpy
#include <conio.h>		//getch,
#define _USE_MATH_DEFINES	// for pi
#include <math.h>		// log, sin

/******************--------Common Settings--------******************/
#define sizeFrame 320							  // Number of Samples per Frame.
#define NumOfFiles 10					//change  // Number of Files for Testing/Training.
#define steadyFrames 5					//change  // Number of Steady Frames to Take.
#define p 12							//change  // p value.
//
#define scaleAmp 5000							  // Max Amplitutde Value to Scale.
#define initIgnoreHeaderLines 5			//change  // Number of Initial Header Lines in Txt Files to Ignore.
#define initIgnoreSamples 0				//change  // Number of Initial samples to ignore.
#define initNoiseFrames 5				//change  // Number of Initial Noise Frames to Consider.
#define thresholdNoiseToEnergyFactor 3	//change  // Noise to Energy Factor.
#define samplingRate 16000						  // Sampling Rate of Recording.

#define totVowels 5
#define MaxFrames 400					//change	// max number of frames to consider overall.
const unsigned short q=p;				//change	//#Coefficients (c_i's) that need to be found
/******************--------Variable Decalaration--------******************/
// VOWELS
const char vowels[]={'a','e','i','o','u','\0'};
const char rollNum[] = "214101058";		//change  //Roll Num Required For File Name // 214101058
// DISTANCE, WEIGHTS
double d_tokhura_min;					// minimum tokhura distance from all vowels calculation
int vowel_index_min;					// minimum tokhura distance vowel index
int correct_count=0;						// number of times correct vowel is detected.
double dt_tokhura_v[totVowels];			// tokhura distance calculated with respect to each vowel.
const double w_tkh[]={1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};		//tokhura weigths provided.

// COEFFICIENTS
double c_files_lift[NumOfFiles][steadyFrames][q+1];			// c_i's coefficients saved from analysing files, and after applied liftering window. 10*5*12
double c_finalAvg[totVowels][steadyFrames][q+1];			// final c_i's after avegare of all #num_files i.e. final 5*12 per vowel
double c_wtest[steadyFrames][q+1];							// this contains c_i's of the test file. 5*12
double w_rsw[q+1];						// weights for raised sine window.

// Speech
int samples[500000], maxAmp = 0;							//max number of samples in recording to consider, maximum amplitute for rescaling.
double normSamples[500000];									//normalised samples.
long cntTotSamples = 0, cntTotFrames = 0;					// Total Samples in recording, Total Frames in Recording.
float DCshift =0;
long start=0, end=0 ;										//start and end marker, Frames

// ZCR and ENERGY
float cntZCR[MaxFrames], avgZCR[MaxFrames], avgEnergy[MaxFrames];  
float totEnergy=0, noiseEnergy=0, initNoiseZCR=0, initNoiseAvgZCR=0;;
float thresholdZCR=0;
double thresholdEnergy=0;
double maxEnergy = 0;										// Max Energy of the Frame.
int STE_Marker;												// Max STE Marker For the Frame

// File Stream
FILE *fp_ip, *fp_norm, *fp_norm_seg, *fp_final_op, *fp_ref, *fp_console;							// file pointer for input stream and output stream.
char fileNameIp[300], completePathIp[300], completePathNorm[300],completePathNormSeg[300], completePathFinOp[300], completePathRef[300], completePathConsole[300];
char charsLineToRead[50];															// number of characters to read per line in file.
const char filePathInputValidation[] = "input_validation_data/test.txt";			// Test Dump Data Path. input_validation_data/test.txt/
const char filePathVowels[] = "input_vowels_data/";		 							// Folder name where all Vowels recordings are placed 1 to 100.	input_vowels_data/
const char fileOutputRecording[] = "output_recordings/";							// Folder name where output of segregated vowels recordings are placed and its normalised files. output_recordings/
const char fileOutputRef[] = "output_reference_file/";								// Folder name where reference file are generated. output_reference_file/
const char fileOutputTest[] = "output_test_result/";								// Folder name where test file analysis are saved. output_test_result/
bool segregate_speech = false;														// True: to segreagate speech data with respect to start and end marker in folderfileOutputRecording[].

/**************************************************************************************
	Calculating w[m] weights for raised sine window and storing in array for future use.
**************************************************************************************/
void CalculateWeightsForRaisedSineWindow(){
	for(int i=1;i<=q;i++){
		w_rsw[i] = 1 + (q/2)*sin(M_PI*i/q);
	}
}

/**************************************************************************************
	Calculating R[i] Values using Auto Correlation Method
	Input:	   *s is the pointer to our samples array.
	Output:	   *Rn is the pointer to our R array for storing final Ri values (0 to p).
**************************************************************************************/
void calculateRi_AutoCorrelation(double *s, double *Rn){
	for(int k=0;k<=p;k++) 
	{ 
		Rn[k]=0;
		for(int m=0; m < sizeFrame-k; m++)			
		{
			Rn[k] += s[m]*s[m+k];				// R[0] R[1] .... R[p]
		}	
	}
}

/**************************************************************************************
	Calculating A[i] Values using Levinson Durbin Process
	Input:	   *R is the pointer to our Auto Correlated Energy part.
	Output:	   *A is the pointer to our A array for storing final Ai values (1 to p).
**************************************************************************************/
void calculateAi_Durbin(double *R, double *A){
	 double E[p+1] = {0};
	 double k[p+1] = {0};
	 double alpha[p+1][p+1] = {0};
	 double sum=0;
	 int i, j;

	E[0]=R[0]; //Initialize E[0] with energy

	for( i=1;i<=p;i++)
	{
		if(i==1)
			k[1]=R[1]/R[0]; //special case i=1
		else //find k(i) for all other remaining values of i
		{
			 sum=0;
			for( j=1;j<=i-1;j++)
			{
				sum+=alpha[i-1][j]*R[i-j];
			}
			k[i]=(R[i]-sum)/E[i-1];
		}
		alpha[i][i]=k[i];

		for( j=1;j<=i-1;j++)
			alpha[i][j]=alpha[i-1][j]-k[i]*alpha[i-1][i-j]; //update coefficients from previous values
		
		E[i]=(1-k[i]*k[i])*E[i-1]; //update E[i]
	}
	for( i=1;i<=p;i++){	
		//A[i] = -1*alpha[p][i];
		A[i] = alpha[p][i];				// A[0] A[1] .... A[p]
	}
}

/**************************************************************************************
	Calculating C[i] Cepstral Coefficient Values.
	Input:	   *A is the pointer to our Ai Array.
				sigma is R[0] value.
	Output:	   *A is the pointer to our A array for storing final Ai values (1 to p).
**************************************************************************************/
void calculateCi_Cepstral(double sigma, double *A, double *c){
	int k,m;
	double sum=0;

	c[0]= logl(sigma*sigma); 
	for(m=1; m<=p; m++)
	{
		sum=0;
		for(k=1;k<=m-1;k++) //sum from older cepstral coefficents to compute new cepstral coefficients
			sum+=k*c[k]*A[m-k]/m;
		c[m]=A[m]+sum;		//new cepstral coefficients
	}
	/*
		if(m>p) // For This Assignment this never get executed as we assume q=p
		{
			for(;m<=q;m++)
			{
				sum=0;
				for(k=1;k<=m-1;k++) //sum from older cepstral coefficents to compute new cepstral coefficients
					sum+=k*c[k]*A[m-k]/m;
				c[m]=sum;		//new cepstral coefficients
			}
		}
	*/
}

/**************************************************************************************
	PRE REQUISITE: Validating Coefficients From Test Dump Data Provided
**************************************************************************************/
int validate_coefficients(){
	FILE *fp_ip_test;

	double testSamples[321];
	int totTestSamples=0;

	// Opening Respective Test Input File.
	fp_ip_test = fopen(filePathInputValidation, "r");	
		if(fp_ip_test == NULL ){
				perror("\nError: ");
				printf("\n File Names are Input : \n%s ", fp_ip_test );
				getch();
				return EXIT_FAILURE;
		}

	// Reading From File and Storing in Array.
	while(!feof(fp_ip_test)){
		fgets(charsLineToRead, 50, fp_ip_test);
		totTestSamples += 1 ;  
			testSamples[totTestSamples - 1] = atof(charsLineToRead);
	}

	// Calculating R_i values using AutoCorrelation Method.
	double R[p+1] = {0};
	 calculateRi_AutoCorrelation(testSamples, R);

	// calculating A_i using Durbin algo.
	double A[p+1] = {0};
	 calculateAi_Durbin(R, A);

	// calculating Cepstral coefficient.
	double C[q+1] = {0};
	 calculateCi_Cepstral(R[0], A, C);

	printf("\n ---- PRE REQUISITE: Validating Coefficients From Test Dump Data Provided ---- \n");	
	printf("\n File: %s",filePathInputValidation);

	printf("\n\nR Coefficient Values \t LPC Coefficient values \t Cepstral Coefficient values\n");
	
	printf("R[%d] = %lf \n",0, R[0]);

	for(int i=1;i<=p;i++){
		printf("R[%d] = %lf \t ",i, R[i]);
		printf("A[%d] = %lf \t ", i, A[i]);
		printf("C[%d] = %lf \n", i, C[i]);
	}
	/* For This Assignment this never get executed as we assume q=p
		for(int i=p+1;i<=q;i++){
			printf("\t\tC[%d] = %lf ", i, C[i])
		}
	*/

	printf("\n");
	
	return 0;
}

/**************************************************************************************
	To Display Common Settings used in our System
	Input: File Pointer in case things needed to be written on file.
**************************************************************************************/
void DisplayCommonSettings(FILE *fp_op=NULL){
	// General Information to Display
	if(fp_op==NULL){
		printf("****-------- WELCOME TO VOWEL RECOGNITION SYSTEM --------****\n");		
		printf("-Common Settings are : -\n");	
		printf(" File Roll Num : %s\n", rollNum);
		printf(" P (=Q) = : %d\n", p);
		printf(" Frame Size : %d\n", sizeFrame);	
		printf(" Num of Files For Training or Testing : %d\n", NumOfFiles);	
		printf(" Num of Steady Frames per file : %d\n", steadyFrames);

		printf(" Tokhura Weights : ");
		for(int i=0; i<q; i++){
			printf("%0.1f(%d) ", w_tkh[i],i+1);
		}
		printf("\n");
		printf(" Amplitutde Value to Scale : %d\n", scaleAmp);			
		printf(" Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		printf(" Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		printf(" Intital Noise Frames Count : %d\n",initNoiseFrames);	
		printf(" Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		printf(" Sampling Rate of Recording: %d\n",samplingRate); 
		printf("----------------------------------------------------------------\n\n");		
	}
	else if(fp_op!=NULL){
		fprintf(fp_op, "****-------- WELCOME TO VOWEL RECOGNITION SYSTEM --------****\n");		
		fprintf(fp_op, "-Common Settings are : -\n");	
		fprintf(fp_op, " File Roll Num : %s\n", rollNum);
		fprintf(fp_op, " P (=Q) = : %d\n", p);
		fprintf(fp_op, " Frame Size : %d\n", sizeFrame);	
		fprintf(fp_op, " Num of Files For Training or Testing : %d\n", NumOfFiles);	
		fprintf(fp_op, " Num of Steady Frames per file : %d\n", steadyFrames);

		fprintf(fp_op, " Tokhura Weights : ");
		for(int i=0; i<q; i++){
			fprintf(fp_op, "%0.1f(%d) ", w_tkh[i],i+1);
		}
		fprintf(fp_op, "\n");
		fprintf(fp_op, " Amplitutde Value to Scale : %d\n", scaleAmp);			
		fprintf(fp_op, " Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		fprintf(fp_op, " Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		fprintf(fp_op, " Intital Noise Frames Count : %d\n",initNoiseFrames);	
		fprintf(fp_op, " Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		fprintf(fp_op, " Sampling Rate of Recording: %d\n",samplingRate); 
		fprintf(fp_op, "----------------------------------------------------------------\n\n");	
	}
}

/**************************************************************************************
	Normalising and DC Shift of Samples.
	Input (global):		fp_ip is to read from recording text file.
						fp_norm is to save normalised samples into another file.
						fp_final_op is to save analysed values in common output file.
**************************************************************************************/
void normalize_dcshift_samples(){
			cntTotSamples=0; maxAmp = 0;

			int totIgnore;
				 totIgnore=initIgnoreHeaderLines + initIgnoreSamples; //5 + 2 = 7
			// totIgnore=0;
			long sample_index = totIgnore+1; // till 7 samples ignored, sample count is 8, so to make array index 0 there is +1 
			long sample_index_norm = 0;
			double normFactor = 0;
			double normOutput = 0;

				while(!feof(fp_ip)){
					fgets(charsLineToRead, 50, fp_ip);
					cntTotSamples += 1 ;  

					if(cntTotSamples > totIgnore){
						sample_index_norm = cntTotSamples - sample_index;
						samples[sample_index_norm] = (int)atoi(charsLineToRead);
						DCshift += samples[sample_index_norm];
								if(abs(samples[sample_index_norm]) > maxAmp)
									maxAmp = abs(samples[sample_index_norm]);
					}
				}

			cntTotSamples = cntTotSamples - totIgnore;		// total number of samples stored in array
			DCshift = DCshift/cntTotSamples;				//average DC Shift
			cntTotFrames = cntTotSamples/sizeFrame;			//total number of frames.

			normFactor = (double)scaleAmp/maxAmp;			//normalising factor

			// saving in normalised file
			for(long i=0; i<cntTotSamples; i++){
				normOutput = (double)(samples[i] - DCshift)*normFactor;
				normSamples[i]=normOutput;
				fprintf(fp_norm, "%lf\n", normSamples[i]);
			}
			
			
	//printf(" TOTAL SAMPLES : %d\n TOTAL FRAMES : %d\n DC SHIFT needed : %lf\n Maximum Amplitude : %d\n Normalization Factor : %lf\n ", cntTotSamples, cntTotFrames, DCshift, maxAmp, normFactor);
	if(fp_final_op!=NULL){
		fprintf(fp_final_op, " TOTAL SAMPLES : %d\n TOTAL FRAMES : %d\n DC SHIFT needed : %lf\n Maximum Amplitude : %d\n Normalization Factor : %lf\n ", cntTotSamples, cntTotFrames, DCshift, maxAmp, normFactor);
	}
}

/**************************************************************************************
	Calculating ZCR and Energy of Frames
	Input (global): fp_norm is to read from normalised file of the samples.
**************************************************************************************/
void zcr_energy_frames(){
	rewind(fp_norm);

	long i,j;
	float s_i, s_i_1=1;

	//totEnergy=0;
	 maxEnergy = 0;										// Max Energy of the Frame.
	 STE_Marker = 0;	
	
	for(i=0;i < cntTotFrames;i++)
		{
			cntZCR[i]=0;
			avgEnergy[i]=0;
			for(j=0;j < sizeFrame ;j++)
			{
				fgets(charsLineToRead, 50, fp_norm); // reading from normalised input
				s_i = (float)atof(charsLineToRead);
				avgEnergy[i] += (s_i*s_i);
				cntZCR[i] +=  (s_i_1*s_i < 0);
				s_i_1 = s_i;
			}	
			avgEnergy[i]/=sizeFrame;
			//avgZCR[i] = cntZCR[i]/sizeFrame;
			//totEnergy+=avgEnergy[i];
			//fprintf(fp_norm, "%f %0.1f \n", avgEnergy[i], cntZCR[i]);	//dumping the features of frames.
			// calculation for detecting STE Frame
			if(avgEnergy[i] > maxEnergy){
				maxEnergy = avgEnergy[i];
				STE_Marker=i;
			}
		}

	
			for(i=start;i <= end;i++)
				{
					if(avgEnergy[i] > maxEnergy){
						maxEnergy = avgEnergy[i];
						STE_Marker=i;
					}
				}
}

/**************************************************************************************
	Calculating ZCR and Energy of Noise Frames, and Finally Thresholds for ZCR and Energy
	Input (global): fp_final_op is to save analysed values in common output file.
**************************************************************************************/
void noiseEnergy_thresholds_frames(){
	noiseEnergy=0; initNoiseZCR=0; initNoiseAvgZCR=0;
	int i;
	for(i=0;i < initNoiseFrames;i++){
			initNoiseZCR+=cntZCR[i];
			//initNoiseAvgZCR+=avgZCR[i];
			noiseEnergy+=avgEnergy[i];
	}
		thresholdZCR=initNoiseZCR/initNoiseFrames;
		noiseEnergy/=initNoiseFrames;
		thresholdEnergy=noiseEnergy*thresholdNoiseToEnergyFactor;

	//printf( "\n---- Initial Noise Frames : %d ----\n\n", initNoiseFrames);
	//printf(" Avg Noise Energy : %lf\n Total Noise ZCR : %0.1f\n Threshold ZCR : %0.1f\n Threshold Energy(Avg Noise*%d) : %0.5lf\n ", noiseEnergy, initNoiseZCR, thresholdZCR, thresholdNoiseToEnergyFactor, thresholdEnergy);
	if(fp_final_op!=NULL){
		fprintf(fp_final_op, "\n---- Initial Noise Frames : %d ----\n\n", initNoiseFrames);
		fprintf(fp_final_op, " Avg Noise Energy : %lf\n Total Noise ZCR : %0.1f\n Threshold ZCR : %0.1f\n Threshold Energy(Avg Noise*%d) : %0.5lf\n ", noiseEnergy, initNoiseZCR, thresholdZCR, thresholdNoiseToEnergyFactor, thresholdEnergy);
	}
	
}

/**************************************************************************************
	Detecting Start and End Marker of the Frame.
	Input (global): fp_final_op is to save analysed values in common output file.
					fp_norm_Seg is to save normalised segregated samples into another file.
**************************************************************************************/
void marker_start_end_segregated(){
	bool flag=false;		//to detect start mark
	// -3 to ignore last 3 frames.
	for(int i=0; i<cntTotFrames-3; ++i){
			if(!flag && avgEnergy[i+1] > thresholdEnergy && avgEnergy[i+2] > thresholdEnergy && avgEnergy[i+3] > thresholdEnergy && avgEnergy[i+4] > thresholdEnergy && avgEnergy[i+5] > thresholdEnergy){
					start = i;
					flag = 1;
			}
			else if(flag && avgEnergy[i+1] <= thresholdEnergy && avgEnergy[i+2] <= thresholdEnergy && avgEnergy[i+3] <= thresholdEnergy && avgEnergy[i+4] <= thresholdEnergy && avgEnergy[i+5] <= thresholdEnergy){
				end = i;
				flag = 0;
				break;
			}
		}
	if(flag == 1) end = cntTotFrames - 5; //if end is not found then making the last frame - 3 as the end marker for the word

	long startSample= (start+1)*sizeFrame;
	long endSample= (end+1)*sizeFrame;
	long totFramesVoice = end-start+1;
	
/****************  saving segregated voice data in different file ****************/
	if(fp_norm_seg!=NULL){
		for(long i=startSample-1; i<endSample; i++){
			fprintf(fp_norm_seg, "%lf\n", normSamples[i]);
		}
	}

		//printf("\n---- Segregated Data Saved in File: %s ----\n\n", completePathNormSeg);
		//printf(" START Frame : %ld\t END Frame : %ld\t Total Frames : %ld\n", start+1, end+1, totFramesVoice);
		//printf(" Starting Sample : %ld\t Ending Sample : %ld\n", startSample, endSample);
		//printf("\n--------\n");
		//printf("\n------------------------------------------------------------------------\n");

	if(fp_final_op!=NULL){
		fprintf(fp_final_op, "\n---- Segregated Data Saved in File: %s ----\n\n", completePathNormSeg);
		fprintf(fp_final_op, " START Frame : %ld\t END Frame : %ld\t Total Frames : %ld\n", start+1, end+1, totFramesVoice);
		fprintf(fp_final_op, " Starting Sample : %ld\t Ending Sample : %ld\n", startSample, endSample);
		fprintf(fp_final_op, "\n--------\n");
		fprintf(fp_final_op, "\n------------------------------------------------------------------------\n");
	}
}

/**************************************************************************************
	Calculating Coefficients for Steady Frames of Reference File
	Input (global): fp_final_op is to save analysed values in common output file.
					normSamples[] array contains normalised values of samples.
					fileCounter is used to C_i values of respective file in proper index.
**************************************************************************************/
void calculateCoefficientsSteadyFramesReference(int fileCounter){
	if(fp_final_op!=NULL)
	{
		fprintf(fp_final_op, "\n\t Max STE Marker Frame : %d\n", STE_Marker);			// Max STE marker calculaed on zcr_energy_frames();
	}
	double steadySamples[sizeFrame]={0};	

	int startOfSteadyFrame = STE_Marker-2;			// -2 frame
	//int skipCounter=0;
	for(int sf=0; sf<steadyFrames; sf++){
		
		long SampleMarkerToCopy= (startOfSteadyFrame + sf)*sizeFrame;	

		for(int i=0; i<sizeFrame; i++){
			steadySamples[i] = normSamples[i + SampleMarkerToCopy];
		}

		// Calculating R_i values using AutoCorrelation Method.
		double R[p+1] = {0};
			calculateRi_AutoCorrelation(steadySamples, R);
			/* Not Applicable for this assignment as we are taking Max STE Frame.
				if(R[0]==0){
					printf("\n R[0] Energy should not be ZERO, Skipping frame %d, \n", (startOfSteadyFrame + sf));
						fprintf(fp_final_op, "\n\tFile Name: %s, Frame: %d, R[0] Energy should not be ZERO, Skipping frame %d,", fileNameIp, sf+1, (startOfSteadyFrame + sf));
					skipCounter++;
					continue;
				}
			*/
		// calculating A_i using Durbin algo.
		double A[p+1] = {0};
			calculateAi_Durbin(R, A);

		// calculating Cepstral coefficient.
		double C[q+1] = {0};
			calculateCi_Cepstral(R[0], A, C);

			if(fp_final_op!=NULL)
			{
				fprintf(fp_final_op, "\n\t\t\tFile Name: %s, Frame: %d", fileNameIp, sf+1);
				fprintf(fp_final_op, "\n\nR Coefficient Values \t LPC Coefficient values \t Cepstral Coefficient values \t Raised Sine Window\n");
			}
		/****************  Final Values of coefficients ****************/ 		
				fprintf(fp_final_op, "R[%d] = %lf \n",0, R[0]);
		for(int i=1;i<=p;i++){	
			c_files_lift[fileCounter-1][sf][i] = C[i]*w_rsw[i];  //fileCounter starting from 1 so -1,  
			if(fp_final_op!=NULL)
			{	
				fprintf(fp_final_op, "R[%d] = %lf \t ",i, R[i]);
				fprintf(fp_final_op, "A[%d] = %lf \t ", i, A[i]);
				fprintf(fp_final_op, "C[%d] = %lf \t", i, C[i]);	
				fprintf(fp_final_op, "*(%lf)=> C[%d] = %lf \n", w_rsw[i], i, c_files_lift[fileCounter-1][sf][i]);
			}
		}
		/* For This Assignment this never get executed as we assume q=p
		for( i=p+1;i<=q;i++){
			c_files_lift[fileCounter-1][sf][i] = C[i]*w_rsw[i];  //fileCounter starting from 1,  
		}
		*/

					
	}//end of sf loop
					
}

/**************************************************************************************
	Calculating Coefficients for Steady Frames of Test File
	Input (global): fp_final_op is to save analysed values in common output file.
					normSamples[] array contains normalised values of samples.
					fileCounter is used to C_i values of respective file in proper index.
**************************************************************************************/
void calculateCoefficientsSteadyFramesTest(){
	if(fp_final_op!=NULL)
	{
		fprintf(fp_final_op, "\n\t Max STE Marker Frame : %d\n", STE_Marker);			// Max STE marker calculaed on zcr_energy_frames();
	}
	double steadySamples[sizeFrame]={0};	

	int startOfSteadyFrame = STE_Marker-2;			// -2 frame
	//int skipCounter=0;
	for(int sf=0; sf<steadyFrames; sf++){
		
		long SampleMarkerToCopy= (startOfSteadyFrame + sf)*sizeFrame;	

		for(int i=0; i<sizeFrame; i++){
			steadySamples[i] = normSamples[i + SampleMarkerToCopy];
		}

		// Calculating R_i values using AutoCorrelation Method.
		double R[p+1] = {0};
			calculateRi_AutoCorrelation(steadySamples, R);
			/* Not Applicable for this assignment as we are taking Max STE Frame.
				if(R[0]==0){
					printf("\n R[0] Energy should not be ZERO, Skipping frame %d, \n", (startOfSteadyFrame + sf));
						fprintf(fp_final_op, "\n\tFile Name: %s, Frame: %d, R[0] Energy should not be ZERO, Skipping frame %d,", fileNameIp, sf+1, (startOfSteadyFrame + sf));
					skipCounter++;
					continue;
				}
			*/
		// calculating A_i using Durbin algo.
		double A[p+1] = {0};
			calculateAi_Durbin(R, A);

		// calculating Cepstral coefficient.
		double C[q+1] = {0};
			calculateCi_Cepstral(R[0], A, C);

			if(fp_final_op!=NULL)
			{
				fprintf(fp_final_op, "\n\t\t\tFile Name: %s, Frame: %d", fileNameIp, sf+1);
				fprintf(fp_final_op, "\n\nR Coefficient Values \t LPC Coefficient values \t Cepstral Coefficient values \t Raised Sine Window\n");
			}
		/****************  Final Values of coefficients ****************/ 		
				fprintf(fp_final_op, "R[%d] = %lf \n",0, R[0]);
		for(int i=1;i<=p;i++){	
			c_wtest[sf][i] = C[i]*w_rsw[i];    //fileCounter starting from 1 so -1,  
			if(fp_final_op!=NULL)
			{	
				fprintf(fp_final_op, "R[%d] = %lf \t ",i, R[i]);
				fprintf(fp_final_op, "A[%d] = %lf \t ", i, A[i]);
				fprintf(fp_final_op, "C[%d] = %lf \t", i, C[i]);	
					fprintf(fp_final_op, "*(%lf)=> C[%d] = %lf \n", w_rsw[i], i, c_wtest[sf][i]);
			}
		}
		/* For This Assignment this never get executed as we assume q=p
		for( i=p+1;i<=q;i++){
			c_files_lift[fileCounter-1][sf][i] = C[i]*w_rsw[i];  //fileCounter starting from 1,  
		}
		*/

					
	}//end of sf loop
					
}

/**************************************************************************************
	Calculating Tokhura Distance with respect to each reference file.
	Find minimum distance overall and vowel index.
	Input (global): Reference file c_i's of each vowel is stored in c_finalAvg[v][f][k]
					Test File c_i's are stored in c_wtest[f][k].
					dt_tokhura_v[v] stored tokhura distance calculated with respect to each vowel.
**************************************************************************************/
void calculateTokhuraDistance()
{
	double temp;
	double dt_frame_avg=0;
	double dt_frame[steadyFrames];				//distance of tokhura per frame
	for(int vi=0 ; vi<totVowels; vi++){	
			dt_frame_avg=0;								fprintf(fp_final_op, "\n\t\t--------------- Vowel Reference: %c: ---------------\n", vowels[vi]);
		for(int f=0;f<steadyFrames;f++){
				dt_frame[f]=0;							fprintf(fp_final_op, "F%d: ", f+1);
			for(int k=1; k<=q; k++){
				temp = c_wtest[f][k] - c_finalAvg[vi][f][k];
				dt_frame[f] += w_tkh[k-1]*temp*temp;
			}
			dt_frame_avg +=dt_frame[f];					fprintf(fp_final_op, " %lf \t", dt_frame[f]);
		}
		dt_tokhura_v[vi] = dt_frame_avg/steadyFrames;		fprintf(fp_final_op, "\n Avg Tokhura Distance: %lf \n", dt_tokhura_v[vi]);
	}

	d_tokhura_min = 999999999999999; //reinitialize min
	vowel_index_min=5;
	for (int t = 0; t < totVowels; ++t) //for each alphabet
	{
		if (dt_tokhura_v[t] < d_tokhura_min) //update min and minIndex
		{
			d_tokhura_min = dt_tokhura_v[t];
			vowel_index_min = t;
		}
	}
}


/**************************************************************************************
		Main Function
***************************************************************************************/
int main()
{	
	 /*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
																				STEP 1: Validating Calculation of Coefficients
	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	validate_coefficients();				// validating calculation of coefficients from test dump data.
	CalculateWeightsForRaisedSineWindow();	// calculating weights for Raised sine window before hand using in program.
	
		system("pause");
		system("cls");
	
	DisplayCommonSettings();		// General Information to Display

	 /*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
																				STEP 2: GENERATING REFERENCE FILE
	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	for(int v = 0 ; v<totVowels ; v++) //iterating through all 5 vowels.
	{			
		printf("\n----------------- GENERATING REFERENCE FILE ------------------\n");
		printf("==> Generating Reference File For Vowel : %c \n",vowels[v]);

		for(int fileCounter=1; fileCounter <= NumOfFiles ; ++fileCounter)//iterating through all files of given vowels (1 to 10).
		{
		/**************** Creating necessary file Path for data. ****************/
				//fileNameIp is the path for input vowel data.
				sprintf(fileNameIp, "%s_%c_%d", rollNum, vowels[v], fileCounter); //fileNameIp = {214101058} + _ + {a} + _ + {1}
			sprintf(completePathIp, "%s%s.txt", filePathVowels, fileNameIp); // completePathIp/ {fileNameIp} + .txt
			sprintf(completePathFinOp, "%s%s_%c_f1_%d_final_analysis.txt", fileOutputRef, rollNum, vowels[v], NumOfFiles); // fileOutputRef/{214101058}+ _ + {a} + _final_analysis.txt
			sprintf(completePathNorm, "%s%s_normalized_samples.txt", fileOutputRecording, fileNameIp); // fileOutputRecording/ {fileNameIp} + _normalized_samples.txt
			sprintf(completePathNormSeg, "%s%s_normalized_segregated_recording.txt", fileOutputRecording, fileNameIp); // fileOutputRecording/ {fileNameIp} + _normalized_segregated_recording.txt
				//sprintf(completePathNormFeature, "%s%s_normalized_features_extract.txt", fileOutputRecording, fileNameIp);
		/**************** Opening respective files. ****************/
			fp_ip = fopen(completePathIp, "r");				//to read input file
			fp_norm = fopen(completePathNorm, "w+");		//to save normalised samples
			fp_norm_seg = fopen(completePathNormSeg, "w");  //to save segregated recording from start to end
			if(fileCounter==1){
				remove(completePathFinOp);					
				fp_final_op = fopen(completePathFinOp, "a"); //to save compelete file data from 1 to num_of_files in one file only.
			}
			if(fp_ip == NULL || fp_norm == NULL ||  fp_final_op==NULL || fp_norm_seg == NULL   ){ 
					perror("\nError: ");				fprintf(fp_final_op, "\n File Reading Error: ");
					printf("\n File Names are Input, NormIp, NormSeg, FinalOp : \n  %s, \n  %s, \n  %s, \n  %s \n", completePathIp, completePathNorm, completePathNormSeg,  fp_final_op );
						fprintf(fp_final_op, "\n File Names are Input, NormIp, NormSeg, FinalOp : \n  %s, \n  %s, \n  %s, \n  %s \n", completePathIp, completePathNorm, completePathNormSeg,  fp_final_op );
					getch();
					return EXIT_FAILURE;
			}
			
			if(fileCounter==1){
				DisplayCommonSettings(fp_final_op); // writing into output file once at beginning.
			}
			
		//printf("\n ----------------------- START ANALYZING OF FILE: %s ----------------------- \n", fileNameIp);  
		fprintf(fp_final_op, "\n ----------------------- START - ANALYZING OF FILE: %s ----------------------- \n", fileNameIp);

		/**************** DC Shift and Normalizing ****************/
			normalize_dcshift_samples();

		/**************** Frames ZCR and Energy. STE Marker ****************/
			zcr_energy_frames();

		   if(segregate_speech){						//only if you want to segregate speech into separate file.
			/****************  calculating noise energy and threshold. ****************/
				noiseEnergy_thresholds_frames();						// if you want to calculate thresholds for zcr and energy
					
			/**************** start and end marker of speech ****************/
				marker_start_end_segregated();							//this and above func, if you want to detect start, end marker of speech, and to save it in separate file.
				fclose(fp_norm_seg);	// closing file stream
			}
		   else
		   {
			   fclose(fp_norm_seg);		// closing file stream
			   remove(completePathNormSeg);		//removing unnecessory file created.
		   }

		/****************  Calculating Coefficients for Steady Frames of File ****************/
				calculateCoefficientsSteadyFramesReference(fileCounter);
				
				//printf("\n ----------------------- END Analyzing OF File: %s ----------------------- \n", fileNameIp);  
				fprintf(fp_final_op, "\n ----------------------- END - ANALYZING OF FILE: %s ----------------------- \n", fileNameIp);
				fprintf(fp_final_op, "----------------------- \n");
				
				fclose(fp_norm);			// closing file stream, as no longer needed.
				remove(completePathNorm);	//comment it if you want to keep normalised data file.
				fclose(fp_ip); // closing file stream
		
		}//end of filecounter loop -------------------------------------------------------------------------------------------------------------------
	
	/****************  Calculating Average C_i From all the values from files. ****************/ 

		fprintf(fp_final_op, "\n-----------------------\n\n"); 
		fprintf(fp_final_op, " ----------- AVG C_i From %d Files for Vowel: %c -----------  \n", NumOfFiles, vowels[v]);

	sprintf(fileNameIp, "%s_%c", rollNum, vowels[v]); //fileNameIp = {214101058} + _ + {a} 
	sprintf(completePathRef, "%s%s_avg_ci_reference.txt", fileOutputRef, fileNameIp); // fileOutputRef/ {fileNameIp} + _avg_ci_reference.txt

	fp_ref = fopen(completePathRef, "w");	
		if(fp_ref == NULL ){
				perror("\nError: ");
				printf("\n Reference File Name: \n\t %s, \n", completePathRef );
					fprintf(fp_final_op, "\nError: ");
					fprintf(fp_final_op, "\n Reference File Name: \n\t %s, \n", completePathRef );
				getch();
				return EXIT_FAILURE;
		}

		/****************  Calculating Average C_i From all the values from files. ****************/ 
		/*	
			c[0][1] = cf[0][0][1] + cf[1][0][1] + ... + cf[9][0][1] //0th frame, c_1
			c[0][2] = cf[0][0][2] + cf[1][0][2] + ... + cf[9][0][2] //0th frame, c_2
			.
			.
			c[0][12] =cf[0][0][12] + cf[1][0][12] + ... + cf[9][0][12] //0th frame, c_12
			
			c[1][1] = cf[0][1][1] + cf[1][1][1] + ... + cf[9][1][1] //1st frame, c_1
			c[1][2] = cf[0][1][2] + cf[1][1][2] + ... + cf[9][1][2] //1st frame, c_2
			.
			.
			c[1][12] =cf[0][1][12] + cf[1][1][12] + ... + cf[9][1][12] //1st frame, c_12
		*/
		for(int i=0;i<steadyFrames;i++){						// for each frame i
				fprintf(fp_final_op, "\nFrame %d: ", i+1);
			for(int j=0;j<NumOfFiles;j++){						// of each file j
				for(int k=1;k<=p;k++){							// taking sum of c_k 1 of each file 
					c_finalAvg[v][i][k] += c_files_lift[j][i][k] ;
				}
			}		
			for(int k=1;k<=p;k++){								// taking avg of c_k 1 to p 
					c_finalAvg[v][i][k] /= NumOfFiles;
					//printf("%lf(%d) \t", c_finalAvg[v][i][k],k);	
						fprintf(fp_final_op, "%lf(%d) \t", c_finalAvg[v][i][k],k);
						fprintf(fp_ref,"%lf \n", c_finalAvg[v][i][k],k);
			}
		}
			
			fprintf(fp_final_op, "\n-----------------------\n\n"); 

		printf("\n -------- Vowel %c ----------- \n\t FILE ANALYSIS DATA 1 to %d: %s \n\t Reference File Generated: %s\n\n", vowels[v], NumOfFiles, completePathFinOp, completePathRef); 
		printf("\n------------------------------------------------------------------------\n");
			
		fflush(fp_ref);
		fflush(fp_final_op);
		fclose(fp_ref);
		fclose(fp_final_op);
		system("pause");
	}//end of vowel loop ------------------------------------------------------------------------------------------------------------------------------
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
																				STEP 3: TESTING
	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	
//	DisplayCommonSettings();
	sprintf(completePathConsole, "%s_%s_aeiou_TEST_RESULT_OUTPUT.txt", fileOutputTest, rollNum); // fileOutputTest/ _ + {214101058} +_aeiou_TEST_RESULT_OUTPUT.txt
	fp_console = fopen(completePathConsole, "w"); //to save compelte test result of each vowel.
		DisplayCommonSettings(fp_console);

	int overall_count=0;

	for(int v = 0 ; v<totVowels ; v++){		//looping through vowels.
		system("cls");
		printf("\n---------------- TESTING --------------------\n");
		printf("==> For Vowel : %c \n",vowels[v]);
			
		
		for(int fileCounter=11; fileCounter <= 20 ; ++fileCounter)	// looping through files of vowels.
		{

			/**************** Creating necessary file Path for data. ****************/
			sprintf(fileNameIp, "%s_%c_%d", rollNum, vowels[v], fileCounter); //fileNameIp = {214101058} + _ + {a} + _ + {1} 
			sprintf(completePathIp, "%s%s.txt", filePathVowels, fileNameIp); // completePathIp/ {fileNameIp} + .txt
			sprintf(completePathNorm, "%s%s_normalized_samples.txt", fileOutputRecording, fileNameIp); // fileOutputRecording/ {fileNameIp} + _normalized_samples.txt
			sprintf(completePathFinOp, "%s%s_test_result.txt", fileOutputTest, fileNameIp); // fileOutputTest/ {fileNameIp} + _test_result.txt
				sprintf(completePathNormSeg, "%s%s_normalized_segregated_recording.txt", fileOutputRecording, fileNameIp); // fileOutputRecording/ {fileNameIp} + _normalized_segregated_recording.txt
		/**************** Opening respective files. ****************/
			fp_ip = fopen(completePathIp, "r");				//to read input file
			fp_norm = fopen(completePathNorm, "w+");		//to save normalised samples
			fp_final_op = fopen(completePathFinOp, "w+");	//to save compelete file data from 1 to #files in one file only.
			fp_norm_seg = fopen(completePathNormSeg, "w");  //to save segregated recording from start to end
			
			if(fp_ip == NULL || fp_norm == NULL ||  fp_final_op==NULL || fp_norm_seg == NULL   ){ 
					perror("\nError: ");				fprintf(fp_final_op, "\n File Reading Error: ");
					printf("\n File Names are Input, NormIp, NormSeg, FinalOp : \n  %s, \n  %s, \n  %s, \n  %s \n", completePathIp, completePathNorm, completePathNormSeg,  fp_final_op );
						fprintf(fp_final_op, "\n File Names are Input, NormIp, NormSeg, FinalOp : \n  %s, \n  %s, \n  %s, \n  %s \n", completePathIp, completePathNorm, completePathNormSeg,  fp_final_op );
					getch();
					return EXIT_FAILURE;
			}

			if(fileCounter==11){
				 // writing into output file once at beginning.
			}
	
		printf("\n ----------------------- START - TESTING OF FILE: %s ----------------------- \n", fileNameIp);  
		fprintf(fp_final_op, "\n ----------------------- START - TESTING OF FILE: %s ----------------------- \n", fileNameIp);
		fprintf(fp_console, "\n ----------------------- START - TESTING OF FILE: %s ----------------------- \n", fileNameIp);

		/**************** DC Shift and Normalizing ****************/
			normalize_dcshift_samples();

		/**************** Frames ZCR and Energy. STE Marker ****************/
			zcr_energy_frames();

			if(segregate_speech){						//only if you want to segregate speech into separate file.
			/****************  calculating noise energy and threshold. ****************/
				noiseEnergy_thresholds_frames();						// if you want to calculate thresholds for zcr and energy
					
			/**************** start and end marker of speech ****************/
				marker_start_end_segregated();							//this and above func, if you want to detect start, end marker of speech, and to save it in separate file.
				fclose(fp_norm_seg);	// closing file stream
			}
			else
			   {
				   fclose(fp_norm_seg);		// closing file stream
				   remove(completePathNormSeg);		//removing unnecessory file created.
			   }
		/****************  Calculating Coefficients for Steady Frames of File ****************/
				calculateCoefficientsSteadyFramesTest();

			fclose(fp_norm);			// closing file stream, as no longer needed.
			remove(completePathNorm);	//comment it if you want to keep normalised data file.
			fclose(fp_ip); // closing file stream
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
																			STEP 4: Tokhura's distance
		-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			
			calculateTokhuraDistance();
		
			printf("\n\n\t RESULT ==> Minimum Tokhura Distance : %lf | Vowel Recognized ==> %c\n\t", d_tokhura_min, vowels[vowel_index_min]); 
				fprintf(fp_console, "\n\n\t RESULT ==> Minimum Tokhura Distance : %lf | Vowel Recognized ==> %c\n\t", d_tokhura_min, vowels[vowel_index_min]);
			for (int j = 0; j < totVowels; ++j)
			{	
				printf("%lf{%c}   ",dt_tokhura_v[j], vowels[j]);
				fprintf(fp_console, "%lf{%c}   ",dt_tokhura_v[j], vowels[j]);
			}
			//vowel recognized == current vowel file
			if(vowels[vowel_index_min] == vowels[v]) correct_count++;

				printf("\n File Testing Data and Result: %s \n", completePathFinOp); 
				printf("\n ----------------------- END - TESTING OF FILE  ----------------------- \n");  
 
				
					fprintf(fp_final_op, "\n\n\t RESULT ==> Minimum Tokhura Distance : %lf | Vowel Recognized ==> %c\n\t", d_tokhura_min, vowels[vowel_index_min]);
					fprintf(fp_final_op, "\n ----------------------- END - TESTING OF FILE  ----------------------- \n");

					
					fprintf(fp_console, "\n File Testing Data and Result: %s \n", completePathFinOp);
					fprintf(fp_console, "\n ----------------------- END - TESTING OF FILE  ----------------------- \n");

		}//end of filecounter loop test ------------------------------------------------------------------------------------------------------------------------------

		double accuracy = (double)(correct_count*1.0/NumOfFiles)*100;
		printf("\n\t For Vowel %c | Accuracy:  %0.2f %%\n\n", vowels[v], accuracy); 
			fprintf(fp_console, "\n\t\t\t For Vowel %c | Accuracy:  %0.2f %%\n\n", vowels[v], accuracy);

		overall_count +=correct_count;
		correct_count=0;

		fflush(fp_final_op);
		fclose(fp_final_op); // closing file stream
		
		system("pause");
	}//end of vowel loop test ------------------------------------------------------------------------------------------------------------------------------
	
	printf("\n-----------------------------------\n"); fprintf(fp_console, "\n-----------------------------------\n");

	double final_accuracy = (double)(overall_count*1.0/(NumOfFiles*totVowels))*100;
	printf("\n\t Overrall Accuracy:  %0.2f %%\n\n", final_accuracy); 
			fprintf(fp_console, "\n\t Overrall Accuracy:  %0.2f %%\n\n", final_accuracy);

	fflush(fp_console);
	fclose(fp_console); // closing file stream

	printf("\n---------------- END OF PROGRAM -------------------\n"); fprintf(fp_final_op, "\n---------------- END OF PROGRAM -------------------\n");
	printf("\nPress Enter To Close The Program\n");

	getch(); 
	return 0;
}
