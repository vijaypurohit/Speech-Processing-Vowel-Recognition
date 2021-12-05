>Speech Processing CS 566
>
>**Assignment 03 (Vowel Recognition)**
>
>Roll No: 214101058 MTech CSE'23 IITG | Vijay Purohit

----------------------------------------------------
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
		*   (optional when var segregate_speech=TRUE (Line #93))
				Segregated Speech data of each file analysed for testing/debugging purpose. Can be played in Cool Edit.
				Folder:  fileOutputRecording[] = "output_recordings/";
				Format: {214XXXXXX} + _ + {a} + _ + {1} + _normalized_segregated_recording.txt
	Details About Each File:
		*   	R1:Format: {214XXXXXX} + _ + {a} + _avg_ci_reference.txt
				Contains final Avg Ci (5*12) of first 10 files of each vowel.
				Total Five Files.
		*   	R2:Format: {214XXXXXX}+ _ + {a} + _final_analysis.txt
				It contains values of DC shift, normalization factor, STE marker Frame of each 10 files.
				Contains Coefficients Values of each 10 files and For each 5 frame.
				Finally the Avg C_i values computed from 10 files in proper readable format.
				Total Five Files.
		*   	T1:Format: {214XXXXXX} + _ + a + {1} + _test_result.txt
				It contains values of DC shift, normalization factor, STE marker Frame of test file.
				Contains Coefficients Values of each 5 frame.
				Finally the tokhura distance calculated with respect to each frame and each vowel reference file.
				Result of minimum tokhura distance and vowel recognized.
				One for each test file.
		*   	T2:Format:  _ + {214XXXXXX} +_aeiou_TEST_RESULT_OUTPUT.txt
				Console Output of all the test file result. (10files*5vowels)
				Result = Their Tokhura Distance and vowel recognized. File Testing data path.
				Accuracy with respect to 10 vowel files and overall accuracy in the end.
				Single file.

----------------------------------------------------
### Instructions to execute Code.
1. Open it in Visual Studio 2010. Main file: SP_AS03.cpp
2. Input format is given above.
   * Change Roll Number Variable rollNum[](Line #55), if required for different files. 
   * Remember vowel data should be present in folder provided by array filePathVowels[] (Line #89).
   * Make Sure files are in proper format as mentioned. Otherwise some logic at (Line #679) needed to be modified.
3. Compile it and Run. Console window will show up.
   * First is the Validation of Calculation of Coefficients. Press Enter.
   * Second is the Generation of Reference file for Each vowel. Press Enter to generate for Each Vowel. (Folder by: fileOutputRef[])
   * Third is the Testing of Remaining file of each vowel. Press Enter to test for Each Vowel. (Folder by: fileOutputTest[])
   * Final Output will be shown in console.
   * As well as it is saved in file [_ + {214XXXXXX} +_aeiou_TEST_RESULT_OUTPUT.txt] in (Folder by: fileOutputTest[])
----------------------------------------------------
THE END.
