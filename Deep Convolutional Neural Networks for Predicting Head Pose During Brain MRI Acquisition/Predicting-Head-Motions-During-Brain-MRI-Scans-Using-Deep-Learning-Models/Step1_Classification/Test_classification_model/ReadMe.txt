ReadMe:

Files:
1, 201995_cnn_classification.h5: the saved model that could classify if the head moves or not at some time.
2, test_classification.py: the python file to do data preprocessing and get prediction value/result.
3, run.sh: bash file to run the python file.

Output:
Log_($date).txt: the log file to monitor the process.
Note: For each subject, it cost almost 1.5 minuntes to get the result. Once it's done, it will print out "completed"

Target:
1, grep the overall accuracy, not_moving accuracy and moving accuracy from the log file
2, grep the number of total and correct not_moving labels and moving labels from the log file

How to run:
1, open terminal and copy the directory where you save the new subject data.
2, enter the directory where includes all three files shown above
3, type "./run.sh" + space + "the location of dataset you copy in the first step" 
   (Note: the location of dataset is the directory including all two csv files and directory of frames)
4, get result: type "grep subject_id Log_(date).txt"

Error may exist:
1, If the log file report any module cannot find, Please install that package
2, If the python3.6 does not exist, please modify the run.sh file: the Third line: change python 3.6 to your version installed.

   