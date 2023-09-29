# ChatBOT
Created a chatbot using Python. It takes in the user input, and stores the user input in a csv file

#SETUP

1. Download all the files

2. Open command prompt/terminal. Create a new directory where all the downloaded files should be stored.

3. Change your current directory to newly create directory, and run the command: python3 -m venv venv

4. After that, run the command: . venv/bin/activate

5. Now you need to install some core dependencies. 

pip install nltk
pip install torch torchvision torchaudio
pip install pandas
pip install numpy

(Note: If you are installing nltk for the first time, you need to run the following commands in your terminal: 1. python3 
                                                                                                               2. import nltk
                                                                                                               3. nltk.download('punkt')

6. After that, run these commands: 1. python3 train.py
                                   2. python3 chat.py

7. After the user enters all the answers, a file "output.csv" will be created
