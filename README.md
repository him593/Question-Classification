# Question-Classification
Classifies a question text in five categories: when, what, who, affirmation and unknown. 


## INSTRUCTIONS TO RUN ##
- Run pip install -r requirements.txt. Might haeve to download nltk data if you dont have it already. 
- Run classify_question.py as python classify_question.py
- At the first run it will train on the given data, report training and testing accuracies and runs the model on 200 sampled question from http://cogcomp.org/Data/QA/QC/train_1000.label and saves the result in the file 'test_file.csv'
- To run the model on keyboard input. Go to the classify_question.py file, go to the main function and comment out the call to train and test functions and uncomment the lines following them. Run the file. You will get a prompt to enter text and the output will follow. Ctrl+C to exit that.
- The models will only be trained once. If the train function is run again it will just reuse the saved models in the model directory.



A run of the model on some sample lines are as follows:

```
what time does the movie begin? when
what is the birthplace of john? what
Who was gandhi who
Is it morning or noon? affirmation
are you a smuggler? affirmation
what time does the train leave? when
what is the time? what


 
