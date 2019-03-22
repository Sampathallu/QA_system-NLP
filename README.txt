QA Systems

TEAM: HARSHITHA, SAMPATH

We implimented the paper with some minor modifications: https://www.cs.utah.edu/~riloff/pdfs/quarc.pdf and multiple rules in addition to it. 

• We begin with taking all questions from the set and storing the questions list. 
• Process the story and take up the main story, remove all the possible stop words from it.
• In a similar way, we process our question and remove stop words to retain the important keywords to be looked upon in an answer.
• We have implemented QUARC rule-based system to find the best possible answer for a given question. Each type of WH question looks for different type of answer. Using our stop word removal, we remove the stopwords and try to match the remaining words against words in the question.
• Each rule will be awarding specific number of points. Rule can assign four values – clue, good_clue, confident and slam_dunk.
• In case of who rules, it first checks if question does not contain any names, or if it contains, it looks for name or sentences the contain the word “name”.
• In case of what rules, it first does generic word matching, and then checks for sentences that contain date expression, it then addresses these questions by checking for ‘from’ and ‘call’ keyword, then it checks for names.
• In case of when rules, it always requires a time expression. It checks for duration and if it doesn’t find the duration or verbs such as ‘first’,’last’,’since’,’ago’, it checks for the verbs ‘begin’, ‘start’ even if there is mention of time.
• In case of where rules, it always requires a location. It checks for word matching, then it checks for location preposition such as ‘in’, ’at’, ‘near’ etc.
• In case of why rules, it checks for word match immediately before or after the sentence mostly matches the question, then it checks for sentence that precedes best word match sentence, then it checks for sentence that follows best word match. It then looks for sentence that has word ‘want’ and if ‘want’ is not present, it looks for sentence that contain ‘so’ or ‘because’.
• We calculate ner score by looking for possible ner tags that can be expected from a question and match it with keywords in the story.
• To find the most probable sentence answer in a paragraph, we have implemented the following methods

	- WordNET similarity
	- spacy similarity
	- bag of words score
	- nerBasedSimilarity
	- bi-grams score
	- tri-grams score
	- jaccard similarity
	- cosine similarity 	

• After finding the most probable sentence in the para, we have processed the output line based on the question types.
• We have tried implementing co reference resolution, but it could not help us boost the f-measure.
• By analyzing the answers for specific types of questions, we have implemented specific strategies to tackle each kind of question to derive the processed answer for each question type correspondingly.


•	We have used the following packages: 
•	Spacy
•	Spacy English words
•	NLTK
•	Stopwords, wordnet, wordNetLemmatizer 

• Tasks of team member:

• Harshitha: qWordsBasedExpectedNer, predictAnsForQuestionSet, bagofwords score calculation, calculation of question type score – whoscore, qbasedNerscore, wherescore, howscore, whatscore, whenscore, process the output line - processTheOpLine, predicting the answers - predictAns, handle f-score, precision and recall, n-grams, question-type based similarity.

• Sampath: processData, stopwords,  WordNet similarity, lemmatize, process and clean the question and answers, NERbasedsimilarity, retrieve golden answers, POSForLine, Jaccard similarity, Cosine Similarity, question-type based similarity, handle f-score, precision and recall, creating Readme file.

• It will approximately take 10 minutes for our system to process the stories.
 
• access the environment with command source 
        /home/u1210425/env/bin/activate.csh
• and to run the shell script run the command where input file is the quesions.txt file
        bash run.sh <inputfile> 
