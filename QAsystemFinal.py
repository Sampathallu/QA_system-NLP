#import all the required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import sys
import re
import re, math
from collections import Counter
from nltk.util import ngrams

#global variables
STORY = ".story"
ANSWERS = ".answers"
QUESTIONS = ".questions"

STOP_WORDS = set(stopwords.words('english')) 
SPACY_WORDS = spacy.load('en')

WORD = re.compile(r'\w+')
en_nlp = spacy.load('en')

#weights
CLUE = 3
GOOD_CLUE = 5
CONFIDENT = 8
SLAM_DUNK = 10
BOW_WEIGHT = 25
JACCARD_WEIGHT = 150
COSINE_WEIGHT = 200
BI_GRAMS_WEIGHT = 10
TRI_GRAMS_WEIGHT = 20

#questions constants
QUESTION = "Question"
QUESTION_ID = "QuestionID"
DIFFICULTY = "Difficulty"

#required NER and strings
QUESTION_TYPE = ["who", "where", "what", "how", "why", "when", "whose", "which", "howVerb"]
QUESTION_NER = {'what': ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"], 'who': ["PERSON", "NORP", "FAC", "ORG", "PRODUCT", "WORK_OF_ART"], 'where': ["GPE", "ORG", "LOC"], 'when': ["DATE", "TIME", "DURATION"], 'how':["MONEY", "QUANTITY", "CARDINAL", "PERCENT", "DURATION"], 'why': ["PERSON", "LOC", "FAC"], 'whose': ["PERSON", "ORG", "LOC", "LANGUAGE", "MONEY", "FAC"], 'which': ["GPE", "LOC","FAC","ORG", "NORP","EVENT","WORK_OF_ART"]}
QUESTION_WORDS_NER =  {"located":["GPE", "LOC"], "location":["GPE", "LOC"], "principal": ["PERSON"], "teacher": ["PERSON"], "doctor": ["PERSON"], "actor": ["PERSON"], "employee": ["PERSON"], "club": ["ORG", "PERSON"], "large":["QUANTITY","CARDINAL"], "big":["QUANTITY","CARDINAL"], "size":["QUANTITY","CARDINAL"], "small":["QUANTITY","CARDINAL"], "cost":["QUANTITY","CARDINAL"], "costing":["QUANTITY","CARDINAL"],"tall": ["QUANTITY","CARDINAL"], "paid": ["QUANTITY","CARDINAL", "MONEY"], "pay": ["QUANTITY","CARDINAL", "MONEY"], "paying": ["QUANTITY","CARDINAL", "MONEY"], "husband": ["PERSON"], "wife":["PERSON"], "boy":["PERSON"], "girl":["PERSON"], "baby":["PERSON"], "lady":["PERSON"], "men":["PERSON"], "man":["PERSON"], "kid":["PERSON"], "uncle":["PERSON"], "aunt":["PERSON"] , "age":["QUANTITY","CARDINAL"], "height":["QUANTITY","CARDINAL"], "weight":["QUANTITY","CARDINAL"], "earn": ["CARDINAL", "MONEY", "QUANTITY"], "money": ["QUANTITY","CARDINAL", "MONEY"], "donation": ["QUANTITY","CARDINAL", "MONEY"], "donated": ["QUANTITY","CARDINAL", "MONEY"], "donating": ["QUANTITY","CARDINAL", "MONEY"], "old": ["QUANTITY","CARDINAL"], "institute": ["NORP", "ORG"], "longs": ["QUANTITY","CARDINAL"], "native": ["PERSON", "NORP"], "country": ["GPE"], "often": ["CARDINAL", "TIME"], "built": ["CARDINAL", "DATE", "TIME"], "build": ["CARDINAL", "DATE", "TIME"], "die": ["CARDINAL", "TIME", "DATE"], "published": ["ORG"], "publishing": ["ORG"], "publish": ["ORG"], "owns": ["ORG"], "organization": ["ORG"], "name": ["PERSON", "ORG"], "instituted": ["CARDINAL",  "TIME", "DATE"], "journey": ["GPE", "LOC"], "target": ["CARDINAL"], "bombing": ["ORG", "FAC", "LOC"], "company": ["ORG"], "team": ["ORG", "PERSON"], "factor": ["NORP"], "date": ["DATE"], "minister": ["PERSON"], "air": ["DATE", "CARDINAL"], "judge": ["PERSON"], "time": ["CARDINAL", "DATE", "TIME"], "arrive": ["GPE", "LOC", "DATE", "TIME"], "arriving": ["CARDINAL", "DATE", "TIME"], "arrival": ["CARDINAL", "DATE", "TIME"], "year": ["CARDINAL", "DATE", "TIME"], "first": ["GPE", "PERSON", "ORG"], "highest": ["CARDINAL", "QUANTITY"],"high": ["CARDINAL", "QUANTITY"], "lowest": ["CARDINAL", "QUANTITY"], "low": ["CARDINAL", "QUANTITY"], "father": ["CARDINAL", "PERSON" ], "party": ["NORP", "ORG"], "found": ["GPE", "TIME", "DATE"], "find": ["GPE", "TIME", "DATE"], "live": ["TIME", "CARDINAL", "FAC", "DATE"], "homeland": ["GPE"], "birthday": ["CARDINAL", "TIME", "DATE"], "spokesman": ["PERSON", "NORP"],"donation": ["QUANTITY", "CARDINAL"], "countries": ["GPE", "LOC"], "often": ["ORDINAL", "CARDINAL", "TIME"],"die": ["TIME", "ORDINAL", "DATE"], "worth": ["MONEY", "QUANTITY", "CARDINAL"], "home": ["FAC", "GPE", "CARDINAL"], "population": ["QUANTITY", "CARDINAL"], "name": ["PERSON", "GPE", "ORG", "FAC", "PRODUCT"], "speed": ["QUANTITY", "CARDINAL"], "married": ["PERSON", "CARDINAL"],"launched": ["TIME", "QUANTITY", "DATE"], "launching": ["TIME", "QUANTITY", "DATE"], "launch": ["TIME", "QUANTITY", "DATE"],"mayor": ["PERSON", "NORP"], "noise": ["QUANTITY", "CARDINAL"], "place": ["LOC", "GPE", "ORG", "FAC"], "professor": ["PERSON"], "student": ["PERSON"],"president": ["PERSON"],"son": ["PERSON"],"daughter": ["PERSON"], "school": ["ORG"], "bank": ["ORG"], "office": ["ORG"], "college": ["ORG"], "university": ["ORG"], "director": ["PERSON"], "lawyer": ["PERSON","LAW"], "fined": ["TIME", "QUANTITY", "DATE"], "laws": ["LAW", "PERSON"], "meet": ["GPE", "LOC", "QUANTITY", "DATE", "TIME"], "elections": ["LOC", "GPE"], "election": ["LOC", "GPE"], "seats": ["QUANTITY"], "comets": ["QUANTITY"], "robots": ["QUANTITY", "ORDINAL"], "capital": ["GPE", "LOC"], "magnitude": ["QUANTITY", "ORDINAL"], "born": ["TIME", "DATE", "GPE", "LOC"], "soldiers": ["QUANTITY", "PERSON"], "island": ["GPE", "LOC", "QUANTITY"], "troops": ["QUANTITY", "PERSON"], "case": ["DATE", "TIME", "QUANTITY", "LAW"], "Territories": ["GPE", "LOC", "QUANTITY"], "originate": ["GPE", "LOC", "QUANTITY"], "fast": ["QUANTITY"], "people": ["QUANTITY"], "landed": ["TIME", "QUANTITY", "DATE"], "partitioned": ["TIME", "QUANTITY", "DATE"]}


HOW_EXCEPTION = ["would", "might", "often", "did", "is", "does", "do", "should", "has", "have", "will", "was", "are"]

def removeStopWords(data):
    dataWords = word_tokenize(data)
    return [w for w in dataWords if not w in STOP_WORDS] 

def isNotStopWord(word):
    return (word not in STOP_WORDS) and (word not in QUESTION_TYPE)

def lemmatizeWord(word):
    return (WordNetLemmatizer().lemmatize(word))

def cleanWord(word):
    alphabets = ['$', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    newWord = ""
    for ch in word:
        if ch.lower() in alphabets:
            newWord += ch
    return newWord.strip(" ")

def processQuestion(ques):
    [qType, ansNer] = expectedNer(ques)
    qKeyWords = cleanQuestion(ques)
    qWordsBasedNer = qWordsBasedExpectedNer(ques)
    return [qType, ansNer, qKeyWords, qWordsBasedNer]

def expectedNer(ques):
    for questionWord in ques.split(" "):
        if questionWord.lower() in QUESTION_NER.keys():
            qType = questionWord.lower()
            if questionWord.lower() == "how": #special how cases non quantitative are considered as why
                qWords = [x.lower() for x in ques.split(" ")]
                print(qWords)
                if qWords[qWords.index("how")+1].lower() in HOW_EXCEPTION:
                    if  qWords[qWords.index("how")+1].lower()  == "often":
                        qType = "when"
                    else:
                        qType = "howVerb"
            return [qType, QUESTION_NER[questionWord.lower()]]
    return["", []]

def qWordsBasedExpectedNer(ques):
    qWordsBasedNer = []
    for questionWord in ques.split(" "):
        lemmatizedWord  = lemmatizeWord(cleanWord(questionWord.lower()))
        if lemmatizedWord in QUESTION_WORDS_NER.keys():
            qWordsBasedNer = qWordsBasedNer + QUESTION_WORDS_NER[lemmatizedWord]
    return list(set(qWordsBasedNer))

def cleanQuestion(ques):
    qKeyWords = []
    for questionWord in ques.split(" "):
        if questionWord.lower() not in QUESTION_NER.keys():
            qKeyWords.append(cleanWord(questionWord))
    return removeStopWords(" ".join(qKeyWords))

def wordNetSimilarity(qWord, pWord):
    qWordSynsets = wordnet.synsets(qWord)
    pWordSynsets = wordnet.synsets(pWord)
    if qWord == pWord:
        return(1)
    elif qWord and pWord and len(qWord)>0 and len(pWord)>0 and len(qWordSynsets) > 0 and len(pWordSynsets) > 0:
        if len(qWordSynsets)> 0 and len(pWordSynsets) > 0 :
            try:
                return(qWordSynsets[0]).wup_similarity(pWordSynsets[0])
            except:
                return(0)
    return(0)

def spacySimilarity(spacyPTokens, spacyQTokens):
    spacySimilarity = 0
    for qWord in spacyQTokens:
        for pWord in spacyPTokens:
            qWord = cleanWord(str(qWord))
            pWord = cleanWord(str(pWord))
            if qWord == pWord:
                spacySimilarity += 1
            elif qWord and pWord and len(qWord)>0 and len(pWord)>0 and isNotStopWord(qWord) and isNotStopWord(pWord):
                    try:
                        spacySimilarity += qWord.similarity(qWord)
                    except:
                        spacySimilarity += 0
    return(spacySimilarity)

def bagOfWordsScore(qWord, pWord):
    return BOW_WEIGHT if lemmatizeWord(qWord) == lemmatizeWord(pWord) else 0

def nerForLine(line):
    tupleArray = []
    nerTuple = tuple()
    doc = SPACY_WORDS(line)
    for ent in doc.ents:
        nerTuple = ent.text,ent.label_
        tupleArray.append(nerTuple)
    return tupleArray

def get_jaccard_sim(str1, str2): 
    str1Words = set(str1.split(" ")) 
    str2Words = set(str2.split(" "))
    wordsIntersection = str1Words.intersection(str2Words)
    return float(len(wordsIntersection)) / (len(str1Words) + len(str2Words) - len(wordsIntersection))

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     return Counter(WORD.findall(text))

def nerBasedSimilarirty(paraWords, ansNer, qWordsBasedNer):
    nerBasedSimilarityScore = 0
    lineWordsNer = nerForLine(paraWords)
    for wordNer in lineWordsNer:
        if wordNer[1] in ansNer:
            nerBasedSimilarityScore += CONFIDENT
    lemmatizedWords = []
    for word in paraWords:
        lemmatizedWords.append(lemmatizeWord(cleanWord(word)))
    lineWordsNer = nerForLine(lemmatizedWords)
    for wordNer in lineWordsNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
            nerBasedSimilarityScore += SLAM_DUNK
    return nerBasedSimilarityScore

def whoScore(qKeyWords, paraLine, ansNer, pKeyWords):
    score = 0
    lineNer = nerForLine(paraLine)
    for wordNer in lineNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
            score += CONFIDENT
    if "name" in [x.lower() for x in pKeyWords]:
        score += GOOD_CLUE
    return score

def whichScore(qKeyWords, paraLine, ansNer, pKeyWords):
    score = 0
    lineNer = nerForLine(paraLine)
    for wordNer in lineNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
            score += SLAM_DUNK
    return score

def whereScore(qKeyWords, paraLine, ansNer, pKeyWords):
    score = 0
    lineNer = nerForLine(paraLine)
    for word in paraLine.split(" "):
        word = cleanWord(word)
        if word.lower() in ["in", "at", "near", "inside", "outside", "beside", "behind"]:
            score += GOOD_CLUE
    for wordNer in lineNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
            score += CONFIDENT
    return score

def posForLine(line):
    posArray= []
    posTuple = tuple()
    doc = SPACY_WORDS(line)
    for token in doc:
        posTuple = token.pos_
        posArray.append(posTuple)
    return posArray

def howVerbScore(qKeyWords, paraLine, ansNer, pKeyWords):
    score = 0
    for word in posForLine(paraLine):
        if word == "VERB":
            score += SLAM_DUNK
    return score

def howScore(qKeyWords, paraLine, ansNer, pKeyWords):
   score = 0
   lineNer = nerForLine(paraLine)
   for word in paraLine.split(" "):
       word = cleanWord(word)
       if word.lower() in ["well", "bad", "good", "growing", "decreasing", "decrease", "increase", "increasing", "year", "between", "from", "first", "last", "since", "ago", "more", "less"]:
           score += CLUE
   for wordNer in lineNer:
       if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
           score += CONFIDENT
   return score

def whatScore(qKeyWords, paraLine, ansNer, pKeyWords):
    score = 0
    applyKindRule = False
    applyNameRule = False
    applyDateRule = False
    for word in qKeyWords:
        if word.lower() == "kind":
            applyKindRule = True
        elif word.lower().find("name") != -1:
            applyNameRule = True
        elif word.lower() in ["month", "year", "date", "time"]:
            applyDateRule = True
    if applyDateRule:
        for word in paraLine.split(" "):
            word = cleanWord(word)
            if word.lower().find('today') != -1 or word.lower().find('yesterday') != -1 or word.lower().find('tomorrow') != -1 or word.lower().find('night') != -1 or word.lower().find('day') != -1:
                score += CLUE
    if applyKindRule:
        for word in paraLine.split(" "):
            word = cleanWord(word)
            if word.lower().find('call') != -1 or  word.lower().find('from') != -1:
                score += GOOD_CLUE
    if applyNameRule:
        for word in paraLine.split(" "):
            word = cleanWord(word)
            if word.lower().find('name') != -1 or word.lower().find('call') != -1 or word.lower().find('known') != -1:
                score += SLAM_DUNK
    return score

def whenScore(qKeyWords, paraLine, ansNer, pKeyWords, ques, jaccardscore, cosine):
    score = 0
    calculateBow = False
    applyRule2 = False
    applyRule3 = True
    lineNer = nerForLine(paraLine)
    for wordNer in lineNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
            score += SLAM_DUNK
            calculateBow = True 
    if calculateBow:
        for qWord in qKeyWords:
            for pWord in paraLine.split(" "):
                pWord = cleanWord(pWord)
                score += bagOfWordsScore(qWord, pWord)
        #jaccard similarity
        score += jaccardscore * JACCARD_WEIGHT  
        #conise similarity
        score += cosine * COSINE_WEIGHT
        #bigrams score
        score += biGramsScore(paraLine, ques["Question"])
        #trigrams score
        score += triGramsScore(paraLine, ques["Question"])
    for word in qKeyWords:
        if word.lower() in ["first", "last", "since", "ago"]:
            applyRule2 = True
        elif word.lower() in ["start", "begin", "since", "year"]:
            applyRule3 = True
    if applyRule2:
        for word in paraLine.split(" "):
            word = cleanWord(word)
            if word.lower() in ["first", "last", "since", "ago"]:
                score += (SLAM_DUNK*2)
    if applyRule3:
        for word in paraLine.split(" "):
            word = cleanWord(word)
            if word.lower() in ["start", "begin", "since", "year"]:
                score += (SLAM_DUNK*2)
    return score

def qWordsBasedNerScore(line, qWordsBasedNer):
    score = 0
    lineNer = nerForLine(line)
    for wordNer in lineNer:
        if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in qWordsBasedNer:
            score += SLAM_DUNK  
    return score

def questionBasedScore(qKeyWords, paraLine, qType, ansNer, pKeyWords, ques, jaccardscore, cosine):
    score = 0
    if qType == "who" or qType == "whose":
        score += whoScore(qKeyWords, paraLine, ansNer, pKeyWords)
    elif qType == "where":
        score += whereScore(qKeyWords, paraLine, ansNer, pKeyWords)
    elif qType == "what":
        score += whatScore(qKeyWords, paraLine, ansNer, pKeyWords)
    elif qType == "when":
        score += whenScore(qKeyWords, paraLine, ansNer, pKeyWords, ques, jaccardscore, cosine)
    elif qType == "how":
        score += howScore(qKeyWords, paraLine, ansNer, pKeyWords)
    elif qType == "which":
        score += whichScore(qKeyWords, paraLine, ansNer, pKeyWords)
    elif qType == "howVerb":
        score += howVerbScore(qKeyWords, paraLine, ansNer, pKeyWords)
    return score

def addDolloar(line, word):
    index = line.find(word)
    if index == -1 or index == 0:
        return False
    return line[index-1] == "$"

def numThere(s):
    return any(i.isdigit() for i in s)

def nounNotInQuestion(word, qKeyWords):
    for x in word:
        if x in qKeyWords:
            del word[word.index(x)]
    return word

def processTheOpLine(line, qType, ansNer, qKeyWords):
    lineWords = []
    if qType == "when":
        lineNer = nerForLine(line)
        for wordNer in lineNer:
            if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
                word = str(wordNer[0].strip(" ")).split(" ")
                if wordNer[1] == "MONEY" and addDolloar(line, word[0]):
                    word[0] = "$"+word[0]
                elif numThere(wordNer[0]):
                    temp = []
                    for singleWord in word:
                        temp.append(singleWord.replace("-"," "))
                    word = temp
                lineWords = lineWords + nounNotInQuestion(word, qKeyWords)
    elif qType == "how":
        lineNer = nerForLine(line)
        for wordNer in lineNer:
            if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
                word = str(wordNer[0].strip(" ")).split(" ")
                if wordNer[1] == "MONEY" and addDolloar(line, word[0]):
                    word[0] = "$"+word[0]
                elif numThere(wordNer[0]):
                    temp = []
                    for singleWord in word:
                        temp.append(singleWord.replace("-"," "))
                    word = temp
                lineWords = lineWords + nounNotInQuestion(word, qKeyWords)
        if len(lineWords) > 0:
            for word in line.split(" "):
                cWord = cleanWord(word)
                if cWord.lower() in ["more", "less", "several", "few"]:
                    lineWords = lineWords + (word.split(" ")) 
    elif qType == "who" or qType == "whose":
        lineNer = nerForLine(line)
        for wordNer in lineNer:
            if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
                word = str(wordNer[0].strip(" ")).split(" ")
                if wordNer[1] == "MONEY" and addDolloar(line, word[0]):
                    word[0] = "$"+word[0]
                elif numThere(wordNer[0]):
                    temp = []
                    for singleWord in word:
                        temp.append(singleWord.replace("-"," "))
                    word = temp
                lineWords = lineWords + nounNotInQuestion(word, qKeyWords)
        if len(lineWords) > 0:
            for word in line.split(" "):
                cWord = cleanWord(word)
                if cWord.lower() in [ "boy", "girl", "women", "men", "lady", "kid", "children", "and", "are", "with"]:
                    lineWords = lineWords + (word.split(" ")) 
    elif qType == "where":
        lineNer = nerForLine(line)
        for wordNer in lineNer:
            if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
                word = str(wordNer[0].strip(" ")).split(" ")
                if wordNer[1] == "MONEY" and addDolloar(line, word[0]):
                    word[0] = "$"+word[0]
                elif numThere(wordNer[0]):
                    temp = []
                    for singleWord in word:
                        temp.append(singleWord.replace("-"," "))
                    word = temp
                lineWords = lineWords + nounNotInQuestion(word, qKeyWords)
        if len(lineWords) > 0:
            for word in line.split(" "):
                cWord = cleanWord(word)
                if cWord.lower() in ["in", "at", "on", "near"]:
                    lineWords = lineWords + (word.split(" ")) 
    elif qType == "which":
        lineNer = nerForLine(line)
        print(lineNer)
        for wordNer in lineNer:
            if len(wordNer[0].strip(" ")) > 0  and wordNer[1] in ansNer:
                word = str(wordNer[0].strip(" ")).split(" ")
                if numThere(wordNer[0]):
                    temp = []
                    for singleWord in word:
                        temp.append(singleWord.replace("-"," "))
                    word = temp
                lineWords = lineWords + nounNotInQuestion(word, qKeyWords)
    if len(lineWords) > 0:
        return " ".join(list(set(lineWords)))
    return line

def addDolloar(line, word):
    index = line.find(word)
    if index == -1 or index == 0:
        return False
    return line[index-1] == "$"

def biGramsScore(line, question):
    token=nltk.word_tokenize(line)
    lineBigrams=ngrams(token,2)
    token=nltk.word_tokenize(question)
    questionBigrams=ngrams(token,2)
    return len(list(set(list(lineBigrams)) & set(list(questionBigrams))) ) * BI_GRAMS_WEIGHT

def triGramsScore(line, question):
    token=nltk.word_tokenize(line)
    lineTrigrams=ngrams(token,3)
    token=nltk.word_tokenize(question)
    questionTrigrams=ngrams(token,3)
    return len(list(set(list(lineTrigrams)) & set(list(questionTrigrams))) ) * TRI_GRAMS_WEIGHT

def predictAns(ques, story):
    [qType, ansNer, qKeyWords, qWordsBasedNer]=  processQuestion(ques["Question"])
    para = story["TEXT"].replace('\n', ' ')
    maxLinesReationScore = 0
    maxScoreLine = ""
    spacyQTokens = SPACY_WORDS(ques["Question"]) 
    for paraLine in para.split("."):
        linesReationScore = 0
        if paraLine != "":
            jaccardques = " ".join(qKeyWords)
            jaccardscore = get_jaccard_sim(jaccardques,paraLine)
            vector1 = text_to_vector(jaccardques)
            vector2 = text_to_vector(paraLine)
            cosine = get_cosine(vector1, vector2)
            pKeyWords = []
            for word in removeStopWords(paraLine.strip(" ")):
                pKeyWords.append(cleanWord(word))
            if len(pKeyWords) > 2:
                for qWord in qKeyWords:
                    for pWord in pKeyWords:
                        #wordnet simlarity
                        wordNetSimilarityScore = wordNetSimilarity(qWord, pWord)
                        if wordNetSimilarityScore:
                            linesReationScore += wordNetSimilarityScore
                        #bow similarity
                        if qType != "when":
                            linesReationScore += bagOfWordsScore(qWord, pWord)
                #question based score
                linesReationScore += questionBasedScore(qKeyWords, paraLine, qType, ansNer, pKeyWords, ques, jaccardscore, cosine)
                #spacy similarity
                spacyPTokens = SPACY_WORDS(paraLine)
                #question based NER
                linesReationScore += qWordsBasedNerScore(paraLine, qWordsBasedNer)
                #spacy based words similarity
                linesReationScore += spacySimilarity(spacyPTokens, spacyQTokens)

                if qType != "when":
                    #jaccard similarity
                    linesReationScore += jaccardscore * JACCARD_WEIGHT  
                    #conise similarity
                    linesReationScore += cosine * COSINE_WEIGHT
                    #bigrams score
                    linesReationScore += biGramsScore(paraLine, ques["Question"])
                    #trigrams score
                    linesReationScore += triGramsScore(paraLine, ques["Question"])
                if maxLinesReationScore < linesReationScore:
                    maxScoreLine = paraLine
                    maxLinesReationScore = linesReationScore
    print(maxScoreLine)
    return  processTheOpLine(maxScoreLine, qType, ansNer, qKeyWords)

def processData(dataFileName):
    try:
        dataFileObject = open(dataFileName, "r")
    except:
        print("file "+dataFileName+" missing")
        return
    dataFile = ""
    for dataLine in dataFileObject:
        dataFile += dataLine
    data = []
    flag = False
    text = ""
    for dataSet in dataFile.split("\n\n"):
        dataAttr = {} 
        for attr in dataSet.split("\n"):
            if flag:
                text += " "+ attr
                continue
            if len(attr.split(":")) > 1:
                if attr.split(":")[0] == "TEXT":
                    flag = True
                    continue
                if len(attr.split(":")) == 2:
                    dataAttr[attr.split(":")[0]] = attr.split(":")[1].strip(" ") 
                else:
                    dataAttr[attr.split(":")[0]] = ":".join(attr.split(":")[1:]).strip(" ")
                    
        if dataAttr != {}:
            data.append(dataAttr)
    if flag:
        data[0]["TEXT"] = text
    return data       

def getGoldenAns(goldenAnswers, questionID):
    for ans in goldenAnswers:
        if ans['QuestionID'] == questionID:
            return ans
    return ""

def predictAnsForQuestionSet(dirPath, answersFile, dataFileName):
    questions =  processData(dirPath+dataFileName+QUESTIONS)
    goldenAnswers =  processData(dirPath+dataFileName+ANSWERS)
    story =  processData(dirPath+dataFileName+STORY)[0]
    for ques in questions:
        if QUESTION in ques.keys():
            goldenAns = getGoldenAns(goldenAnswers, ques['QuestionID'])
            predictedAns = predictAns(ques, story)
            answersFile.write("QuestionID: "+ ques['QuestionID']+"\n")
            answersFile.write("Answer: "+ predictedAns+"\n\n")
            print("QuestionID: "+ ques['QuestionID']+"\n")
            print("Question: "+ ques['Question']+"\n")
            print("Answer: "+ predictedAns+"\n")
            print("Golden Ans: "+ goldenAns['Answer']+"\n\n")

questionsFileName = sys.argv[1]
questionsFile = open(questionsFileName, "r")
answersFile= open("answers.txt","w+")
count = 0
dirPath = ''
for questionsLine in questionsFile:
    if count == 0:
        dirPath = questionsLine[:-1]
    else:
        predictAnsForQuestionSet(dirPath, answersFile, questionsLine[:-1])
    count += 1
answersFile.close()