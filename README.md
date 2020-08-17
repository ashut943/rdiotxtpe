# Radiology-Text-Interpreter-Parser
_Navigate to the PDF Document **FinalPleuralEffusionProjectReport.pdf** to read all the results, detailed methodology, insights, and inspirations for this project_
##  Introducton:
A Rule-Based NLP approach to classifying Free-Text Radiology Reports based on existence of significant evidence for pleural effusion

Radiological reports, particularly the free-text, are a good source of clinical data which can be used to assist with
surveillance of disease. Pleural Effusion and other radiological findings on chest X-ray or chest computed tomography (CT) scans are
one type of relevant result to, both, health services and the medical community at large. In this study, we examined the ability of a
Hybrid system to identify Pleural Effusion from free-text radiological reports. We used a hybrid of a machine learning and rule-based
NLP system. The system encoded the reports, and then a defined set of rules were created aimed at the identification of the pleural
effusion. The rules were executed against the encodings of the radiological reports, followed by further classification. Four different
methods for classification were used to compare and conclude the best approach. The accuracy of the reports was compared with a
Clinician review of the Radiological Reports. We find that NLP based computable rules are accurate enough for the automated biosurveillance
of Pleural Effusion from radiological reports. However, this requires further validation with multiple large databases and
more diverse database

We use the NLP to detect Pleural Effusion (which shall be referred to as PE in this document) from the text of given radiology reports. We approach the problem by initially modelling the PE diagnosis as a set of different concepts, make a dictionary of words and phrases specific to the same and then using these as rules for the study.

The approach used is Machine Learning on Rule Based approach. As stated in a study by W. Christopher Baughman , Eamon Johnson and Gultekin Ozsoyoglu, titled Mixing Domain Rules with Machine Learning for Radiology Text Classification, using binary rule verdicts on records as features for machine learning is a simple task: the rule verdict may be added to the feature vector along with the other training and test inputs, and subjected to the same learning scheme. Rules producing continuous output may be harder to integrate into the learning process due to potential normalization issues between the training data and test data. 

Our approach was thus to make pre-defined rules, add it to the feature vector and then using a supervised learning method, classify data on Pleural Effusion as either 0-indicating absent (pleural effusion is absent)- or 1-indiciating there is some evidence indicating Pleural Effusion’s presence.

## Method:

### Pre-processing: 

Before anything, one important thing to do was pre-process the data. There were few stages that I decide in pre-processing the data specific to this study
Firstly we made a RTF Tag Remover. Since each of these documents were of the Rich Text Format, it was quintessential to remove these RTF specific tags to get some sort of meaningful result. 

This was followed by normal pre-processing steps such as changing all the text to lower case. After this, I made a report-specific list of ‘stopwords’-words that really didn’t contribute much to the meaning of the sentence. These were the same as those normally used in libraries, but certain words such as ‘no’ and ‘not’ were important, as defined later.

Using a Rule-Based Approach meant we needed to make a negation detector from the very scratch. Although the method used was grossly inadequate, we proceeded with it(the possible suggestions and continuations are mentioned later in this report)
Using a Rule-Based approach furthermore also meant that we had to define a dictionary of words that were commonly occurring in reports and that were related to the objective – to detect pleural effusion.

Thus, after manually analysing the pre-annotated reports, we decided to proceed by, initially,making a dictionary of the most frequently occurring unigrams, bigrams, trigrams, and quadrigrams from the list of words that were important features of Pleural Effusion-such as “left upper lobe”. We found occurrence of such words, and concatenated them into one. So, ‘left upper lobe’ became ‘left_upper_lobe’.

We had a similar approach to negation detection. So for example, if there was a phrase ‘no evidence’, it indicated a negation, and thus we concatenated them into one – ‘no_evidence’ – and proceeded as detailed below. This method was opted for because the structure of the sentences wasn’t very complex. Although, yes it is still inadequate, it wasn’t the worst method possible.


## Mapping and Tagging of words:
After the pre-processing stage, all the important n-grams were now one single words, thus we could proceed with the mapping and tagging phase.
We used the manually analysed list of important features (such as ‘left_pleural_effusion’), and indicated them as ‘FTR’ (meaning feature). Similarly, for the negations indicating no evidence, we now replaced the with the words ‘SAFE’. This we did for words that indicated ‘RISK’ and adjectives as well(which were annotated as ‘ADJ’).

This process of mapping and tagging of words was essential. We had now defined our rules – the existence of these defined dictionary of words- and had implemented it, essentially, in our pre-processing step to make the learning easier. 

Evidently, there were certain words that we must have missed, or those that were redundant didn’t contribute much to our study. To handle this, we used Rule-Association Mining.

We used Python’s Apyori Librari’s Apiori’s algorithm to conduct Rule-Association mining with the pre-defined features to get a sense of the applicability of the features we were using. This alongside a manual analysis of the database led to a more robust dictionary at the end. 

However, there are still problems in this method. The applicability and performance of this method is now solely dependant on one factor and one factor alone – the robustness of the Dictionary we used. Since we were only analysis presence of Pleural Effusion meant that we could do manual checking and make a dictionary manually. However, the case was that we were making it a more general program, another approach or a pre-defined database of medical words would have become necessary.

## Learning Phase: 
We tried multiple different approaches before sticking with one supervised learning algorithm-Logistic Regression. 

But before going directly to logistic regression, we needed to vectorise our corpus. We use a Count Vectorizer to vectorize our edited corpus of data into a count-vectorizer. Then, we fed this into the Logistic Regression Algorithm and trained it using this Count Vectorizer.



# Initial Test:
For initial Testing, we initially did a 80-20 train-test split. Vectorized the Training data, and then did a transform or the testing data. We then trained our logistic regression algorithm using a train data, and then initially tested it using the test data.

We used Random Test-Train Data split, and then analysed the results.
