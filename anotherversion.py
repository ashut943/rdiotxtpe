from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from math import *
from numpy import *
from sklearn.pipeline import Pipeline
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import os
import string
import gensim
from gensim.models import Word2Vec
import pandas as pd
import random
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import nltk.data
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from collections import Counter
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from sklearn.metrics import confusion_matrix

def striprtf(text):
   pattern = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I)
   # control words which specify a "destionation".
   destinations = frozenset((
      'aftncn','aftnsep','aftnsepc','annotation','atnauthor','atndate','atnicn','atnid',
      'atnparent','atnref','atntime','atrfend','atrfstart','author','background',
      'bkmkend','bkmkstart','blipuid','buptim','category','colorschememapping',
      'colortbl','comment','company','creatim','datafield','datastore','defchp','defpap',
      'do','doccomm','docvar','dptxbxtext','ebcend','ebcstart','factoidname','falt',
      'fchars','ffdeftext','ffentrymcr','ffexitmcr','ffformat','ffhelptext','ffl',
      'ffname','ffstattext','field','file','filetbl','fldinst','fldrslt','fldtype',
      'fname','fontemb','fontfile','fonttbl','footer','footerf','footerl','footerr',
      'footnote','formfield','ftncn','ftnsep','ftnsepc','g','generator','gridtbl',
      'header','headerf','headerl','headerr','hl','hlfr','hlinkbase','hlloc','hlsrc',
      'hsv','htmltag','info','keycode','keywords','latentstyles','lchars','levelnumbers',
      'leveltext','lfolevel','linkval','list','listlevel','listname','listoverride',
      'listoverridetable','listpicture','liststylename','listtable','listtext',
      'lsdlockedexcept','macc','maccPr','mailmerge','maln','malnScr','manager','margPr',
      'mbar','mbarPr','mbaseJc','mbegChr','mborderBox','mborderBoxPr','mbox','mboxPr',
      'mchr','mcount','mctrlPr','md','mdeg','mdegHide','mden','mdiff','mdPr','me',
      'mendChr','meqArr','meqArrPr','mf','mfName','mfPr','mfunc','mfuncPr','mgroupChr',
      'mgroupChrPr','mgrow','mhideBot','mhideLeft','mhideRight','mhideTop','mhtmltag',
      'mlim','mlimloc','mlimlow','mlimlowPr','mlimupp','mlimuppPr','mm','mmaddfieldname',
      'mmath','mmathPict','mmathPr','mmaxdist','mmc','mmcJc','mmconnectstr',
      'mmconnectstrdata','mmcPr','mmcs','mmdatasource','mmheadersource','mmmailsubject',
      'mmodso','mmodsofilter','mmodsofldmpdata','mmodsomappedname','mmodsoname',
      'mmodsorecipdata','mmodsosort','mmodsosrc','mmodsotable','mmodsoudl',
      'mmodsoudldata','mmodsouniquetag','mmPr','mmquery','mmr','mnary','mnaryPr',
      'mnoBreak','mnum','mobjDist','moMath','moMathPara','moMathParaPr','mopEmu',
      'mphant','mphantPr','mplcHide','mpos','mr','mrad','mradPr','mrPr','msepChr',
      'mshow','mshp','msPre','msPrePr','msSub','msSubPr','msSubSup','msSubSupPr','msSup',
      'msSupPr','mstrikeBLTR','mstrikeH','mstrikeTLBR','mstrikeV','msub','msubHide',
      'msup','msupHide','mtransp','mtype','mvertJc','mvfmf','mvfml','mvtof','mvtol',
      'mzeroAsc','mzeroDesc','mzeroWid','nesttableprops','nextfile','nonesttables',
      'objalias','objclass','objdata','object','objname','objsect','objtime','oldcprops',
      'oldpprops','oldsprops','oldtprops','oleclsid','operator','panose','password',
      'passwordhash','pgp','pgptbl','picprop','pict','pn','pnseclvl','pntext','pntxta',
      'pntxtb','printim','private','propname','protend','protstart','protusertbl','pxe',
      'result','revtbl','revtim','rsidtbl','rxe','shp','shpgrp','shpinst',
      'shppict','shprslt','shptxt','sn','sp','staticval','stylesheet','subject','sv',
      'svb','tc','template','themedata','title','txe','ud','upr','userprops',
      'wgrffmtfilter','windowcaption','writereservation','writereservhash','xe','xform',
      'xmlattrname','xmlattrvalue','xmlclose','xmlname','xmlnstbl',
      'xmlopen',
   ))
   # Translation of some special characters.
   specialchars = {
      'par': '\n',
      'sect': '\n\n',
      'page': '\n\n',
      'line': '\n',
      'tab': '\t',
      'emdash': u'\u2014',
      'endash': u'\u2013',
      'emspace': u'\u2003',
      'enspace': u'\u2002',
      'qmspace': u'\u2005',
      'bullet': u'\u2022',
      'lquote': u'\u2018',
      'rquote': u'\u2019',
      'ldblquote': u'\201C',
      'rdblquote': u'\u201D', 
   }
   stack = []
   ignorable = False       # Whether this group (and all inside it) are "ignorable".
   ucskip = 1              # Number of ASCII characters to skip after a unicode character.
   curskip = 0             # Number of ASCII characters left to skip
   out = []                # Output buffer.
   for match in pattern.finditer(text):
      word,arg,hex,char,brace,tchar = match.groups()
      if brace:
         curskip = 0
         if brace == '{':
            # Push state
            stack.append((ucskip,ignorable))
         elif brace == '}':
            # Pop state
            ucskip,ignorable = stack.pop()
      elif char: # \x (not a letter)
         curskip = 0
         if char == '~':
            if not ignorable:
                out.append(u'\xA0')
         elif char in '{}\\':
            if not ignorable:
               out.append(char)
         elif char == '*':
            ignorable = True
      elif word: # \foo
         curskip = 0
         if word in destinations:
            ignorable = True
         elif ignorable:
            pass
         elif word in specialchars:
            out.append(specialchars[word])
         elif word == 'uc':
            ucskip = int(arg)
         elif word == 'u':
            c = int(arg)
            if c < 0: c += 0x10000
            if c > 127: out.append(chr(c))
            else: out.append(chr(c))
            curskip = ucskip
      elif hex: # \'xx
         if curskip > 0:
            curskip -= 1
         elif not ignorable:
            c = int(hex,16)
            if c > 127: out.append(chr(c))
            else: out.append(chr(c))
      elif tchar:
         if curskip > 0:
            curskip -= 1
         elif not ignorable:
            out.append(tchar)
   return ''.join(out)

oof='./train_pe.csv'
def prerprocess(pth):
    import re
    data = pd.read_csv(pth)
    del data['Identifier']
    workabledata=data.copy()
    twword=[('mid','zones'),('pleural', 'effusion'),('right', 'lobe'),('lower', 'lobe'),('middle', 'lobe'),('upper', 'lobe'),('left', 'lobe'),('right', 'side'),('left', 'side'),('left', 'kidney'),('right', 'kidney'),('wall', 'thickening'),('bilateral', 'kidneys'),('focal', 'lesion'),('urinary', 'bladder'),('gall', 'bladder'),('mass', 'seen'),('wall', 'oedema'),('lymph', 'nodes'),('thick', 'walled'),('free', 'fluid'),('soft', 'tissues'),('bladder', 'wall')]
    oofsdfd=[]
    for x in twword:
        a=x[0]+'_'+x[1]
        oofsdfd.append(a)
    df = pd.read_csv(pth, usecols=[2])
    for i,j in df.iterrows(): 
        text=df.Result[i]
        df.Result[i]=striprtf(text)
    for i,j in df.iterrows(): 
        text=df.Result[i]
        df.Result[i]=text.lower()
    to_remove = ['no', 'nor','there','is']
    new_stopwords = set(stopwords.words('english')).difference(to_remove)
    for i,j in df.iterrows():                  
        text=df.Result[i]
        words=text.split()
        meaningful_words = [w for w in words if not w in new_stopwords]   
        df.Result[i]=(" ".join( meaningful_words))
    test_related_pe=['CT Thorax / Chest - HRCT Contrast','CT Thorax / Chest - HRCT Plain','CT Thorax / Chest - Plain','CT Whole Abdomen - Plain','CT Whole Abdomen with Contrast','CT Whole Thorax (Contrast)','CT Whole Thorax (Plain)','MRI Cholangiography (MRCP)','Ultrasound Chest','Ultrasound KUB','Ultrasound Whole Abdomen','USG Chest','USG Upper Abdomen','USG Whole Abdomen','X-ray (any Portable single exposure)','X-Ray Chest PA/AP View']
    twword=[('mid','zones'),('pleural', 'effusion'),('right', 'lobe'),('lower', 'lobe'),('middle', 'lobe'),('upper', 'lobe'),('left', 'lobe'),('right', 'side'),('left', 'side'),('left', 'kidney'),('right', 'kidney'),('wall', 'thickening'),('bilateral', 'kidneys'),('focal', 'lesion'),('urinary', 'bladder'),('gall', 'bladder'),('mass', 'seen'),('wall', 'oedema'),('lymph', 'nodes'),('thick', 'walled'),('free', 'fluid'),('soft', 'tissues'),('bladder', 'wall')]
    thrword=[('left', 'upper_lobe'),('right', 'upper_lobe'),('right','pleural_effusion'),('gall_bladder', 'walls'),('left', 'pleural_effusion')]
    s=workabledata.copy()
    sd=workabledata[workabledata.Pleural_effusion>0]
    sdt=sd.copy()
    ops=1
    sd = sd.reset_index(drop=True)
    oofsdf=[]
    for x in oofsd:
        txt=x
        if(txt[0]=='no'):
            oofsdf.append(txt)
    del s['Result']
    s=pd.concat([s,df],axis=1)
    okbad=["x-ray - chest (portable) results","hrct chest (non contrast)","chest x-ray pa view ","usg whole abdomen","results","report mri brain","ncct scan chest","usg knee","usg","cect head","ct pns axial,coronal & sag","ct whole abdomen","ct thorax/chest","mri brain plain","x-ray abdomen",'ncct kub','nct head','nephrogram','voluetric scanning chest','mr angio brain','cect scan thorax','usg chest','usg guided','ultrasoung guided pigtail','cect thorax & abdomen','ct scan pulmonary angio','usg pelvis','trus','x-ray ankle','colour doppler','mri report lower adomen','ultrasound doppler left upper limb arterial','arterial colour doppler','venous doppler','mri report','ct head/brain','ct thorax/chest','mri brain screening','usg breasts','mammography','x-ray left foot joint','x-ray pelvis hips','x-ray right foot joint','x-ray knee','usg','x-ray','ct kub','mri lumbo-sacral spine','mri screening','mr cholangioagraphy','x-ray lumbar spine','x-ray lumar spine','usg screening','cect pns','ct aortogram','kub','chest paap view']
    badie=['report','reports','result','results','advise','investigation','investigations','study','please correlate clinically',' please correlate clinically.']
    moth=['aug','jun''july']
    for i,j in s.iterrows():
        text=s.Result[i]
        for y in okbad:
            if(y in text):
                text=text.replace(y,'')
        for y in badie:
            if(y in text):
                text=text.replace(y,'')
        for y in moth:
            if(y in text):
                text=text.replace(y,'')
        df.Result[i]=text
       
    del s['Result']
    s=pd.concat([s,df],axis=1)
    test_related_pe=['CT Thorax / Chest - HRCT Contrast','CT Thorax / Chest - HRCT Plain','CT Thorax / Chest - Plain','CT Whole Abdomen - Plain','CT Whole Abdomen with Contrast','CT Whole Thorax (Contrast)','CT Whole Thorax (Plain)','MRI Cholangiography (MRCP)','Ultrasound Chest','Ultrasound KUB','Ultrasound Whole Abdomen','USG Chest','USG Upper Abdomen','USG Whole Abdomen','X-ray (any Portable single exposure)','X-Ray Chest PA/AP View']
    twword=[('pleural','fluid'),('pleural','cavity'),('pleural', 'effusion'),('right', 'lobe'),('lower', 'lobe'),('middle', 'lobe'),('upper', 'lobe'),('left', 'lobe'),('right', 'side'),('left', 'side'),('bilateral','cp'),('cp', 'angle'),('free', 'fluid'),('subpulmonic','effusion')]
    thrword=[('',''),('left','pleural_fluid'),('right','pleural_fluid'),('left','pleural_cavity'),('right','pleural_cavity'),('right','cp_angle'),('left','cp_angle'),('left', 'upper_lobe'),('right', 'upper_lobe'),('right','pleural_effusion'),('bilateral', 'pleural_effusion'),('left', 'pleural_effusion')]
    fourword=[('cp_angle','free_fluid'),('right','sided')]
    klo=[('right','sided'),('left','sided'),('right', 'side'),('left', 'side'),('moderate','sized'),('there','is'),]
    to2=[]
    ps = PorterStemmer() 
    lemmatizer = WordNetLemmatizer() 
    for i,j in df.iterrows():
        text=df.Result[i]
        to=nltk.tokenize.word_tokenize(text)
        for x in range (len(to)):
            y=to[x]
            to[x]=lemmatizer.lemmatize(y)
        txt=' '.join(to)
        pattern = '[0-9]'
        txt2 = [re.sub(pattern, '', i) for i in to]
        txt3=' '.join(txt2)
        df.Result[i]=txt3
    #print(df.Result[1696])
    for i,j in s.iterrows():
        text=df.Result[i]
        to=nltk.tokenize.word_tokenize(text)
        oof=[]
        for x in range(len(to)-1):
            a=(to[x],to[x+1])
            if(a in twword):
                oof.append(x)
        for x in oof:
            a=to[x]+'_'+to[x+1]
            to[x]=a
            to[x+1]=''
    
        oof2=[]
        for x in range(len(to)-1):
            a=(to[x],to[x+1])
            if(a in thrword):
                #print(a)
                oof2.append(x)
        for x in oof2:
            a=to[x]+'_'+to[x+1]
            to[x]=a
            to[x+1]=''
        oof3=[]
        for x in range(len(to)-1):
            a=(to[x],to[x+1])
            if(a in fourword):
                oof3.append(x)
        for x in oof3:
            a=to[x]+'_'+to[x+1]
            to[x]=a
            to[x+1]=''
        oof3=[]
        for x in range(len(to)-1):
            a=(to[x],to[x+1])
            if(a in klo):
                oof3.append(x)
        for x in oof3:
            a=to[x]+'_'+to[x+1]
            to[x]=a
            to[x+1]=''
        oof3=[]
        for x in range(len(to)-1):
            a=(to[x],to[x+1])
            if(a in oofsdf):
                oof3.append(x)
        for x in oof3:
            a=to[x]+'_'+to[x+1]
            to[x]=a
            to[x+1]=''
    
        for x in to:
            if(x==''):
                to.remove(x)
        txt=' '.join(to)
        df.Result[i]=txt
    
    ps = PorterStemmer() 
    lemmatizer = WordNetLemmatizer() 
    for i,j in df.iterrows():
        text=df.Result[i]
        to=nltk.tokenize.word_tokenize(text)
        for x in range (len(to)):
            y=to[x]
            to[x]=lemmatizer.lemmatize(y)
        txt=' '.join(to)
        pattern = '[0-9]'
        txt2 = [re.sub(pattern, '', i) for i in to]
        txt3=' '.join(txt2)
        df.Result[i]=txt3
    del s['Result']
    s=pd.concat([s,df],axis=1)
    import re
    sopd=s.copy()
    unique_words=set()
    
    test_related_pe=['CT Thorax / Chest - HRCT Contrast','CT Thorax / Chest - HRCT Plain','CT Thorax / Chest - Plain','CT Whole Abdomen - Plain','CT Whole Abdomen with Contrast','CT Whole Thorax (Contrast)','CT Whole Thorax (Plain)','MRI Cholangiography (MRCP)','Ultrasound Chest','Ultrasound KUB','Ultrasound Whole Abdomen','USG Chest','USG Upper Abdomen','USG Whole Abdomen','X-ray (any Portable single exposure)','X-Ray Chest PA/AP View']
    twword=[('pleural','cavity'),('pleural', 'effusion'),('bilateral','cp'),('cp', 'angle'),('free', 'fluid')]
    thrword=[('left','pleural_cavity'),('right','pleural_cavity'),('right','cp_angle'),('left','cp_angle'),('right','pleural_effusion'),('bilateral', 'pleural_effusion'),('left', 'pleural_effusion')]
    fourword=[('cp_angle','free_fluid')]
    safe=['no','clear','no_evidence','no_significant','unremarkable']
    ad2j=['right_sided','left_sided','right_side','left_side','bilateral']
    
    commonevidencesuggestive=['haze','there_is','present','minimal','present','seen','hazy','due to','s/o','mild','evidence','moderate_sized','noted','?','??','suggest']
    wordsrelatedtope=['left_pleural_effusion,left_pleural_fluid','right_pleural_effusion','right_pleural_fluid','pleural_effusion','bilateral_cp','free_fluid','cp_angle_free_fluid','bilateral_pleural_effusion','left_pleural_effusion','right_pleural_effusion','bilateral_pleural_effusion','cp_angle','left_cp_angle','right_cp_angle','pleural_cavity','left_pleural_cavity','right_pleural_cavity','subpulmonic_effusion','effusion']
    
    okdokie=[]
    for x in twword:
        a=x[0]+'_'+x[1]
        okdokie.append(a)
    for x in thrword:
        a=x[0]+'_'+x[1]
        okdokie.append(a)
    for x in fourword:
        a=x[0]+'_'+x[1]
        okdokie.append(a)   
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    m=0
    oop=[]
    for i,j in s.iterrows():
        oledole=[0 for i in range(50)]
        text=s.Result[i]
        to=tokenizer.tokenize(text)
        incre=[]
        for x in ((safe)):
            if(x in text):
                incre.append(x)
        for x in ((ad2j)):
            if(x in text):
                incre.append(x)
        for x in ((commonevidencesuggestive)):
            if(x in text):
                incre.append(x)
        for x in ((wordsrelatedtope)):
            if(x in text):
                incre.append(x)
        for x in ((okdokie)):
            if(x in text):
                incre.append(x)
        m=max(m,len(incre))
        oop.append(incre)
    olelelo=[]
    for x in safe:
        olelelo.append(x)
    for x in ad2j:
        olelelo.append(x)
    for x in commonevidencesuggestive:
        olelelo.append(x)
    for x in wordsrelatedtope:
        olelelo.append(x)
    for x in okdokie:
        olelelo.append(x)
    
    
    new_df = pd.DataFrame(data=oop)
    sop=s.copy()
    n=0
    u=[]
    for i,j in s.iterrows():
        text=s.Result[i]
        to=tokenizer.tokenize(text)
        if(s.Pleural_effusion[i]==1 or s.Pleural_effusion[i]==0):
            for z in range(len(to)): 
                replace=0
                toy=to[z]
                oofdiedoofdie=nltk.tokenize.word_tokenize(toy)
                for x in range(len(oofdiedoofdie)):
                    os=oofdiedoofdie[x]
                    for y in okdokie:
                        if(os==y):
                            replace+=1
                            oofdiedoofdie[x]='FTR'
                    for y in safe:
                        if(os==y):
                            replace+=1
                            oofdiedoofdie[x]='SFE'
                    for y in ad2j:
                        #print(y)
                        if(os==y):
                            replace+=1
                            #print(y)
                            oofdiedoofdie[x]='ADJ'
                    for y in commonevidencesuggestive:
                        if(os==y):
                            replace+=1
                            oofdiedoofdie[x]='RISK'
                    for y in wordsrelatedtope:
                        if(os==y):
                            replace+=1
                            oofdiedoofdie[x]='FTR'
                        #if(x>=1 ):
                        #    if(x<len(oofdiedoofdie)-1):
                        #        if(os==y and oofdiedoofdie[x-1]!='RISK' and oofdiedoofdie[x+1]!='RISK' and oofdiedoofdie[x-1]!='SFE' and oofdiedoofdie[x+1]!='SFE'):
                        #            oofdiedoofdie[x]='RISK'
                        #    else:
                        #        if(os==y and oofdiedoofdie[x-1]!='RISK' and oofdiedoofdie[x-1]!='SFE'):
                        #            oofdiedoofdie[x]='RISK'
                        #else:
                        #    if(os==y and oofdiedoofdie[x+1]!='RISK' and oofdiedoofdie[x+1]!='SFE'):
                        #        oofdiedoofdie[x]='RISK'
                    
                            
                to[z]=' '.join(oofdiedoofdie)
        text=' '.join(to)
        df.Result[i]=text
    del s['Result']
    s=pd.concat([s,df],axis=1)
    import re
    unique_words=set()
    punctuations = list(string.punctuation)
    for i,j in df.iterrows():
        text=df.Result[i]
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*~+/='''
        text3=""
        for char in text:
            if char not in punctuations:
                text3 = text3 + char
        unique_words_temp=set(text3.split())
        unique_words=unique_words.union(unique_words_temp)
        df.Result[i]=text3.lstrip()
        
    del s['Result']
    s=pd.concat([s,df],axis=1)
    return s

s=prerprocess('./train_pe.csv')

#initial testing
new_stopwords = set(stopwords.words('english')).difference(to_remove)
x_t, x_2, y_train, y_test = train_test_split(s.Result, s.Pleural_effusion,test_size=0.2,random_state=1)
vectorizer = CountVectorizer(lowercase=True, stop_words=new_stopwords)
x_train = vectorizer.fit_transform(raw_documents=x_t)
x_test = vectorizer.transform(raw_documents=x_2)
print("-----Next Logistic-----")
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(x_train,y_train)
ospdo=lor.predict(x_test)
print ("Accuracy :",accuracy_score(y_test,ospdo))
print ("F1;:",f1_score(y_test,ospdo))
print("Precision:",precision_score(y_test,ospdo))
print("Recall:",recall_score(y_test,ospdo))
cm = confusion_matrix(y_test,ospdo)
print(cm)
tn, fp, fn, tp = cm.ravel()
specificity=tn / (tn+fp)
sensitivity=tp / (tp+fn)
print("Specificity:",specificity)
print("Sensitivity:",sensitivity)

#final testing
path2='./train_pe.csv'
#please_input_the_path_of_the_csv_file
training_stuff=prerprocess(path2)
new_stopwords = set(stopwords.words('english')).difference(to_remove)
#x_t, x_2, y_train, y_test = train_test_split(s.Result, s.Pleural_effusion,test_size=0.2,random_state=1)
vectorizer = CountVectorizer(lowercase=True, stop_words=new_stopwords)
x_train = vectorizer.fit_transform(raw_documents=s.Result)
x_test = vectorizer.transform(raw_documents=training_stuff.Result)
y_train=s.Pleural_effusion
y_test=training_stuff.Pleural_effusion
print("-----Next Logistic-----")
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(x_train,y_train)
ospdo=lor.predict(x_test)
print ("Accuracy :",accuracy_score(y_test,ospdo))
print ("F1;:",f1_score(y_test,ospdo))
print("Precision:",precision_score(y_test,ospdo))
print("Recall:",recall_score(y_test,ospdo))
cm = confusion_matrix(y_test,ospdo)
print(cm)
tn, fp, fn, tp = cm.ravel()
specificity=tn / (tn+fp)
sensitivity=tp / (tp+fn)
print("Specificity:",specificity)
print("Sensitivity:",sensitivity)


