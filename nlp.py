import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

text=["Intro\nFade the [music] out.[1] Let's roll. Hol2d there. Lights. Do the lights. Thank you. Thank you very much. I appreciate that. I don't necessarily agree with you, but I appreciate very much. Well, this is a nice place.","This is Dave. He tells dirty jokes for a living. That stare is where most of his hard work happens."]
df=pd.DataFrame(text,columns=['Transcript'])

def clean_data(text):
    text=re.sub('\[.*?\]','',text)
    text=re.sub('\n','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation) ,'',text)
    text=re.sub('[''""]','',text)
    text=re.sub('\w*\d\w*','',text)
    return text

df=pd.DataFrame(df.Transcript.apply(clean_data))
cv=CountVectorizer(stop_words='english')
q=cv.fit_transform(df.Transcript)
df=pd.DataFrame(q.toarray(),columns=cv.get_feature_names_out())
print(df)
# print(clean_data(text))
