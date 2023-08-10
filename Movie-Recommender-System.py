#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import ast


# In[50]:


#importing data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[51]:


movies.head()


# In[52]:


credits.head(1)


# In[53]:


#meagre datasets(movies and credits)
movies = movies.merge(credits,on='title')


# In[54]:


movies.shape


# In[55]:


credits.shape


# In[56]:


movies.head()


# In[57]:


#Preprocessing Dataset
#remove tags which are not very useful
#make tags for every movie=overview+genres+keywords+cast+crew+release_date

#genres
#id
#keywords
#title
#overview
#release_date
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew','release_date']]


# In[58]:


#movies['original_language'].value_counts()


# In[59]:


movies.info()


# In[60]:


movies.head()


# In[61]:


movies.isnull().sum()


# In[62]:


movies.dropna(inplace=True)


# In[63]:


movies.duplicated().sum()


# In[64]:


#use iloc for row extraction
movies.iloc[0].genres


# In[65]:


#use ast.literal_evalmodule for get a list(Action,Adventure,Fantasy,Sci-fi)
def convert(obj):
   L = []
   for i in ast.literal_eval(obj):
       L.append(i['name'])
   return L


# In[66]:


movies['genres'] = movies['genres'].apply(convert)


# In[67]:


movies.head()


# In[68]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[69]:


movies.head()


# In[70]:


#for take top three cast
def convert3(obj):
       L = []
       counter = 0
       for i in ast.literal_eval(obj):
           if counter != 3:
               L.append(i['name'])
               counter+=1
           else:
               break
       return L


# In[71]:


movies['cast'] = movies['cast'].apply(convert3)


# In[72]:


movies.head()


# In[73]:


movies['crew'][0]


# In[74]:


#For get name of director of each movie
def fetch_director(obj):
       L = []
       for i in ast.literal_eval(obj):
           if i['job'] == 'Director' :
               L.append(i['name'])
               break
       return L


# In[75]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[76]:


movies.head()


# In[77]:


movies['overview'][0]


# In[78]:


#For converting string into a list
movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[79]:


movies.head()


# In[80]:


#Use List Comprehension for removing space between two words
movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ","")for i in x])


# In[81]:


movies.head()


# In[82]:


movies['release_date']=movies['release_date'].apply(lambda x:x.split())


# In[83]:


movies.head()


# In[84]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']+movies['release_date']


# In[85]:


movies.head()


# In[86]:


new_df = movies[['movie_id','title','tags']]


# In[87]:


new_df


# In[88]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[89]:


new_df.head()


# In[90]:


new_df['tags'][0]


# In[91]:


#for converting into lower case
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[100]:


new_df.head()


# In[101]:


#apply text vectorizer (bag of words) for converting text into vector and removing stop words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[102]:


#convert into an array
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[103]:


#sparse matrix
vectors


# In[104]:


(cv.get_feature_names())


# In[105]:


#natural language processing library
get_ipython().system('pip install nltk')


# In[106]:


#we use stem for removing similar word(uses,use,using)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[107]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[108]:


new_df['tags']=new_df['tags'].apply(stem)
stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron 2009-12-10')


# In[109]:


#use cosine_distance for find dist between two vectors
from sklearn.metrics.pairwise import cosine_similarity


# In[110]:


cosine_similarity(vectors).shape


# In[117]:


similarity = cosine_similarity(vectors)


# In[118]:


similarity[0]


# In[119]:


list(enumerate(similarity[0]))


# In[120]:


#use enumerate function for retain index value
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# In[121]:


#use enumerate function for retain index value
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[97]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[99]:


(recommend('Batman')


# In[124]:


new_df.iloc[1196].title


# In[125]:


#convert into a website


# In[126]:


import pickle


# In[127]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[128]:


new_df['title'].values


# In[129]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[130]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




