```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
# 문장 토큰화(sent tokenize)
from nltk import sent_tokenize
text_sample = 'The Matrix is everywhere its all around us, here even in this room. \
               You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes.'
sentences = sent_tokenize(text=text_sample)
print(sentences)
print(type(sentences),len(sentences))
```

    ['The Matrix is everywhere its all around us, here even in this room.', 'You can see it out your window or on your television.', 'You feel it when you go to work, or go to church or pay your taxes.']
    <class 'list'> 3
    


```python
# 단어 토큰화(word_tokenize) 
from nltk import word_tokenize
sentence = 'The Matrix is everywhere its all around us, here even in this room.'
words = word_tokenize(sentence)
print(words)
print(type(words), len(words))
```

    ['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.']
    <class 'list'> 15
    


```python
from nltk import word_tokenize, sent_tokenize

def tokenize_text(text):
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

word_tokens = tokenize_text(text_sample)
print(word_tokens)
print(type(word_tokens), len(word_tokens))
```

    [['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 'window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]
    <class 'list'> 3
    


```python
# 스톱 워드 제거 : is, the, a, will
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
# NLTK의 english stopwords 갯수 확인
print('영어 stop words 갯수:', len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:20])
```

    영어 stop words 갯수: 179
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']
    


```python
# stopwords 필터링을 통한 제거
import nltk
stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens:
    filtered_words=[]
    for word in sentence:
        word=word.lower()
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)
print(all_tokens)
```

    [['matrix', 'everywhere', 'around', 'us', ',', 'even', 'room', '.'], ['see', 'window', 'television', '.'], ['feel', 'go', 'work', ',', 'go', 'church', 'pay', 'taxes', '.']]
    


```python
# 문법적 또는 의미적으로 변화하는 단어의 원형을 찾는 방법
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused'))
```

    work work work
    amus amus amus
    


```python
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
# 정확한 원형 단어 추출을 위해 단어의 품사를 입력
from nltk.stem.wordnet import WordNetLemmatizer

lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing','v'), lemma.lemmatize('amuses','v'),\
      lemma.lemmatize('amused','v'))
```

    amuse amuse amuse
    


```python
# KoNLPy 설치
# Java 시스템 환경변수 편집 path 설정(C:\Program Files\Java\jdk1.8.0_181\bin)
# JPype1 다운로드 받고 설치(pip install jpype1)
# pip install konlpy
```


```python
# 형태소 분석으로 문장을 단어로 분할
from konlpy.tag import Okt
okt = Okt()
print(okt.pos('아름다운 꽃과 파란 하늘'))
```

    [('아름다운', 'Adjective'), ('꽃', 'Noun'), ('과', 'Josa'), ('파란', 'Noun'), ('하늘', 'Noun')]
    


```python
# 형용사인 품사만 선별해 리스트에 담기
sentences = ['아름다운 꽃과 파란 하늘']
from konlpy.tag import Okt
okt=Okt()
for sentence in sentences:
    morph = okt.pos(sentence)
    print(morph)
print()    
adj_list = []
for word, tag in morph:
    if tag =='Adjective':
        adj_list.append(word)
print(adj_list)
```

    [('아름다운', 'Adjective'), ('꽃', 'Noun'), ('과', 'Josa'), ('파란', 'Noun'), ('하늘', 'Noun')]
    
    ['아름다운']
    


```python
from konlpy.tag import Okt
okt = Okt()
malist1 = okt.nouns('나는 오늘 방콕에 가고싶다')
print('명사')
print(malist1)
malist2 = okt.pos('나는 오늘 방콕에 갔다',norm=True, stem=True)
print('원형')
print(malist2)
malist3 = okt.morphs('친절한 코치와 재미있는 친구들이 있는 도장에 가고 싶다.')
print('형태소')
print(malist3)
malist4 = okt.pos('나는 오늘도 장에 가고 싶다', norm=True, stem=True, join=True)
print('형태소/태그')
print(malist4)
malist5 = okt.pos('나는 오늘 장에 가고싶을깤ㅋㅋ?', norm=True, stem=True)
print('정규화,원형')
print(malist5)
```

    명사
    ['나', '오늘', '방콕']
    원형
    [('나', 'Noun'), ('는', 'Josa'), ('오늘', 'Noun'), ('방콕', 'Noun'), ('에', 'Josa'), ('가다', 'Verb')]
    형태소
    ['친절한', '코치', '와', '재미있는', '친구', '들', '이', '있는', '도장', '에', '가고', '싶다', '.']
    형태소/태그
    ['나/Noun', '는/Josa', '오늘/Noun', '도/Josa', '장/Noun', '에/Josa', '가다/Verb', '싶다/Verb']
    정규화,원형
    [('나', 'Noun'), ('는', 'Josa'), ('오늘', 'Noun'), ('장', 'Noun'), ('에', 'Josa'), ('가다', 'Verb'), ('ㅋㅋ', 'KoreanParticle'), ('?', 'Punctuation')]
    


```python
# 데이터 가져오기
from sklearn.datasets import fetch_20newsgroups
news_data = fetch_20newsgroups(subset='all', random_state=0)
news_data.keys()
```




    dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])




```python
news_data.data[5]
```




    'From: ykim@cs.columbia.edu (Yong Su Kim)\nSubject: Fast wireframe graphics\nDistribution: usa\nOrganization: Columbia University Department of Computer Science\nLines: 29\n\n\nI am working on a program to display 3d wireframe models with the user\nbeing able to arbitrarily change any of the viewing parameters.  Also,\nthe wireframe objects are also going to have dynamic attributes so\nthat they can move around while the user is "exploring" the wireframe\nworld.\n\nTo do this, I am thinking of using the SRGP package described in the\nVan Dam, Foley and Feiner book, but I was wondering if there was\nanother PD graphics package out there which was faster.  I would like\nto make the program as fast as possible so that it provides\nsatisfactory real time performance on a Sun IPX.\n\nIdeally, I\'m looking for a PD graphics package which will allow me to\nopen a new window under X, and allow me to draw lines within the\nwindow.  Also, it would also need to have some sort of event driven\ninteraction handling since the user is going to move around the\nwireframe models using the keyboard.\n\nIf you know or wrote such a package, I would be grateful if you could\ndirect me to a ftp site which contains the package.\n\nThank you.\n\n-- \n===============================================================================\nYong Su Kim, Class of 1993\t\t|  Internet: yk4@cunixb.cc.columbia.edu\nColumbia College, Columbia University   |        or  ykim@cs.columbia.edu\n\t\t\t\n'




```python
news_data.DESCR
```




    '.. _20newsgroups_dataset:\n\nThe 20 newsgroups text dataset\n------------------------------\n\nThe 20 newsgroups dataset comprises around 18000 newsgroups posts on\n20 topics split in two subsets: one for training (or development)\nand the other one for testing (or for performance evaluation). The split\nbetween the train and test set is based upon a messages posted before\nand after a specific date.\n\nThis module contains two loaders. The first one,\n:func:`sklearn.datasets.fetch_20newsgroups`,\nreturns a list of the raw texts that can be fed to text feature\nextractors such as :class:`sklearn.feature_extraction.text.CountVectorizer`\nwith custom parameters so as to extract feature vectors.\nThe second one, :func:`sklearn.datasets.fetch_20newsgroups_vectorized`,\nreturns ready-to-use features, i.e., it is not necessary to use a feature\nextractor.\n\n**Data Set Characteristics:**\n\n    =================   ==========\n    Classes                     20\n    Samples total            18846\n    Dimensionality               1\n    Features                  text\n    =================   ==========\n\nUsage\n~~~~~\n\nThe :func:`sklearn.datasets.fetch_20newsgroups` function is a data\nfetching / caching functions that downloads the data archive from\nthe original `20 newsgroups website`_, extracts the archive contents\nin the ``~/scikit_learn_data/20news_home`` folder and calls the\n:func:`sklearn.datasets.load_files` on either the training or\ntesting set folder, or both of them::\n\n  >>> from sklearn.datasets import fetch_20newsgroups\n  >>> newsgroups_train = fetch_20newsgroups(subset=\'train\')\n\n  >>> from pprint import pprint\n  >>> pprint(list(newsgroups_train.target_names))\n  [\'alt.atheism\',\n   \'comp.graphics\',\n   \'comp.os.ms-windows.misc\',\n   \'comp.sys.ibm.pc.hardware\',\n   \'comp.sys.mac.hardware\',\n   \'comp.windows.x\',\n   \'misc.forsale\',\n   \'rec.autos\',\n   \'rec.motorcycles\',\n   \'rec.sport.baseball\',\n   \'rec.sport.hockey\',\n   \'sci.crypt\',\n   \'sci.electronics\',\n   \'sci.med\',\n   \'sci.space\',\n   \'soc.religion.christian\',\n   \'talk.politics.guns\',\n   \'talk.politics.mideast\',\n   \'talk.politics.misc\',\n   \'talk.religion.misc\']\n\nThe real data lies in the ``filenames`` and ``target`` attributes. The target\nattribute is the integer index of the category::\n\n  >>> newsgroups_train.filenames.shape\n  (11314,)\n  >>> newsgroups_train.target.shape\n  (11314,)\n  >>> newsgroups_train.target[:10]\n  array([ 7,  4,  4,  1, 14, 16, 13,  3,  2,  4])\n\nIt is possible to load only a sub-selection of the categories by passing the\nlist of the categories to load to the\n:func:`sklearn.datasets.fetch_20newsgroups` function::\n\n  >>> cats = [\'alt.atheism\', \'sci.space\']\n  >>> newsgroups_train = fetch_20newsgroups(subset=\'train\', categories=cats)\n\n  >>> list(newsgroups_train.target_names)\n  [\'alt.atheism\', \'sci.space\']\n  >>> newsgroups_train.filenames.shape\n  (1073,)\n  >>> newsgroups_train.target.shape\n  (1073,)\n  >>> newsgroups_train.target[:10]\n  array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0])\n\nConverting text to vectors\n~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nIn order to feed predictive or clustering models with the text data,\none first need to turn the text into vectors of numerical values suitable\nfor statistical analysis. This can be achieved with the utilities of the\n``sklearn.feature_extraction.text`` as demonstrated in the following\nexample that extract `TF-IDF`_ vectors of unigram tokens\nfrom a subset of 20news::\n\n  >>> from sklearn.feature_extraction.text import TfidfVectorizer\n  >>> categories = [\'alt.atheism\', \'talk.religion.misc\',\n  ...               \'comp.graphics\', \'sci.space\']\n  >>> newsgroups_train = fetch_20newsgroups(subset=\'train\',\n  ...                                       categories=categories)\n  >>> vectorizer = TfidfVectorizer()\n  >>> vectors = vectorizer.fit_transform(newsgroups_train.data)\n  >>> vectors.shape\n  (2034, 34118)\n\nThe extracted TF-IDF vectors are very sparse, with an average of 159 non-zero\ncomponents by sample in a more than 30000-dimensional space\n(less than .5% non-zero features)::\n\n  >>> vectors.nnz / float(vectors.shape[0])\n  159.01327...\n\n:func:`sklearn.datasets.fetch_20newsgroups_vectorized` is a function which \nreturns ready-to-use token counts features instead of file names.\n\n.. _`20 newsgroups website`: http://people.csail.mit.edu/jrennie/20Newsgroups/\n.. _`TF-IDF`: https://en.wikipedia.org/wiki/Tf-idf\n\n\nFiltering text for more realistic training\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nIt is easy for a classifier to overfit on particular things that appear in the\n20 Newsgroups data, such as newsgroup headers. Many classifiers achieve very\nhigh F-scores, but their results would not generalize to other documents that\naren\'t from this window of time.\n\nFor example, let\'s look at the results of a multinomial Naive Bayes classifier,\nwhich is fast to train and achieves a decent F-score::\n\n  >>> from sklearn.naive_bayes import MultinomialNB\n  >>> from sklearn import metrics\n  >>> newsgroups_test = fetch_20newsgroups(subset=\'test\',\n  ...                                      categories=categories)\n  >>> vectors_test = vectorizer.transform(newsgroups_test.data)\n  >>> clf = MultinomialNB(alpha=.01)\n  >>> clf.fit(vectors, newsgroups_train.target)\n  MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)\n\n  >>> pred = clf.predict(vectors_test)\n  >>> metrics.f1_score(newsgroups_test.target, pred, average=\'macro\')\n  0.88213...\n\n(The example :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py` shuffles\nthe training and test data, instead of segmenting by time, and in that case\nmultinomial Naive Bayes gets a much higher F-score of 0.88. Are you suspicious\nyet of what\'s going on inside this classifier?)\n\nLet\'s take a look at what the most informative features are:\n\n  >>> import numpy as np\n  >>> def show_top10(classifier, vectorizer, categories):\n  ...     feature_names = np.asarray(vectorizer.get_feature_names())\n  ...     for i, category in enumerate(categories):\n  ...         top10 = np.argsort(classifier.coef_[i])[-10:]\n  ...         print("%s: %s" % (category, " ".join(feature_names[top10])))\n  ...\n  >>> show_top10(clf, vectorizer, newsgroups_train.target_names)\n  alt.atheism: edu it and in you that is of to the\n  comp.graphics: edu in graphics it is for and of to the\n  sci.space: edu it that is in and space to of the\n  talk.religion.misc: not it you in is that and to of the\n\n\nYou can now see many things that these features have overfit to:\n\n- Almost every group is distinguished by whether headers such as\n  ``NNTP-Posting-Host:`` and ``Distribution:`` appear more or less often.\n- Another significant feature involves whether the sender is affiliated with\n  a university, as indicated either by their headers or their signature.\n- The word "article" is a significant feature, based on how often people quote\n  previous posts like this: "In article [article ID], [name] <[e-mail address]>\n  wrote:"\n- Other features match the names and e-mail addresses of particular people who\n  were posting at the time.\n\nWith such an abundance of clues that distinguish newsgroups, the classifiers\nbarely have to identify topics from text at all, and they all perform at the\nsame high level.\n\nFor this reason, the functions that load 20 Newsgroups data provide a\nparameter called **remove**, telling it what kinds of information to strip out\nof each file. **remove** should be a tuple containing any subset of\n``(\'headers\', \'footers\', \'quotes\')``, telling it to remove headers, signature\nblocks, and quotation blocks respectively.\n\n  >>> newsgroups_test = fetch_20newsgroups(subset=\'test\',\n  ...                                      remove=(\'headers\', \'footers\', \'quotes\'),\n  ...                                      categories=categories)\n  >>> vectors_test = vectorizer.transform(newsgroups_test.data)\n  >>> pred = clf.predict(vectors_test)\n  >>> metrics.f1_score(pred, newsgroups_test.target, average=\'macro\')\n  0.77310...\n\nThis classifier lost over a lot of its F-score, just because we removed\nmetadata that has little to do with topic classification.\nIt loses even more if we also strip this metadata from the training data:\n\n  >>> newsgroups_train = fetch_20newsgroups(subset=\'train\',\n  ...                                       remove=(\'headers\', \'footers\', \'quotes\'),\n  ...                                       categories=categories)\n  >>> vectors = vectorizer.fit_transform(newsgroups_train.data)\n  >>> clf = MultinomialNB(alpha=.01)\n  >>> clf.fit(vectors, newsgroups_train.target)\n  MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)\n\n  >>> vectors_test = vectorizer.transform(newsgroups_test.data)\n  >>> pred = clf.predict(vectors_test)\n  >>> metrics.f1_score(newsgroups_test.target, pred, average=\'macro\')\n  0.76995...\n\nSome other classifiers cope better with this harder version of the task. Try\nrunning :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py` with and without\nthe ``--filter`` option to compare the results.\n\n.. topic:: Recommendation\n\n  When evaluating text classifiers on the 20 Newsgroups data, you\n  should strip newsgroup-related metadata. In scikit-learn, you can do this by\n  setting ``remove=(\'headers\', \'footers\', \'quotes\')``. The F-score will be\n  lower because it is more realistic.\n\n.. topic:: Examples\n\n   * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`\n\n   * :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`\n'




```python
# 텍스트 정규화
# 뉴스그룹 기사내용을 제외한 다른 정보 제거
# 제목, 소속, 이메일 등 헤더와 푸터 정보들은 분류의 타겟 클래스 값과 유사할 수 있음

train_news = fetch_20newsgroups(subset='train', \
                                remove=('header','footer','quotes'), random_state=0)
X_train = train_news.data
y_train = train_news.target
test_news = fetch_20newsgroups(subset='test',\
                              remove=('header','footer','quotes'),random_state=0)
X_test = test_news.data
y_test = test_news.target
print(len(X_train),len(X_test))
```

    11314 7532
    


```python
# 피처 벡터화 변환
from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
cnt_vect.fit(X_train)
X_train_cnt_vect = cnt_vect.transform(X_train)
# 학습 데이터로 fit()된 CounterVectorizer를 이용, 테스트 데이터 변환시 적용
# 피처 개수가 동일해야 함
X_test_cnt_vect = cnt_vect.transform(X_test)
```


```python
# 머신러닝 모델 학습/예측/평가
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

lr_clf = LogisticRegression()
lr_clf.fit(X_train_cnt_vect,y_train)
lr_pred = lr_clf.predict(X_test_cnt_vect)
print(accuracy_score(y_test, lr_pred))
```

    0.7523898035050451
    
