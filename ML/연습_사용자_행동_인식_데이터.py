#!/usr/bin/env python
# coding: utf-8

# In[1]:


# features.txt : 피처 인덱스와 피처명을 가지고 있으므로 불러와 확인
# 피처 이름 index와 피처명이 공백으로 분리되어 있음

import pandas as pd
feature_name_df = pd.read_csv('./data/human_activity/features.txt', sep = '\s+', # \s : 공백문자, + : 해당 패턴이 하나 이상의 연속된 문자와 매칭
                             header = None, names = ['column_index', 'column_name']) # header는 없이, col명은 column_index, column_name으로

feature_name_df.head()

# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:, 1].values.tolist() # column_name만 불러와서 리스트로 변경(tolist())
print('전체 피처명에서 10개만 추출 : ', feature_name[:10])


# - 인체의 움직임과 관련된 속성의 평균/표준편차가 X, Y, Z축 값으로 돼 있다
# 
# - 중복된 피처가 있어 원본 피처명에 _1 또는 _2를 추가로 부여하기

# In[2]:


# 중독된 피처명 확인

feature_dup_df = feature_name_df.groupby('column_name').count() # column_name으로 그룹지어 확인
feature_dup_df.head() 

# column_index가 1 이상인 것 가져오기
print(feature_dup_df[feature_dup_df['column_index'] > 1].count()) # 1이상인 것의 갯수
feature_dup_df[feature_dup_df['column_index'] > 1].head()


# In[3]:


# 중복된 피처명에 대해 원본 피처명에 _1 또는 _2를 추가해 새로운 피처명을 갖도록 get_new_feature_name_df()를 생성

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data = old_feature_name_df.groupby('column_name').cumcount(),# column_name으로 그룹한 후 그룹별로 각 행에 대해 누적된 카운트를 반환
                                  columns = ['dup_cnt']) # 누적 카운트의 col명은 dup_cnt
    feature_dup_df = feature_dup_df.reset_index() # 인덱스 지정
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how = 'outer') # old_feature_name_df와 feature_dup_df를 외부 조인
    
    # 중복된 열 이름에 대해 '_숫자'를 추가하여 새로운 열 이름을 생성
    # x[0] : column_name, x[1] : dup_cnt
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0] + '_'+str(x[1]) if x[1] > 0 else x[0], axis = 1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis = 1) # index col을 제거
    return new_feature_name_df


# 학습용 피처 데이터 세트와 레이블 데이터 세트, 테스트용 피처 데이터 파일과 레이블 데이터 파일을 각 학습/테스트용 dataframe에 로드

# In[4]:


# get_new_feature_name_df()는 get_human_dataset()내에서 적용돼 중복 된 피처명을 새로운 피처명으로 할당

def get_human_dataset():
    
    # 각 데이터 파일은 공백으로 분리되어있다
    feature_name_df = pd.read_csv('./data/human_activity/features.txt', sep = '\s+', # \s : 공백문자, + : 해당 패턴이 하나 이상의 연속된 문자와 매칭
                             header = None, names = ['column_index', 'column_name']) # header는 없이, col명은 column_index, column_name으로
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용해 신규 피처명을 생성
    new_feature_df = get_new_feature_name_df(feature_name_df)
    
    # dataframe에 피처명을 컬럼으로 부여하기 위해 리스트 객체 반환
    feature_name = new_feature_df.iloc[:, 1].values.tolist() # column_name만 불러와서 리스트로 변경(tolist())
    
    # 학습 피처 데이터세트와 테스트 피처 데이터를 DataFrame로 로딩, col은 feature_name으로
    X_train = pd.read_csv('./data/human_activity/train/X_train.txt', sep = '\s+', names = feature_name)
    X_test = pd.read_csv('./data/human_activity/test/X_test.txt', sep = '\s+', names = feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터를 DataFrame로 로딩, col은 action으로 
    y_train = pd.read_csv('./data/human_activity/train/y_train.txt', sep = '\s+', header = None, names = ['action'])
    y_test = pd.read_csv('./data/human_activity/test/y_test.txt', sep = '\s+', header = None, names = ['action'])    
    
    # 불러온 학습/테스트용 DataFrame을 모두 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()


# In[5]:


print('학습 피처 데이터셋 info()')
print(X_train.info())


# - 학습 데이터 세트는 7352개의 레코드로 561개의 피처를 갖고 있다

# In[6]:


y_train['action'].value_counts()


# - 값이 왜곡되지 않고 비교적 고르게 분포
# 
# - DecisionTreeClassifier를 이용해 동작 예측 분류 수행

# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 예제 반복 시마다 동일한 예측 결과 도축을 위해 random_state 설정
dt = DecisionTreeClassifier(random_state = 156) # 모델 생성
dt.fit(X_train, y_train) # 학습
pred = dt.predict(X_test) # 예측
accuracy = accuracy_score(y_test, pred) # 평가
print('결정 트리 예측 정확도 : {:.4f}'.format(accuracy))

# DecisionTreeClassifier의 하이퍼파라미터 추출
print('DecisionTreeClassifier 기본 하이퍼 라파미터 : \n', dt.get_params())


# In[10]:


# max_depth값을 변화시키면서 예측 성능을 확인

from sklearn.model_selection import GridSearchCV

params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24],
    'min_samples_split' : [16]
}

grid = GridSearchCV(dt, param_grid = params, scoring = 'accuracy', cv = 5, verbose = 1)
grid.fit(X_train, y_train)
print('GridSearchCv 최고 평균 정확도 수치 : {:.4f}'.format(grid.best_score_))
print('GridSearchCv 최고 하이퍼 파라미터 : ', grid.best_params_)


# In[11]:


# 5개의 cv 세트에서 max_depth 값에 따라 어떻게 예측 성능이 변했는지 확인
# gridsearchcv 객체의 cv_results_ 속성을 dataframe으로 생성

cv_results_df = pd.DataFrame(grid.cv_results_)

# max_depth 파라미터 값과 그때의 테스트 세트, 학습 데이터 세트의 정확도 수치 추출
cv_results_df[['param_max_depth', 'mean_test_score']]


# In[13]:


# 별도의 테스트 데이터 세트에서 결정 트리의 정확도 측정

max_depths = [6, 8, 10, 12, 16, 20, 24]
# max_depth 값을 변화시키면서 그때마다 학습과 테스트 세트에서의 예측 성능 측정

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth = depth, min_samples_split = 16, random_state = 156)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print('max_depth = {}정확도 : {:.4f}'.format(depth, accuracy))


# In[14]:


parms = {
    'max_depth' : [8, 12, 16, 20],
    'min_samples_split' : [16, 24]
}

grid = GridSearchCV(dt, param_grid = params, scoring = 'accuracy', cv = 5, verbose = 1)
grid.fit(X_train, y_train)
print('GridSearchCV최고 평균 정확도 수치 : {:.4f}'.format(grid.best_score_))
print('GridSearchCV최고 하이퍼 파라미터 : ', grid.best_params_)


# In[16]:


# 위 하이퍼 파라미터를 적용
best_dt = grid.best_estimator_
pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('결정 트리 예측 정확도 : {:.4f}'.format(accuracy))


# In[20]:


import seaborn as sns
import matplotlib.pylab as plt


ftr_importances_values = best_dt.feature_importances_

# top 중요도로 정렬을 쉽게 하고, sns의 막대그래프로 쉽게 표현하기 위해 series 변환
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)

# 중요도값 순으로 series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending = False)[:20]

plt.figure(figsize = (8, 6));
plt.title('Feature importances Top 20');
sns.barplot(x = ftr_top20, y = ftr_top20.index);


# In[ ]:




