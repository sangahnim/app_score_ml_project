## [**app_score_ml_project**](https://github.com/sangahnim/section_project/blob/main/section_2/AI_05_%EC%9D%B4%EC%83%81%EC%95%84_Section2.ipynb)


1. **프로젝트 주제** : 어플평점 4점을 넘기위해 고려해야 할 사항에 대한 Machine Learning Project

>* 해결하고자하는 문제 : 어플 고평점 예측
> * 데이터 : 캐글의 2017년 앱스토어 어플 통계데이터(7197개)
>> * 선정이유
>> * 코로나시국, 비대면 사회에서 가장 중요한 도구는 휴대폰이다.
>> * 사용할 Target 특성은 4.5이상 평점을 받은 binary data로 분류문제이다.
>> * 어떤 특징에 따라 고평점을 받을 수 있는지 없는지, 예측하는 모델링을 할 것이다.


2. **Import Library, Data** 



3. **가설, 기준모델(Baseline model), 평가지표설명**:

> **가설**

>> * 가설1 : 소프트웨어산업 중 가장 높은 비율을 차지하는 장르가 추천할 확률도 가장 높을 것이다.

>> * 어플 장르에 게임이 차지하는 비율이 압도적으로 높다.
>> * 전체를 기준으로 추천율, 비추천율을 장르별로 구분하여도 게임이 다른 장르에 비해서 모두 높다.
>> * 마지막 그래프인 recommendation ratio by genre2를 보면, 게임의 점유율과 무관하게 추천율은 높지 않음을 확인할 수 있으며, 가장 추천율이 높은 장르는 Book으로 확인하였다.


>> * 가설2 : 가격의 영향력은 크기 때문에, 저렴하면 추천될 확률이 높을 것이다.

>> * 추천과 비추천의 데이터 수가 무료에 다른 가격에 비해 많고, 추천과 비추천 모두 가격이 적을수록 수치가 높았다.
>> * 각 가격별로 추천비율을 구했을 때, 18.99달러와 23.99달러의 추천율이 1로, 데이터가 하나씩있는 이상치임을 확인하였다.
>> * 1이 아닌 비율 중 가장 추천율이 높은 가격은 8.99달러로 확인된다.


> **기준모델 및 평가지표설명**

>> * Target 특성은 어플 평점이 4.5점을 넘으면 True로 표시하는 새로운 특성을 생성.
>> * Baseline Model로는 초기 최빈값인 0.92에서 RandomForest의 AUC Score로 변경하였다


4. **EDA, preprocessing** :

>   1) 불필요한 컬럼 제거 및 특성이름정리 : 중복 등 불필요한 컬럼은 제거, 특성이름은 일관성있게 맞춰줌
>   2) data 정리 : ver 특성의 데이터 중 불필요한 데이터 제거 및  count_rating 특성의 데이터 정리 (ex) '6.3.5'-> 6, '4+'->4)
>   3) 이상치제거 : 데이터의 분포확인 후, user_rating_tot의 컬럼이 0이면 이상치로보고 제거 (어떤 회사라도 1개는 있을 것이다는 가정. 점수가 없다면 만들어진지 얼마 안된 어플일 것이다.)
>   4) Feature Engineering : recommendation 특성을 생성하여 True/False의 이진분류문제로 정리.


5. **Modeling**
    
> 1) 1차 모델링(RandomForest), 기준모델
>> * Ordinal Encoder로 RandomForest를 이용해서 1차 Baseline Modeling을 진행했다.
>> * 기본적인 모델링으로 특별한 Parameter없이 진행했으며, 훈련정확도가 1로 과적합임을 확인하였다.
>> * 하지만 검증 Set에서의 정확도가 최빈값 Baseline을 넘었으므로 이번 Modeling에서 유심히 볼 평가지표인 'AUC Score'를 Baseline으로 두고 진행한다.
>> * Target의 imbalance함이 score에 영향을 주어 소수인 '1'에 대한 score는 전체적으로 낮은데, 밸런스화가 이번 모델링의 핵심으로 보인다.
>> * 추후 SMOTE를 적용하면 얼마나 성능이 좋아질 지 확인해보기로 한다.
>> * AUC 점수 : 0.5243213993213992

> 2) 2차 모델링 (CatBoost)
>> *  AUC 점수 : 0.5243213993213992

> 3) Hyper Parameter Tuning the RandomForest with SMOTE
>> RandomizedSearchCV를 이용하여 hyper parameter를 동일하게 적용하고, imbalance를 조정한다.  
>> (1) RandomForest with SMOTE1 : auc점수 :  0.5746753246753247  
>> (2) RandomForest with SMOTE2 : auc점수 :  0.5  
>> (3) RandomForest with SMOTE3 : auc점수 :  0.5  

> 4) Hyper Parameter Tuning the CatBoost with SMOTE
>> RanomizedSearch를 이용해서 Hyper parameter를 동일하게 적용하고 imbalance를 조정한다.  
>> (1) CatBoost with SMOTE1 : auc점수 :  0.5397069147069148  
>> (2) CatBoost with SMOTE2 : auc점수 :  0.5118901368901368  
>> (3) CatBoost with SMOTE3 : auc점수 :  0.5118901368901368  

> 5) Hyper Parameter Tuning the RandomForest with OverSampling
>> SMOTE와 동일한 조건으로 모델링한다.  
>> auc점수 :  0.59496021996022

> 6) Hyper Parameter Tuning the CatBoost with OverSampling
>> SMOTE와 동일한 조건으로 모델링한다.  
>> (1) CatBoost with OverSampling1 : auc점수 :  0.5386392886392886  
>> (2) CatBoost with OverSampling2 : auc점수 :  0.549973674973675  


6. **머신러닝모델 해석결과**

> 1) 최종모델 : OverSampling 적용된 RandomForest모델(Threshold 적용).
>> Validation set 기준 auc점수 :  0.6114718614718615  
>> Test set 기준     auc점수 :  0.5808398950131233

> 2) Feature Importance  

>> Feature Importance는 트리기반 앙상블모델에서 사용되는 중요도로 노드가 중요할수록 불순도가 크게 감소하는 특징을 이용하여 중요도를 나타낸 것으로, lang_num, prime_genre, size_bytes Feature가 중요함을 보여준다.  

> 3) Permutation Importance (순열 중요도)  

>> Permutation Importance는 무작위로 섞어 노이즈로 만들기에 예측값과 실제값이 어람나 차이나는지 영향력을 파악하는 것인데, rating_count_tot, size_bytes, ipadSc_urls_num의 순으로 중요한 Feature임을 보여준다.

> 4) PDP
>> * Feature Importance에서 중요한 상위 4가지 Feature에 대해 PDPlot 확인

> 5) 프로젝트 회고
  
>> * Feature Importance와 Permutation Importance에서의 특성이 비슷하다고 하기는 어려웠고, PDP에서 추천되는 부분에 대한 시각화가 잘 나타나지 않았다. 
>> * Imbalance한 데이터의 모델링에 성능을 높이기 위해 많은 시도를 하였으나, 최종모델에 Test Data를 넣었을 때, AUC Score가 0.58로 마무리지은 것이 아쉽다.
>> * 모델의 성능이 좋다고 하기는 어려운데, 이는 데이터 특성의 정보가 부족하기 때문이라고 생각한다.
>> * 어플의 추천시 특성상 주관적인 판단이 많이 들어가는데, 캐글 데이터로 주관적인 내용을 구분하기는 쉽지 않았다.
>> * 연령대나 성별 등 이용자의 정보가 추가적으로 특성에 반영된다면 더 좋은 모델링을 할 수 있을 것이라 기대한다. 
>> * 어플의 특성은 주관적인 판단이 많이 들어가는데, 캐글 데이터의 특성은 한계가 있었고, 주관적인 내용은 사실 구분하기 쉽지않다.
>> * 연령대나 성별 등 이용자의 정보가 추가적으로 특성에 반영된다면 더 좋은 모델링을 할 수 있을 것이라 기대한다.