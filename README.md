## [**app_score_ml_project**](https://github.com/sangahnim/section_project/blob/main/section_2/AI_05_%EC%9D%B4%EC%83%81%EC%95%84_Section2.ipynb)


1. **프로젝트 주제** : 어플평점 4점을 넘기위해 고려해야 할 사항에 대한 Machine Learning Project

>* 해결하고자하는 문제 : 어플 고평점 예측
> * 데이터 : 캐글의 2017년 앱스토어 어플 통계데이터(7197개)
>> * 선정이유
>> * 코로나시국, 비대면 사회에서 가장 중요한 도구는 휴대폰이다.
>> * 사용할 Target 특성은 4.5이상 평점을 받은 binary data로 분류문제이다.
>> * 어떤 특징에 따라 고평점을 받을 수 있는지 없는지, 예측하는 모델링을 할 것이다.


2. **Import Library, Data** 

<img width="843" alt="스크린샷 2022-07-29 오후 4 59 31" src="https://user-images.githubusercontent.com/86824895/181713402-4d24781b-4c6f-459f-ba18-771edca58f59.png">
<img width="859" alt="스크린샷 2022-07-29 오후 4 59 42" src="https://user-images.githubusercontent.com/86824895/181714233-5bc7e1e8-aa3a-4abe-8675-30fbabce5626.png">

<img width="833" alt="스크린샷 2022-07-29 오후 4 59 51" src="https://user-images.githubusercontent.com/86824895/181714248-806a5a2c-f06f-4076-ac93-8cfaa308aee2.png">




3. **가설, 기준모델(Baseline model), 평가지표설명**:

> **가설**

>> * 가설1 : 소프트웨어산업 중 가장 높은 비율을 차지하는 장르가 추천할 확률도 가장 높을 것이다.
<img width="929" alt="스크린샷 2022-07-29 오후 5 00 01" src="https://user-images.githubusercontent.com/86824895/181713519-cb88cd1a-8fde-4cbf-8c44-7fb7d8fa0090.png">

<img width="838" alt="스크린샷 2022-07-29 오후 5 00 11" src="https://user-images.githubusercontent.com/86824895/181713567-64ec338e-32c2-4aa5-a3d8-7ad4ecb7f0a4.png">

>> * 어플 장르에 게임이 차지하는 비율이 압도적으로 높다.
>> * 전체를 기준으로 추천율, 비추천율을 장르별로 구분하여도 게임이 다른 장르에 비해서 모두 높다.
>> * 마지막 그래프인 recommendation ratio by genre2를 보면, 게임의 점유율과 무관하게 추천율은 높지 않음을 확인할 수 있으며, 가장 추천율이 높은 장르는 Book으로 확인하였다.


>> * 가설2 : 가격의 영향력은 크기 때문에, 저렴하면 추천될 확률이 높을 것이다.


<img width="909" alt="스크린샷 2022-07-29 오후 5 00 22" src="https://user-images.githubusercontent.com/86824895/181713582-962de1ec-9198-4cae-a582-316a30b89ca2.png">


>> * 추천과 비추천의 데이터 수가 무료에 다른 가격에 비해 많고, 추천과 비추천 모두 가격이 적을수록 수치가 높았다.
>> * 각 가격별로 추천비율을 구했을 때, 18.99달러와 23.99달러의 추천율이 1로, 데이터가 하나씩있는 이상치임을 확인하였다.
>> * 1이 아닌 비율 중 가장 추천율이 높은 가격은 8.99달러로 확인된다.


> **기준모델 및 평가지표설명**

>> * Target 특성은 어플 평점이 4.5점을 넘으면 True로 표시하는 새로운 특성을 생성.
<img width="800" alt="스크린샷 2022-07-29 오후 5 00 34" src="https://user-images.githubusercontent.com/86824895/181713924-88c9f57d-0307-4a9e-94c5-cd5f1995a04a.png">


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

> 3) Permutation Importance (순열 중요도)  

<img width="812" alt="스크린샷 2022-07-29 오후 5 01 05" src="https://user-images.githubusercontent.com/86824895/181713781-f49aaadb-e686-46ce-8ad8-f61305157208.png">

>> Feature Importance는 트리기반 앙상블모델에서 사용되는 중요도로 노드가 중요할수록 불순도가 크게 감소하는 특징을 이용하여 중요도를 나타낸 것으로, lang_num, prime_genre, size_bytes Feature가 중요함을 보여준다.  

>> Permutation Importance는 무작위로 섞어 노이즈로 만들기에 예측값과 실제값이 어람나 차이나는지 영향력을 파악하는 것인데, rating_count_tot, size_bytes, ipadSc_urls_num의 순으로 중요한 Feature임을 보여준다.

> 4) PDP
<img width="720" alt="스크린샷 2022-07-29 오후 5 00 51" src="https://user-images.githubusercontent.com/86824895/181713763-08f5e77c-9af8-4cba-b41e-ed105227dee7.png">

>> * Feature Importance에서 중요한 상위 4가지 Feature에 대해 PDPlot 확인

> 5) 프로젝트 회고
  
>> * Feature Importance와 Permutation Importance에서의 특성이 비슷하다고 하기는 어려웠고, PDP에서 추천되는 부분에 대한 시각화가 잘 나타나지 않았다. 
>> * Imbalance한 데이터의 모델링에 성능을 높이기 위해 많은 시도를 하였으나, 최종모델에 Test Data를 넣었을 때, AUC Score가 0.58로 마무리지은 것이 아쉽다.
>> * 모델의 성능이 좋다고 하기는 어려운데, 이는 데이터 특성의 정보가 부족하기 때문이라고 생각한다.
>> * 어플의 추천시 특성상 주관적인 판단이 많이 들어가는데, 캐글 데이터로 주관적인 내용을 구분하기는 쉽지 않았다.
>> * 연령대나 성별 등 이용자의 정보가 추가적으로 특성에 반영된다면 더 좋은 모델링을 할 수 있을 것이라 기대한다. 
>> * 어플의 특성은 주관적인 판단이 많이 들어가는데, 캐글 데이터의 특성은 한계가 있었고, 주관적인 내용은 사실 구분하기 쉽지않다.
>> * 연령대나 성별 등 이용자의 정보가 추가적으로 특성에 반영된다면 더 좋은 모델링을 할 수 있을 것이라 기대한다.

![MLProject 001](https://user-images.githubusercontent.com/86824895/181692186-855167ba-4b77-41a3-a606-2b0647c66042.jpeg)
![MLProject 002](https://user-images.githubusercontent.com/86824895/181692197-dfe9cdcb-ab14-4d65-825a-b58ab3cc8529.jpeg)
![MLProject 003](https://user-images.githubusercontent.com/86824895/181692204-3c553bae-719a-416a-8718-ae817d946a78.jpeg)
![MLProject 004](https://user-images.githubusercontent.com/86824895/181692209-c07e60aa-65f1-4678-a3f3-cc212eefa72e.jpeg)
![MLProject 005](https://user-images.githubusercontent.com/86824895/181692217-b565c579-76c9-417c-893e-fb1bc4bab010.jpeg)
![MLProject 006](https://user-images.githubusercontent.com/86824895/181692219-62d37e81-b4c2-4245-b7ce-a4fbe099c608.jpeg)
![MLProject 007](https://user-images.githubusercontent.com/86824895/181692224-d6afdb53-1c56-492f-98ef-4baff49bab97.jpeg)
![MLProject 008](https://user-images.githubusercontent.com/86824895/181692227-22a4e492-0f03-43ab-b122-64f0412efede.jpeg)
![MLProject 009](https://user-images.githubusercontent.com/86824895/181692228-959cfd69-e53e-46f7-b5b1-81714d708fed.jpeg)
![MLProject 010](https://user-images.githubusercontent.com/86824895/181692232-117eaeba-2d86-45a3-9576-bbc6a4cd545b.jpeg)
![MLProject 011](https://user-images.githubusercontent.com/86824895/181692235-8a89057a-59e4-4b00-bde0-ff4037964a7c.jpeg)
![MLProject 012](https://user-images.githubusercontent.com/86824895/181692237-b82beae8-47db-413e-8c6c-154c9d78c1b4.jpeg)
![MLProject 013](https://user-images.githubusercontent.com/86824895/181692238-5556d732-0880-4ce9-bc18-83dfab70e045.jpeg)
![MLProject 014](https://user-images.githubusercontent.com/86824895/181692239-abba1f36-1995-490b-9d78-67a4b2260925.jpeg)
![MLProject 015](https://user-images.githubusercontent.com/86824895/181692243-8a1ac303-fb2d-4ab4-96e9-88ee3c328500.jpeg)
![MLProject 016](https://user-images.githubusercontent.com/86824895/181692247-41e61798-d181-4c98-8d73-b02dafbb0845.jpeg)
![MLProject 017](https://user-images.githubusercontent.com/86824895/181692250-0bdf0642-9d1e-45b4-a4cd-8e3904bc2174.jpeg)
![MLProject 018](https://user-images.githubusercontent.com/86824895/181692251-2dd929cc-ab59-4f29-832f-9a120872f7dd.jpeg)
![MLProject 019](https://user-images.githubusercontent.com/86824895/181692255-fdb15354-f2f1-40ed-8a4d-14d887c558d4.jpeg)
![MLProject 020](https://user-images.githubusercontent.com/86824895/181692258-468f77a2-321d-4407-ba1e-93079dc4c3d4.jpeg)
![MLProject 021](https://user-images.githubusercontent.com/86824895/181692261-84daecfb-4e90-4800-8d22-e21b1ef4d18f.jpeg)
![MLProject 022](https://user-images.githubusercontent.com/86824895/181692264-647d3473-04ef-4888-a78a-3ebada9a4f11.jpeg)
![MLProject 023](https://user-images.githubusercontent.com/86824895/181692266-1c2985ba-d605-4bea-80f1-95cedebc6834.jpeg)
![MLProject 024](https://user-images.githubusercontent.com/86824895/181692267-12b0fb91-c282-49b3-b36f-11b0c18f3c95.jpeg)
![MLProject 025](https://user-images.githubusercontent.com/86824895/181692270-d887706d-9f47-4cc2-9fb5-3c7f6d7f7545.jpeg)
![MLProject 026](https://user-images.githubusercontent.com/86824895/181692272-2e6a4064-3162-497b-b3ea-a7a90d922e23.jpeg)
![MLProject 027](https://user-images.githubusercontent.com/86824895/181692275-682a5c26-c35d-4e2a-9ca6-01fd45606572.jpeg)
![MLProject 028](https://user-images.githubusercontent.com/86824895/181692279-506cd443-8958-4c68-bdee-8515ab3a9db6.jpeg)
![MLProject 029](https://user-images.githubusercontent.com/86824895/181692281-62e2b558-6292-4c77-912c-6f8dd400cd55.jpeg)
![MLProject 030](https://user-images.githubusercontent.com/86824895/181692285-8c312276-afa2-48b5-a6b5-c7f868728a8f.jpeg)
![MLProject 031](https://user-images.githubusercontent.com/86824895/181692287-8ce35343-2c85-40a1-9b83-9bab88f74aa8.jpeg)
![MLProject 032](https://user-images.githubusercontent.com/86824895/181692291-0dfae277-1336-41e7-8caa-1f7be9d17368.jpeg)
![MLProject 033](https://user-images.githubusercontent.com/86824895/181692292-8742620a-1ba3-4961-b8ed-93d924076cfc.jpeg)
![MLProject 034](https://user-images.githubusercontent.com/86824895/181692293-673bff54-8178-4e80-a75f-95c3edee5b26.jpeg)
![MLProject 035](https://user-images.githubusercontent.com/86824895/181692297-6aac92fc-3fb1-4881-89d3-8250095e6823.jpeg)
![MLProject 036](https://user-images.githubusercontent.com/86824895/181692301-0117fe1f-f5ea-45e1-a890-cb1cff2b5328.jpeg)
![MLProject 037](https://user-images.githubusercontent.com/86824895/181692303-5053590c-7a34-4111-88c8-8f945ab1f039.jpeg)
![MLProject 038](https://user-images.githubusercontent.com/86824895/181692304-5f7a6ac3-eea6-4d2b-ac17-92f1818db0a0.jpeg)

