
## Filtering noise data in SNS Using one-class SVM


-  글로우픽의 경우 화장품 리뷰를 크롤링한 것이므로 모두 화장품과 관련된 댓글이지만, 유튜브, 네이브 블로그, 인스타그램 등 SNS 데이터의 경우 noise가 상당수 발견됨
-  예를 들면 해당 게시물을 올린 유튜버를 향한 댓글, 블로거와 대화 내용 등이 존재 

   ![image](https://user-images.githubusercontent.com/60679596/139695908-0751db33-255c-46d1-aa23-df3e4f38d1bd.png)
   ![image](https://user-images.githubusercontent.com/60679596/139695918-0569dae6-5c94-4a5a-b185-9d4e1932b2ff.png)
   ![image](https://user-images.githubusercontent.com/60679596/139695926-13ae2a67-e57f-4406-ba4c-39c3bbfbb8be.png)

- 이러한 이유로 수집한 댓글 데이터 중 화장품과 관련 없는 댓글인 스팸성 데이터를 검출해내고자 한다 

<br/>
<br/>

### Dataset
- 화장품과 관련 있는 데이터는 1, 화장품과 관련 없는 데이터는 -1  레이블 지정
- 글로우픽 리뷰 데이터의 경우 모두 화장품과 관련 있는 데이터이므로, 레이블을 모두 1로 두고, 타 플랫폼의 경우 테스트를 위한 데이터셋을 구성하기 위해 직접 하나씩 확인하여 레이블링 진행
- 학습 데이터로는 레이블이 1인 데이터만을 이용 (비율 : 글로우픽 데이터 20000 + 타 플랫폼 label 1인 데이터 3000)


   ![image](https://user-images.githubusercontent.com/60679596/139696345-da2ce5c0-5737-4305-935a-4789a4063f8b.png)

    <레이블링 결과>

<br/>
<br/>

### Modeling
- Oneclass SVM
- One Class SVM이란 주어진 데이터를 잘 설명할 수 있는 최적의 support vector를 구하고 이 영역 밖의 데이터들은 outlier로 간주하는 방식
- 파라미터는 grid search 활용해서 설정

![image](https://user-images.githubusercontent.com/60679596/146883204-31a67838-1424-4599-b852-064ac088a1db.png)


그림 출처 : https://www.researchgate.net/figure/One-class-SVM-boundary-and-outlier-detection_fig5_281455041

<br/>
<br/>

### 실험 결과
- nu값에 따라 성능 변화가 큰 것을 확인
- 해당 실험은 noise data(label=-1)를 잘 걸러내는 것이 중요하므로, 정확도가 어느 정도 이상일때 -1에 대한 Recall값이 큰 모델이 이상적이라 판단
- nu값이 1.5, 2, 2.5일때의 모델 결과를 각각 저장

