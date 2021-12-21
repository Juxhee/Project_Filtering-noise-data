## Ifamily Project 1 
### spam성 noise data 필터링 실험

### 1. 개요
#### -  DB에 수집한 글로우픽, 유튜브, 네이버 블로그, 파우더룸의 댓글 데이터 중 글로우픽 리뷰 데이터는 모두 화장품과 관련된 댓글이지만, 타 플랫폼의 댓글에는 noise가 상당수 발견됨
#### -  예를 들면 해당 게시물을 올린 유튜버를 향한 댓글, 블로거와 대화 내용 등이 존재 

![image](https://user-images.githubusercontent.com/60679596/139695908-0751db33-255c-46d1-aa23-df3e4f38d1bd.png)
![image](https://user-images.githubusercontent.com/60679596/139695918-0569dae6-5c94-4a5a-b185-9d4e1932b2ff.png)
![image](https://user-images.githubusercontent.com/60679596/139695926-13ae2a67-e57f-4406-ba4c-39c3bbfbb8be.png)

#### 이러한 이유로 수집한 댓글 데이터 중 화장품과 관련 없는 댓글인 spam성 데이터를 검출해내고자 한다 

###

### 2. 데이터 구성 
#### - 화장품과 관련 있는 데이터는 1, 화장품과 관련 없는 데이터는 -1  레이블 지정
#### - 글로우픽 리뷰 데이터의 경우 모두 화장품과 관련 있는 데이터이므로, 레이블을 모두 1로 두고, 타 플랫폼의 경우 테스트를 위한 데이터셋을 구성하기 위해 직접 하나씩 확인하여 레이블링 진행
#### - 학습 데이터로는 레이블이 1인 데이터만을 이용


![image](https://user-images.githubusercontent.com/60679596/139696345-da2ce5c0-5737-4305-935a-4789a4063f8b.png)

<레이블링 결과 예시>



### 3) 활용 모델 : Oneclass SVM
- One Class SVM이란 주어진 데이터를 잘 설명할 수 있는 최적의 support vector를 구하고 이 영역 밖의 데이터들은 outlier로 간주하는 방식
- 파라미터는 grid search 활용해서 설정

![image](https://user-images.githubusercontent.com/60679596/146883204-31a67838-1424-4599-b852-064ac088a1db.png)


그림 출처 : https://www.researchgate.net/figure/One-class-SVM-boundary-and-outlier-detection_fig5_281455041
