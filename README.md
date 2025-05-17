# stock_ai

주식 예측 모델 (Stock Prediction Model)
한국 주식 시장 분석 및 예측을 위한 머신러닝 기반 도구입니다. XGBoost를 활용하여 주가 방향 예측 및 자동 트레이딩 신호를 생성합니다.
주요 기능

방향성 예측: 향후 N일 주가 상승/하락 예측 (분류 모델)
자동 트레이딩 신호: 매수/매도/관망 신호 자동 생성
재무정보 통합: 네이버 금융 데이터와 기술적 지표 결합
백테스트: 예측 모델의 수익률, 샤프 비율 등 성과 측정
병렬 처리: 여러 종목 동시 분석으로 처리 속도 향상
결과 보고서: 매매 추천 종목, 상세 분석 결과 자동 저장

필요 라이브러리
pandas
numpy
matplotlib
seaborn
sklearn
xgboost
talib (기술적 지표)
yfinance
tqdm
beautifulsoup4
requests
TA-Lib 설치 방법
기술적 지표를 계산하기 위한 TA-Lib 설치가 필요합니다. 다음 방법 중 하나를 선택하세요:
방법 1: Conda 사용
bashconda install -c conda-forge ta-lib
방법 2: pip으로 설치 (Windows)
bashpip install TA-Lib
방법 3: 소스 컴파일 (Linux/macOS)
bash# C 라이브러리 설치
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Python 패키지 설치
pip install ta-lib
방법 4: 미리 컴파일된 wheel 사용
bashpip install https://files.pythonhosted.org/packages/00/37/0896c185651269bb768bd3d2bd77ab1020bce9fd7adcf5e1ce08a1c13ee/TA_Lib-0.4.28-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
대체 방법: ta 라이브러리 사용
TA-Lib 설치가 어려운 경우, ta 라이브러리를 대안으로 사용할 수 있습니다:
bashpip install ta
이 경우 코드 수정이 필요합니다.
주요 클래스 및 기능
1. StockPredictionModel
주식 가격 예측을 위한 핵심 클래스입니다.
python# 모델 초기화
model = StockPredictionModel(
    prediction_type='classification',  # 'classification' 또는 'regression'
    prediction_period=5,               # 향후 예측 기간 (일)
    confidence_threshold=0.6           # 매매 신호 확신도 임계값
)

# 데이터 로드
model.load_data(df=stock_data)

# 특성 생성
model.create_features(advanced=True, stock_code='005930')  # 삼성전자

# 데이터 준비
model.prepare_data(test_size=0.2)

# 모델 학습
model.train_model(optimize=True)

# 모델 평가
metrics = model.evaluate_model()

# 백테스트
backtest_results = model.backtest()

# 미래 예측
predictions = model.predict_future(days=1)
2. StockAnalyzer
여러 종목을 동시에 분석하고 결과를 비교할 수 있는 클래스입니다.
python# 분석기 초기화
analyzer = StockAnalyzer()

# KRX 종목 코드 가져오기
stock_codes = analyzer.get_krx_stock_codes()

# 병렬 처리로 여러 종목 분석
results = analyzer.analyze_all_stocks_parallel(
    stock_codes=stock_codes, 
    period='5y',
    max_stocks=100,  # 최대 분석 종목 수
    workers=4        # 병렬 작업자 수
)

# 상위 종목 가져오기
top_stocks = analyzer.get_top_stocks(criteria='backtest_return', top_n=10)

# 보고서 생성
report = analyzer.generate_report()
3. 주요 함수

get_financial_info(stock_code): 네이버 금융에서 재무정보 스크래핑
create_ensemble_model(stock_code, period, prediction_period): 재무정보와 기술적 지표를 결합한 앙상블 모델 생성
ensemble_predictions(stock_code, periods): 여러 예측 기간의 모델을 앙상블하여 예측

사용 예시
단일 종목 분석
pythonfrom stock_prediction import StockPredictionModel
import yfinance as yf

# 삼성전자 데이터 다운로드
ticker = "005930.KS"
stock = yf.Ticker(ticker)
df = stock.history(period="5y")
df.reset_index(inplace=True)

# 모델 초기화 및 학습
model = StockPredictionModel(prediction_period=5)
model.load_data(df=df)
model.create_features(advanced=True, stock_code='005930')
model.prepare_data()
model.train_model(optimize=True)

# 백테스트 및 미래 예측
backtest_results = model.backtest()
predictions = model.predict_future()

print(f"예측 방향: {predictions['Prediction'].iloc[0]}")
print(f"확신도: {predictions['Confidence'].iloc[0]:.4f}")
print(f"백테스트 수익률: {backtest_results['total_return']:.4f}")
다중 종목 분석
pythonfrom stock_prediction import StockAnalyzer

# 분석기 초기화
analyzer = StockAnalyzer()

# 코스피/코스닥 상위 30개 종목 분석
krx_stocks = analyzer.get_krx_stock_codes()
results = analyzer.analyze_all_stocks_parallel(
    stock_codes=krx_stocks,
    max_stocks=30
)

# 상위 종목 출력
top_stocks = analyzer.get_top_stocks(criteria='backtest_return', top_n=5)
print(top_stocks)
결과 파일
프로그램 실행 시 다음과 같은 결과 파일이 생성됩니다:

buy_signals_YYYYMMDD.csv: 매수 추천 종목 목록
sell_signals_YYYYMMDD.csv: 매도 추천 종목 목록
optimized_signals_YYYYMMDD.csv: 최적화된 매매 신호 목록
all_stock_analysis_YYYYMMDD.csv: 전체 분석 결과

주의사항

Yahoo Finance API를 사용하여 데이터를 가져오므로, 인터넷 연결이 필요합니다.
네이버 금융 크롤링은 데이터 제공 방식이 변경되면 작동하지 않을 수 있습니다.
TA-Lib 설치가 까다로울 수 있으니, 위의 설치 방법을 참고하세요.
모델 최적화 시간이 오래 걸릴 수 있으므로, 필요에 따라 optimize=False 옵션을 사용하세요.
이 프로그램은 투자 조언을 제공하지 않으며, 실제 투자 결정은 사용자 본인의 책임입니다.

향후 개선 사항

딥러닝 모델 (LSTM, Transformer 등) 추가
감성 분석 및 뉴스 데이터 통합
웹 인터페이스 구현
실시간 트레이딩 API 연동
포트폴리오 최적화 기능 추가
