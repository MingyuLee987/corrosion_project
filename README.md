# 희토류 첨가 부식 분석

`실행.bat`을 더블클릭하거나 PowerShell에서 `./run.ps1`을 실행합니다.
Python 환경이 없으면 Python 설치가 필요하며, 첫 라이브러리 설치에는 인터넷이 필요합니다.

- `corrosion_re_expand.py`: 독립 실행 코드. 47행 기준과 3행 추가 후 50행을 논문 단위로 비교합니다.
- `data/training_51.csv`: 원본 입력(실제 47행).
- `data/training_RE_expanded.csv`: 실행 시 생성하는 50행 데이터.
- `results/RE_expanded/`: 실행 시 생성하는 지표와 예측 결과.
- `requirements.txt`, `.venv/`: 실행 라이브러리와 환경.

결측값은 학습 폴드 안에서만 대체합니다. 현재 결과는 탐색 분석이며 별도 예측 배포 모델은 없습니다.
추가 3행의 출처·전사값과 기존 검증 PDF 해시를 코드에 보존하여 논문 PDF 없이 실행합니다.
