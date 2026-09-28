# 듀얼하면서 배우는 Anima3 / Jev

두 일반 클라이언트가 7x + Explosion Potion **training** 경기를 진행합니다.
경기장 서버는 경기와 리플레이만 관리하며, 학습은 참가자가 실행하는 anima3에서 합니다.

## 무엇을 배우나

`standard`, `control`, `poison`, `interrupt`, `sustain` 다섯 전술의 경기 보상을 누적합니다.
승리=1, 패배/무승부=0인 보수적인 보상으로 전술을 비교합니다. 각 경기 후 승률,
Beta(1,1) 사전분포를 적용한 승률 평균, Wilson 구간을 저장합니다.
이는 유한 전술 정책 학습입니다. Jev/Qwen의 신경망 가중치를 fine-tuning하지 않습니다.
Jev는 허용된 행동 메뉴에서 선택하며 실패·지연 시 기존 전투 규칙을 계속 실행합니다.

- 전술별 최소 10경기를 모두 수집한 다음 후보를 고정합니다.
- 후보와 현재 champion의 **새로운 40경기**를 평가합니다. 계정과 경기장 방향을 교대합니다.
- 99% 승률 구간 하한 > 0.5 및 방향별 표본 조건을 만족해야 자동 승격합니다.
- 후보가 실패하면 추가 평가로 유리한 표본을 골라내지 않습니다.
- 학습 경기에는 랭킹이 적용되지 않습니다. 사람과의 공개 경기는 승격 평가에 섞지 않습니다.
- 서버 리플레이 SHA256, 참여자, 규칙, 완료·승패를 재검증합니다. 중단 경기는 학습하지 않습니다.

## 실행

```sh
uv sync --extra dev --extra jeff
# 아래 환경변수 값은 로컬에서 설정합니다. 키/암호를 문서나 Git에 넣지 마세요.
export SPAR_PASSWORD_A='...'
export SPAR_PASSWORD_B='...'
export TYPESAFE_API_KEY='...'
uv run --extra jeff python -m anima3.sparring \
  --user-a MY_TRAIN_A --user-b MY_TRAIN_B \
  --backend jev --model jev-1.13.0 \
  --matches 90 --minimum 10 --sample 40 --max-model-calls 1000 \
  --log-dir .logs/learning-jev
```

`--backend scripted`는 외부 API 없이 실행됩니다. `qwen`, `jeff`도 선택할 수 있습니다.
Jev용 설치 extra 이름은 기존 호환성을 위해 `jeff`입니다. Jev 모델 버전을 고정합니다.
두 계정은 전용 일반 플레이어 계정이어야 하며, 접속 중인 사용자의 계정을 사용하지 마세요.

한 실행당 클라이언트별 API 호출 상한을 적용합니다. 재시도는 하지 않으며 요청 timeout은
1.2초입니다. 상한에 도달하면 현재 경기는 기본 전투 규칙으로 끝내고 다음 경기를 시작하지
않습니다. **호출 상한 때문에 규칙으로 대체된 경기는 승격용 학습 표본에서 제외**합니다.
다시 실행하면 호출 예산이 새로 시작됩니다. `--matches`는 재개 전 완료 경기까지 포함한 총수입니다.
90경기는 초기 수집+첫 평가를 위한 한도이며 반드시 승격하거나 평가가 끝난다는 뜻은 아닙니다.

같은 디렉터리는 backend/model/계정/서버/평가 조건이 같을 때만 재개할 수 있습니다.
다른 모델 비교에는 별도 디렉터리를 사용하세요. 부분 기록·손상된 리플레이는 자동 덮어쓰지 않습니다.

## 학습 결과를 대전 에이전트에 적용

```sh
export ARENA_BOT_PASSWORD='...'
uv run --extra jeff python -m anima3.duel_wait \
  --user MY_PUBLIC_BOT --backend jev --model jev-1.13.0 \
  --policy-file .logs/learning-jev/policies/champion.json \
  --max-model-calls 2000 --log-dir .logs/learning-public
```

champion 파일은 접속 준비 및 **다음 경기 준비 시점**에 읽습니다. 경기 도중 정책을 교체하지
않습니다. 정책 backend/model이 실행 옵션과 다르면 시작을 거부합니다. 후보 파일을 직접 지정하지
말고 champion 파일을 사용하세요. 대전 대기 에이전트의 호출 예산은 재접속 후에도 유지됩니다.
예산 소진 후 대기 에이전트는 규칙으로 계속 응답하므로 새 실행 전 로그를 확인하세요.

## 확인할 파일

| 파일 | 내용 |
|---|---|
| `experiment.json` | 고정한 실험 조건, 비밀정보 없음 |
| `learning.jsonl` | 경기별 서버 근거, 정책, 승패, 호출 수 |
| `learning-progress.json` | 전술별 보상 통계와 남은 표본 |
| `status.json` | 현재 경기·학습 단계·중단 사유 |
| `<경기ID>/replay.jsonl` | 다운로드한 서버 리플레이 |
| `<경기ID>/player-*-round-*.jsonl` | 상태·허용 행동·선택·모델 채택 여부 |
| `policies/evaluation.json` | 고정 표본 평가와 통과 여부 |
| `policies/history/` | 이전 champion 및 승격/실패 근거 |

`python -m anima3.benchmark .logs/learning-jev --output .logs/learning-jev/benchmark.json`
으로 고정된 정책 버전의 평가 결과를 비교할 수 있습니다. 초기 단계에는 표본 부족이 정상입니다.

Jev API: https://github.com/typesafe-ai/typesafe-sdk-python
