# Jev + LLM 전략 에이전트

`--brain hybrid --backend jev`를 사용하면 세 계층이 함께 동작합니다.

| 계층 | 역할 | 실행 주기 |
|---|---|---|
| 전투 규칙 | 허용 행동, 시전·타깃·포션·해독·이동 실행 | 매 tick, 모델을 기다리지 않음 |
| Jev | 현재 위험과 LLM 계획을 보고 단기 전술 선택 | 기본 8초, 위험 조건 변경 시 최소 2초 간격 |
| 로컬 Qwen LLM | 상대 주문·자원·지난 경기 결과로 기본 전술과 조건별 대응 계획 작성 | 라운드 시작, 이후 최소 30초 간격 |

현재 로컬 Qwen3-4B-4bit 모델을 사용합니다. LLM 계획은 `primary`, `responses`, `reason`
JSON으로만 받습니다. 대응 조건은 저체력·중독·마나 부족·상대 시전·쇼다운입니다.
선택 가능한 전술은 standard/control/poison/interrupt/sustain입니다. 모델이 임의의 명령이나
스킬·스탯·서버 규칙 변경을 실행할 수 없으며, 쇼다운 등 서버 제한은 전투 규칙이 계속 적용합니다.

Jev가 급한 상황에 맞춰 LLM의 전술을 바꿀 수 있습니다. LLM/Jev가 실패하면 현재 전술과 기존
전투 규칙을 계속 실행합니다. 이전 경기·라운드·상대에 대한 응답은 적용하지 않습니다.
Jev 응답은 1.5초, LLM 응답은 20초가 지나면 폐기합니다. 느린 요청이 남아 있으면 새 스레드를
추가하지 않습니다. 대기 봇의 호출 예산은 재접속·라운드 변경으로 초기화되지 않습니다.

## 대전 대기

```sh
uv sync --extra jeff --extra qwen --extra dev
# 로컬에 준비한 암호와 API 키를 환경변수로 설정
export ARENA_BOT_PASSWORD='...'
export TYPESAFE_API_KEY='...'
uv run --extra jeff --extra qwen python -m anima3.duel_wait \
  --user MY_BOT --backend jev --brain hybrid --model jev-1.13.0 \
  --llm-model ~/dev/jev/models/Qwen3-4B-4bit \
  --version hybrid-v1 --max-model-calls 1000 --max-strategy-calls 100 \
  --log-dir .logs/hybrid-public
```

Qwen 추론은 Mac에서 실행되고, Jev에는 게임 상태와 전략만 보냅니다. 상대의 숨겨진 마나나
서버 관리자 정보는 포함하지 않습니다. 자유 채팅은 프롬프트에 넣지 않고 알려진 마법 주문만
분류해서 사용합니다. `--llm-model`은 이미 설치된 MLX 모델 디렉터리여야 합니다.

테스트한 로컬 패키지 조합: mlx/metal 0.32.2, mlx-lm 0.31.3, typesafe-sdk 0.7.2.

## 스파링과 경험 축적

```sh
uv run --extra jeff --extra qwen python -m anima3.sparring \
  --user-a MY_TRAIN_A --user-b MY_TRAIN_B --backend jev --brain hybrid \
  --matches 10 --max-model-calls 200 --max-strategy-calls 30 \
  --log-dir .logs/hybrid-sparring
```

대전 종료 후 SHA256·참가자·완료 여부를 검증한 서버 리플레이를 분석해 상대별 최근 결과를
저장합니다. 이후 동일 상대·동일 규칙의 계획에 최대 4경기를 참고합니다. 저장은 최대 50경기로
제한됩니다. 공개 대전은 백그라운드에서 리플레이를 가져오므로 대기와 전투를 막지 않습니다.
중단되거나 검증할 수 없는 경기는 승패 경험에 추가하지 않고 로그에 남깁니다.

**적응형 hybrid 경기와 고정 정책 평가를 분리합니다.** 경기 중 LLM이 전술을 바꾸는 경기로
기존 champion 승격 조건을 충족했다고 주장하지 않습니다. hybrid 로그는 별도 디렉터리를
사용하며 `adaptive_experience`로 기록하고, 고정 정책 학습/benchmark에서 제외합니다.
직접 정책 학습을 이어가려면 [학습 안내](ARENA_LEARNING.ko.md)의 `--brain direct` 기본 모드를 사용합니다.
모델 가중치를 재학습하는 기능은 아닙니다. 우세한 성능은 동일 조건 비교 실험으로 별도 확인해야 합니다.

## 확인

- `brain/strategy.jsonl`: 계획 요청·검증·적용, Jev 요청·채택, 지연/오류, 리플레이 기억
- `brain/opponent-memory.json`: 검증된 상대별 경기 경험
- 스파링에서는 위 파일이 `brain-0`, `brain-1`에 각각 저장됨
- 기존 경기별 행동 로그와 서버 리플레이도 그대로 저장됨

Jev 공식 SDK: https://github.com/typesafe-ai/typesafe-sdk-python
MLX 공식 사용법: https://github.com/ml-explore/mlx-lm

## 스파링 실행 안정성 및 점검

- 최초 로그인은 최대 3회만 재시도하고, 준비 단계의 DuelState 응답은 최대 30초 기다립니다.
- 전투 상태가 3초 이상 갱신되지 않으면 행동을 멈춥니다. 잠깐 지연됐다는 이유로
  같은 라운드의 에이전트·시전 절차·전술 기억을 재생성하지 않습니다.
- 15초 넘게 전투 상태가 없으면 실행을 중단합니다. 통신 장애로 끝난 실행은
  임의로 승패를 만들거나 학습 결과에 추가하지 않습니다.
- 실행 현황은 `status.json`, 완료된 경기와 모델 판단은 다음 명령으로 확인합니다.

```sh
python -m scripts.review_sparring .logs/improve-live
```

`review.json`에는 SHA256 검증된 경기의 피해·회복·시전·실패·포션 기록,
라운드 내 에이전트 초기화 횟수, 모델 오류·지연 횟수가 저장됩니다.
자기대련 몇 경기의 승패만으로 성능 향상을 확정하지 않습니다.
