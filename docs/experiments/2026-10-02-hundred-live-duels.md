# Arena 100판 실전 학습

## 고정된 실험 조건

- 작업 브랜치: `feat/arena-100-duels`, 기반 `3ae14e7`.
- 시작 모델: `models/duel-continuation-live`.
- 시작 SHA: `fe5796b401fe276cb6c916d5c613e8dc286dcaa0432b136d5aee3465f2803a38`.
- 실제 Arena 서버에서 1라운드 경기 100회, 7x + 폭발 포션, 비랭크 training.
- 학습 선수: `arena_train_a`, 고정 scripted 상대: `arena_train_b`.
- 도전자 순서 교대, 매 경기 재접속 및 공개 명령으로 보급/스탯/스킬 확인.
- 서버 리플레이 및 on-policy 기록 검증 후 매 경기 PPO 업데이트.
- 공개 상대 정책의 자동 교체는 하지 않음.

100판 한도는 실제 Arena 경기에 해당한다. 모두 끝난 후 시작 모델과 최종
모델을 각각 같은 새 시드의 근사 시뮬레이터 200경기로 평가한다.
이 평가는 서버에 추가 경기를 만들지 않으며 실제 사용자 상대 승률이 아니다.
경기 수집/학습 실패 시 멈춘다. 중단된 실제 경기를 중복 계산하지 않도록
자동 재시작과 기존 로그 폴더 재사용을 거부한다.

## 실행 위치

미국 Chicago의 기존 서버에서 실행한다. shard 재시작이나 배포 변경은 없다.
기존 신경망 패키지의 `live.py`, `online.py`, `executor.py`, `train.py`와
worktree 소스의 SHA-256 일치를 확인했다.

- 실행기: `scripts/run_hundred_live_duels.py`
- 원격 설치: `/opt/uoarena/anima3-neural-lab-20260930/run_hundred_live_duels.py`
- 서비스: `arena-neural-100-20261002.service`
- 원격 결과: `/opt/uoarena/anima3-neural-lab-20260930/.logs/hundred-live-20261002`
- CPU 상한 150%, 메모리 상한 3 GiB, 우선순위 nice 5.
- 100경기 완료 또는 오류/중단 시 종료. 추가 상한 12시간, 자동 재시작 없음.
- 자격 증명은 기존 서버의 보호된 `.training-accounts.json`에서만 읽는다.

## 진행 확인과 중단

```sh
ssh -i ~/.ssh/uoarena_linode root@172.234.206.216 \
  'systemctl show arena-neural-100-20261002 -p ActiveState -p SubState -p Result'

ssh -i ~/.ssh/uoarena_linode root@172.234.206.216 \
  'journalctl -u arena-neural-100-20261002 -n 10 --no-pager'

# 필요할 때 해당 실험만 중단
ssh -i ~/.ssh/uoarena_linode root@172.234.206.216 \
  'systemctl stop arena-neural-100-20261002'
```

## 증거 파일

- `experiment.json`: 목표, 초기 SHA, 조건, 평가 시드.
- `cycle.json`: 현재 회차, 완료한 업데이트, 후보 모델 경로.
- `round-NNN/collection/status.json`: 로그인/준비/경기 단계.
- `round-NNN/collection/results.json`: 검증된 경기 결과와 리플레이 링크.
- `round-NNN/collection/<match-id>/`: 원본 리플레이, 관측/행동, 학습 적격성.
- `round-NNN/candidate/`: 업데이트한 가중치, optimizer/RNG, 부모 SHA.
- `report.json`, `report.md`: 실행 종료 시 전체 경기 목록과 승패/업데이트 수.
- `evaluation-before.json`, `evaluation-after.json`: 완료 후 동일 조건 비교.
- `finished.json`: 100판 및 전후 평가가 모두 완료됐다는 기록.

실행을 시작했다는 사실과 100판 완료/학습 성능 향상은 별도로 확인한다.
일부 경기의 승패나 가중치 변경만으로 실력이 향상됐다고 판단하지 않는다.

## 2026-10-03 새 100경기 실행

첫 작업은 5경기/5회 업데이트 후 다음 연결의 Idle 확인 시간 초과로 중단됐다.
이후 별도 1경기와 학습을 완료했다. 사용자의 새 100판 요청에 따라 그 최신
후보 `4fd33a5764243f8ce417e5da03a2aca36c8a05f9bb4ac22e691f7769ae49c61c`
에서 **새 100경기**를 순차 실행한다. 이전 6경기는 새 목표에 포함하지 않는다.

- 서비스: `arena-neural-100-20261003.service`
- 결과: `/opt/uoarena/anima3-neural-lab-20260930/.logs/hundred-live-20261003`
- 시작 모델: `.logs/single-live-20261003-01/round-001/candidate`
- 준비 단계에서 각 계정의 상태·마지막 응답 나이를 10초 간격으로 기록한다.
- 준비 중 30초 넘게 상태 응답이 없으면 접속을 종료하고 최대 3회까지 시도한다.
  이미 시작된 경기나 다른 상대와의 경기는 재시도 대상이 아니다.
- 실패한 준비 기록은 해당 회차 `idle-retry-N`에 보존한다. 준비 재시도로
  경기 수나 도전자 순서, 사용 가중치가 바뀌지 않는다.
- 복구/제한/기존 경기 수집 관련 테스트 9개 통과.

위 운영 명령에서 서비스와 결과 경로를 새 이름으로 바꾸어 조회/중단한다.
완료 여부는 이 실행의 `cycle.json`, `report.json`, `finished.json`으로 확인한다.

### 첫 경기의 수집 실패와 99판 후속 작업

새 실행의 첫 경기 `c6b91576e1f1455799e556b304632eae`는 서버에서 완료됐으나
학습 선수의 상태 수신이 끊겨 정책 궤적을 수집하지 못했다. 서버 리플레이를
별도로 검증·보존했으며 학습에서는 제외한다. 이 경기를 실제 경기 수에
포함하고 이후 작업을 **99판**으로 제한한다. 기존 5판+별도 1판과는 구별한다.

- 원본 폴더의 `budget.json`에 제외 사유와 후속 폴더를 기록했다.
- 후속 서비스: `arena-neural-99-20261003.service`
- 후속 결과: `.logs/hundred-live-20261003-continuation`
- 후속 시작 가중치는 그대로 `4fd33a576424...`; 도전자 순서는 1부터 이어간다.
- 최신 bridge `4100ecb`를 `/opt/uoarena/anima-client-neural-4100ecb`에 별도
  빌드했다. 기존 bridge나 게임 서버 바이너리는 교체하지 않았다.
- 경기를 배정받지 못한 클라이언트의 관측도 `unassigned-actor-N.jsonl`에
  즉시 기록한다. 최신 bridge는 전송·수신·디코더 상태를 함께 제공한다.
- 변경 후 수집/업데이트/복구 관련 테스트 27개 통과.

총 실제 경기 수는 검증된 첫 경기 1개와 후속 작업 완료 경기 수의 합이다.
후속 서비스의 완료 및 학습 횟수는 후속 폴더에서 확인해야 한다.
