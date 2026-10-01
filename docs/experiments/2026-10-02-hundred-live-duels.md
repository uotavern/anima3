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
