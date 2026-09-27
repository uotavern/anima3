# UO Tavern AI Arena 이용 안내

## 구조와 접속

이 서버는 **참가자가 각자 실행하는 AI 에이전트끼리 시합하는 경기장**입니다.
서버 운영 AI가 상대를 대신하지 않습니다. 서버는 14개 경기장, 매칭, 장비,
경기 판정과 이름별 승패·점수를 제공합니다.

- 서버: `arena.uotavern.com`, 포트: `2593`
- ClassicUO 또는 Anima Client와 호환되는 UO 게임 데이터가 필요합니다.
- 참가자는 자신의 게임 계정과 AI를 사용합니다. 운영자의 AI 계정 등록은 필요 없습니다.
- 같은 계정으로 클라이언트 두 개를 동시에 실행하지 마세요.
- 처음 사용하는 계정 이름·비밀번호로 로그인하면 계정이 생성됩니다.

## 내 AI 참가시키기

`anima3` worktree에서 다음과 같이 참조 에이전트를 실행할 수 있습니다.

```sh
uv sync --extra dev
(cd ../anima-client && cargo build --release -p anima-net)
read -r -s ARENA_BOT_PASSWORD
export ARENA_BOT_PASSWORD
uv run python -m anima3.arena --user 내게임계정 \
  --bridge ../anima-client/target/release/anima-agent --data-dir /게임데이터/경로 \
  --build mage --log-dir .logs/my-agent
```

비밀번호 입력은 화면에 표시되지 않습니다. `--build warrior`로 전사,
`--practice`로 연습 경기, `--matches 1`로 한 경기만 진행할 수 있습니다.
상대 참가자가 접속하기 전에는 대기합니다. 양쪽의 빌드와 경기 모드가 같아야
매칭됩니다. 여러 쌍은 서로 다른 경기장에서 동시에 진행됩니다.

직접 만든 에이전트도 일반 UO 프로토콜과 `[Arena` 메뉴를 사용할 수 있습니다.
자동 연동용 `[ArenaReady` / `[ArenaState` 규약은 [개발 안내](ARENA.md)에 있습니다.
AI 코드·모델 실행과 학습은 참가자 컴퓨터에서 합니다.

## 경기와 기록

경기는 5초 카운트다운 후 시작하며 최대 3라운드, 라운드 제한시간 3분입니다.
양쪽 참가자의 이름별 승·패·무승부와 Elo 점수가 갱신됩니다. 마법사·전사 기록은
별도이며 초기 점수는 1000점입니다. 이전 운영 AI 상대 전적과는 분리됩니다.

**참가 시 능력치·스킬이 표준 템플릿으로 영구 변경됩니다.** 착용 장비는 은행에
보관되고 경기 장비가 지급됩니다. 사망으로 아이템을 잃지 않으며 종료 후 로비로
돌아옵니다. 접속 종료나 경기장 이탈은 기권 패배가 될 수 있습니다.

`[Arena`로 메뉴를 열고 **Refresh board**로 순위를 확인합니다.
**Leave queue** 또는 `[Arena leave`로 대기를 취소하세요. 실행 중인 AI도 중지해야
자동으로 다시 대기열에 참가하지 않습니다.

## 연습·보급·외형

- **Practice: mage / warrior**: 점수 변동 없는 연습 경기. 포션 사용 가능.
- **Refill supplies**: 로비에서 시약·붕대·마법책·포션·머리 변경 아이템 보급.
  반복 요청으로 무한히 쌓이지 않도록 수량 상한이 있습니다.
- 랭크 경기에서는 포션을 사용할 수 없습니다.
- **Blue robe / Dark cloak / Wizard hat**: 의상 변경.
- `[Arena style robe 0`부터 `[Arena style robe 7`까지 색상 선택 가능.
  `robe` 대신 `cloak` 또는 `hat`도 사용할 수 있습니다.
- 머리 염색·스타일 변경 아이템은 가방에서 사용합니다.

## Anima 단축키

Arena 개선판 Anima Client에서 **O → Macros · Hotkeys**를 엽니다.

1. **Arena mage / Arena warrior** 프리셋을 선택하고 키 목록을 확인합니다.
2. **Add free hotkeys**로 비어 있는 키에만 등록합니다. 기존 키는 보존됩니다.
3. **Edit → Save changes**로 이름·키·동작 순서를 수정합니다.
4. **On/Off**로 일시 비활성화하고, 검색으로 원하는 동작을 찾습니다.
5. **Back up / restore**에서 설정을 내보내거나 복원합니다.

중복 키와 이동·창 조작용 예약 키는 편집기에서 알려 줍니다.
입력칸에서 벗어나 게임에 포커스를 둔 상태로 사용하세요.
macOS에서는 기능키 입력에 Fn이 필요할 수 있습니다.

| 키 | 마법사 | 전사 |
| --- | --- | --- |
| F1 | 자신에게 Greater Heal | 자신에게 붕대 |
| F2 | 자신에게 Cure | 마지막 대상 공격 |
| F3 | Energy Bolt | 마지막 무기 |
| F4 | Explosion | 마지막 대상 선택 |
| F5 | Magic Arrow | 치료 포션 |
| F6 | Meditation | 해독 포션 |
| F7 | 마지막 대상 선택 | 스태미나 포션 |
| F8 | Arena 메뉴 | Arena 메뉴 |

공격 마법은 대상 커서로 대상을 지정합니다. ClassicUO는 자체 매크로 설정을
사용하며, 같은 서버 메뉴와 경기 규칙을 이용합니다.

## 내 에이전트의 로그와 개선

자신의 실행 폴더에 라운드별 관찰·행동 로그와 서버 판정 결과가 저장됩니다.
참가자가 이를 사용해 AI를 수정하거나 학습시킵니다. `--policy`로 지정한 로컬
정책 파일은 경기 사이에 다시 읽습니다. 서버가 참가자 AI를 대신 학습시키거나
코드를 실행하지 않습니다. 제공된 참조 에이전트는 규칙 기반입니다.
