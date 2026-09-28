# Claude Code 요청: Arena 웹사이트에 에이전트 참가 안내 추가

요청일: 2026-09-28. 사용자 요청을 전달하는 Codex 작성 지시서.
기존 웹사이트 작업 흐름과 디자인에 맞춰 구현하고 실제 페이지를 확인해주세요.
이 문서 자체는 공개용 원고와 작업자용 근거를 함께 담고 있으므로 통째로 게시하지 마세요.

## 목적 / 페이지 구성

arena.uotavern.com의 랜딩 페이지에서 쉽게 찾을 수 있는 **Run your agent / 에이전트 참가 안내**를 추가해주세요.
기존 duel 성적 웹페이지와 상호 링크하고, 사람이 AI와 대결하는 안내와 개발자가 자신의 에이전트를 연결하는 안내를 구분해주세요.
미국 이용자 중심이므로 영어를 기본으로 하고 사이트의 기존 다국어 체계가 있으면 한국어도 제공합니다.
실제 arena 랜딩 소스와 uotavern 웹 소스 위치를 먼저 확인하고 기존 배포 방식을 사용하세요.
현재 다른 작업의 수정/배포와 충돌하지 않도록 웹사이트 범위로 진행해주세요.

### 1. 첫 화면 설명 (영어 초안)

**Bring your agent to the arena.**
Run your own Ultima Online agent and challenge human players or other agents.
UO Tavern provides the arenas, supplies and match referee. Your agent runs on your computer.
Connect to `arena.uotavern.com:2593` with an ordinary game account.

CTA: “Run an agent” / “Challenge an agent” / “View results”.
서버는 여러 경기장과 판정/결과를 제공하고 참가자가 AI를 실행합니다.
서버가 항상 AI 상대를 자동 생성한다거나 서버에서 중앙 강화학습을 수행한다고 설명하지 마세요.

### 2. 가장 먼저 소개할 모드

**Standard 7x + Explosion potions** (`standard7-explosion`).
현재 참조 에이전트는 이 정확한 규칙의 직접 도전을 기다렸다가 자동 수락합니다.
다른 규칙의 도전은 수락하지 않습니다. 모든 5x/7x preset을 이 에이전트가 지원한다고 쓰지 마세요.

자동 준비: 각 100의 Magery, Evaluating Intelligence, Meditation, Resisting Spells,
Wrestling, Anatomy, Alchemy. STR/DEX/INT = 100/25/100.
공개 skill/stat ball과 보급 명령을 사용하며 **캐릭터의 기존 스킬/스탯을 변경**합니다.
별도의 일반 게임 계정을 권장합니다. GM 권한이나 운영자 bot allowlist가 필요하지 않습니다.
시약, 마법책, 포션을 준비하며 전투 중 마법과 폭발 포션을 사용합니다.
현재 참조 전투 정책은 scripted baseline입니다. LLM이나 실시간 강화학습을 필수로 요구하지 않습니다.

### 3. 실행 가이드

확인된 소스: `/Users/dkkang/dev/uo/arena-worktrees/anima3`의 `feat/arena-service` 브랜치.
`anima3/duel_wait.py`는 일반 메인 브랜치에 있다고 가정하지 마세요.
공개 설치 링크를 넣기 전에 원격 공개 여부/실제 파일 존재를 확인하고, 없는 릴리스 다운로드나 원클릭 실행을 만들지 마세요.
Python 프로젝트 설치, Rust anima-agent bridge 빌드, 사용자가 보유한 UO 데이터 경로가 필요합니다.
기존 `docs/ARENA.md`의 설치 단계를 참고하되 아래 실행은 **직접 도전 대기 모드**입니다.

실행 예 (필요한 버전 설치 후, anima3 디렉터리에서):

```sh
uv run python -m anima3.duel_wait \
  --host arena.uotavern.com --port 2593 \
  --user YOUR_GAME_ACCOUNT \
  --bridge /path/to/anima-agent \
  --data-dir /path/to/uo-data \
  --log-dir .logs/my-duel-agent
```

실행 전에 `ARENA_BOT_PASSWORD` 환경변수에 자신의 게임 비밀번호를 안전하게 설정하도록 안내합니다.
평문 비밀번호를 명령 예제, URL, 웹 폼, 로그, 스크린샷에 넣지 마세요.
웹사이트에서 게임 비밀번호를 수집할 필요가 없습니다.
같은 게임 계정으로 ClassicUO와 에이전트를 동시에 실행하지 않도록 설명합니다.
콘솔 `ready` 이벤트의 캐릭터 이름/serial/rules를 확인하면 준비 완료입니다.
에이전트를 실행한 컴퓨터와 프로세스를 켜두어야 도전 수락이 가능합니다. 종료는 Ctrl+C.
접속이 끊기면 재접속을 시도하고 경기 종료 후 다시 준비합니다.

### 4. ClassicUO / Anima Client로 도전하기

게임 주소: `arena.uotavern.com`, 포트 `2593`.
로비 이동: `[Arena enter`.
NPC Rowan / 안내 표지에서 스킬볼·스탯볼·보급품 안내를 확인할 수 있습니다.
`[Arena duel` 메뉴에서 7x Standard를 선택하고 Explosion 옵션을 켠 뒤 상대를 지정합니다.
명령 방식: `[Challenge <opponent-serial> 3 standard7-explosion`.
여기서 3은 best-of-three (2선승)입니다. 상대는 조건이 맞으면 자동 수락합니다.
실제 상대 serial은 대기 중인 에이전트 정보를 사용합니다.
`Tavern Mage 3 / 0x3`는 2026-09-27에 확인한 테스트 캐릭터일 뿐, 항상 온라인인 상대처럼 고정 노출하지 마세요.
현재 피드가 단순 캐릭터 온라인 상태만 제공한다면 그것을 '에이전트 준비 완료'로 해석하지 마세요.

### 5. 결과 / 로그 / 학습

승패와 결과는 서버가 판정합니다. 클라이언트가 점수나 승리를 제출하지 않습니다.
직접 7x 도전 결과는 일반 duel history/standings에 표시됩니다.
별도 mage/warrior ranked queue의 Elo와 동일한 점수 체계라고 설명하지 마세요.
`anima3.arena`는 별도 자동 매칭 큐 진입점이고 `anima3.duel_wait`와 사용법을 섞지 마세요.
참가자는 로컬 `client.jsonl` 및 경기/라운드별 관측·행동 JSONL로 자신의 정책을 분석/개선할 수 있습니다.
자동 학습/새 모델 승격/상시 자가 대전 서비스가 공개 서버에서 동작한다고 광고하지 마세요.

### 6. 개발자용 접이식 상세 안내

에이전트도 보통 플레이어 클라이언트처럼 로그인하여 UO 프로토콜로 행동합니다.
`[DuelState`는 요청한 캐릭터 자신의 경기 또는 받은 도전을 서버 시스템 저널 JSON으로 반환합니다.
자기 도전의 규칙과 token을 확인한 후 `[DuelAccept <challenge-id>`로 수락합니다.
다른 도전으로 교체되거나 만료된 token은 수락되지 않습니다.
참조 구현은 서버 메시지 serial 검증 및 상태 검증을 수행합니다. 다른 플레이어의 채팅을 명령으로 신뢰하지 않습니다.
정식 서버 판정이 상대/라운드/진행 여부를 결정하며 클라이언트는 정상적인 관측과 행동만 사용합니다.
이 상세 내용은 초보자의 기본 참가 흐름을 방해하지 않게 별도 영역에 둡니다.

## 구현 완료 확인

- 랜딩에서 agent 안내와 duel 결과로 이동 가능, 모바일/데스크톱 모두 읽기 쉬움.
- 복사 버튼은 실제 명령을 복사하며 placeholder를 명확하게 표시.
- 실행 안내가 제공되는 실제 소스 버전과 일치, 다운로드/저장소 링크가 실제로 열림.
- '온라인', '대기 중', '경기 중'은 근거가 있는 경우만 표시; 피드 장애와 0명을 구분.
- 비밀키/운영 계정/로컬 절대경로/배포 백업/PID는 공개 페이지에 노출하지 않음.
- 사이트 빌드 및 실제 렌더링 확인 후 변경 파일과 게시 URL을 보고.

## 작업자용 근거 (게시 금지)

- anima3: `/Users/dkkang/dev/uo/arena-worktrees/anima3/anima3/duel_wait.py`
- explosion potion policy: 위 디렉터리의 `anima3/magic.py`
- public queue: `anima3/arena.py`, `docs/ARENA.md`
- 상세 인수인계: `docs/ARENA_PEER_HANDOFF.md`
- ServUO: `/Users/dkkang/dev/uo/arena-worktrees/servuo/Scripts/Services/Dueling/`
- 2026-09-27 배포: ServUO `55ad7d126` (Hybrid `31fbb2deb` + skill filter `97bee0afd` 포함).
- 로컬 실경기: 준비 → 잘못된 규칙 거절 → 정확한 규칙 수락 → 포션/마법 → 서버 승패 판정 → 재준비 확인.
- 공개 서버에서는 당시 bot ready 및 skill UI 확인. 오늘의 상시 온라인 여부는 별도 조회가 필요함.

## 2026-09-28 추가 요청: Showdown

서버 추가 기능: 모든 duel의 각 라운드 FIGHT 시점부터 180초 후 Showdown.
30초 전 예고하며 Heal/Greater Heal, 붕대, 회복 포션, 자연 HP 재생을 금지합니다.
해독 및 마나/스태미나 회복은 허용. 300초까지 미결이면 해당 라운드는 무승부이며
다음 라운드에서 초기화됩니다. 이전 3분 무승부 설명을 3분 Showdown / 5분 무승부로 바꿔주세요.
피드 top-level `showdownAfterSeconds`, `roundLimitSeconds`, 각 live 경기의 `showdown`(bool),
`showdownRemaining`(seconds), `roundLimitSeconds`가 추가됩니다.
`phase`는 계속 fighting이며 Showdown을 새 phase로 가정하지 마세요.
이 필드가 없는 구버전 피드에는 '정상/쇼다운 아님'을 추정하지 말고 해당 표시를 생략하세요.
배포 완료 여부는 `ARENA_PEER_HANDOFF.md`의 후속 기록으로 확인합니다.

## 2026-09-28 추가: 실제 경기 리플레이와 스파링

`docs/ARENA_REPLAY.md`에 서버 API/NDJSON 스키마/보관 정책/학습 상태 표시 계약을 작성했습니다.
UO 화면 재생기는 anima-client의 `?replay=<id>` 모드입니다.
`/duel/replays/` 목록에서 `visualVersion:1`인 완료 경기를 `/replay/?replay=<id>`로
연결해주세요. 경기 목록·설명은 기존 웹사이트 디자인을 따르고, 경기 화면은 이 UO 렌더러를 사용하세요.
훈련 경기는 Training 표시, 일반 경기 성적과 분리. 시전 시도와 실제 피해를 혼동하지 마세요.
사용자가 승인한 스파링/학습 기능의 local `status.json`은 아직 공개 HTTP 피드가 아니므로
데이터 연결 없이 가짜 학습 화면이나 자동 개선 문구를 만들지 마세요.

## Required replay presentation (user correction, 2026-09-28)

Use **actual UO graphics**, reusing anima-client's read-only `?replay=<id>` mode.
Skin hue, hair/beard, dyed clothes, held weapons, walk/attack/cast motions and spell
effects must be visible. No schematic dots/grid and no video capture.
Embed `/replay/?replay=<id>` in an iframe or open it as a full page; the isolated
anima-client worktree owns the renderer and asset endpoints. The public replay
index remains `/duel/replays/`; filter `visualVersion == 1` and `complete == true`.
The branded page can supply match title/results/navigation around that renderer.
See ARENA_REPLAY.md for the current event/data contract.
