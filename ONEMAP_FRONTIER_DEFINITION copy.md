# Onemap에서의 프런티어 정의

## 개요

Onemap에서 프런티어는 **고전적인 프런티어 정의와 다르게** `confidence_map`을 활용하여 정의됩니다. 단순히 "탐색된 영역과 미탐색 영역의 경계"가 아니라, **"충분히 탐색된 영역과 관측은 있지만 충분하지 않은 영역의 경계"**로 정의됩니다.

---

## 핵심 개념

### 1. Confidence Map

`confidence_map`은 각 셀에 대해 **얼마나 많이 관측되었는지**를 나타내는 값입니다.

- **높은 confidence**: 해당 셀을 여러 번 관측함 (충분히 탐색됨)
- **낮은 confidence**: 해당 셀을 적게 관측함 (아직 탐색 부족)
- **confidence = 0**: 해당 셀을 한 번도 관측하지 않음 (미탐색)

### 2. Fully Explored Map

`fully_explored_map`은 `confidence_map`의 역수를 사용하여 계산됩니다:

```python
# mapping/feature_map.py:314-315
self.fully_explored_map = (np.nan_to_num(1.0 / self.confidence_map.cpu().numpy())
                           < self.fully_explored_threshold)
```

**의미**:
- `1.0 / confidence_map`이 작을수록 → confidence가 높음 → 충분히 탐색됨
- `fully_explored_threshold`보다 작으면 → `fully_explored = True`

**예시**:
- `confidence = 10.0` → `1.0 / 10.0 = 0.1` → `fully_explored_threshold = 0.2`보다 작음 → **fully explored**
- `confidence = 2.0` → `1.0 / 2.0 = 0.5` → `fully_explored_threshold = 0.2`보다 큼 → **not fully explored**
- `confidence = 0.0` → `1.0 / 0.0 = inf` → **not fully explored**

### 3. 프런티어 정의

**Onemap의 프런티어**: 
> "at the border from fully explored to confidence > 0"

즉, **충분히 탐색된 영역(`fully_explored_map`)과 관측은 있지만 충분하지 않은 영역(`confidence > 0` but not fully explored)의 경계**입니다.

---

## 고전적인 프런티어 정의와의 차이

### 고전적인 프런티어 (예: VLFM)
```
탐색된 영역 (explored) ↔ 미탐색 영역 (unexplored)
```
- **이진 분류**: 탐색됨 vs 미탐색
- **단순한 경계**: 한 번이라도 관측되면 "탐색됨"

### Onemap의 프런티어
```
충분히 탐색된 영역 (fully_explored) ↔ 관측은 있지만 충분하지 않은 영역 (confidence > 0)
```
- **3단계 분류**: 
  1. 미탐색 (`confidence = 0`)
  2. 관측은 있지만 충분하지 않음 (`0 < confidence < threshold`)
  3. 충분히 탐색됨 (`fully_explored`)
- **세밀한 경계**: 충분히 탐색된 영역과 부분적으로 탐색된 영역의 경계

---

## 프런티어 추출 과정

### Step 1: 프런티어 경계면 감지

**위치**: `mapping/navigator.py:407-422`

```python
def compute_frontiers_and_POIs(self, px, py):
    """
    Computes the frontiers (at the border from fully explored to confidence > 0)
    """
    # 프런티어 경계면 감지
    frontiers, unexplored_map, largest_contour = detect_frontiers(
        self.one_map.navigable_map.astype(np.uint8),      # 탐색 가능한 전체 맵
        self.one_map.fully_explored_map.astype(np.uint8), # 충분히 탐색된 영역
        self.one_map.confidence_map > 0,                  # 관측이 있는 영역 (미탐색 제외)
        area_thresh
    )
```

**입력**:
- `navigable_map`: 탐색 가능한 전체 영역
- `fully_explored_map`: 충분히 탐색된 영역 (explored_mask 역할)
- `confidence_map > 0`: 관측이 있는 영역 (known_th 역할)

**과정** (`mapping/nav_goals/frontier.py:177-235`):
1. `fully_explored_map`의 컨투어 찾기
2. 미탐색 영역 마스크 생성: `unexplored_mask = navigable_map - fully_explored_map`
3. `unexplored_mask`에서 `confidence > 0`인 부분만 프런티어로 선택
4. Bresenham 알고리즘으로 컨투어 보간
5. 미탐색 영역과 접하는 부분만 필터링

### Step 2: 프런티어 스코어링

**위치**: `mapping/navigator.py:454-466`

각 프런티어에 대해 **도달 가능한 영역 내의 최대 similarity score**를 계산합니다.

```python
for i_frontier, frontier in enumerate(frontiers):
    frontier_mp = get_frontier_midpoint(frontier).astype(np.uint32)
    
    # 프런티어 중점에서 BFS로 도달 가능한 영역 탐색
    score, n_els, best_reachable, reachable_area = Planning.compute_reachable_area_score(
        frontier_mp,                                    # 프런티어 중점
        (self.one_map.confidence_map > 0).cpu().numpy(), # 도달 가능한 영역 (관측이 있는 영역)
        adjusted_score_frontier,                        # similarity score 맵
        self.frontier_depth                            # 최대 탐색 깊이
    )
    
    # 프런티어 객체 생성
    self.nav_goals.append(
        Frontier(frontier_midpoint=frontier_mp, 
                points=frontier, 
                frontier_score=score)
    )
```

**스코어링 알고리즘** (`planning/planning_utils.py:27-41`):
1. 프런티어 중점에서 시작
2. BFS로 `confidence > 0`인 영역 내에서 탐색
3. `frontier_depth` 깊이까지 탐색
4. 도달 가능한 영역 내의 **최대 similarity score**를 프런티어 스코어로 사용

**의미**:
- 프런티어 뒤에 있는 영역에서 **목표 객체와 유사한 영역**이 얼마나 있는지 평가
- 높은 스코어 = 해당 프런티어를 통해 목표 객체를 찾을 가능성이 높음

### Step 3: 프런티어 정렬

```python
self.nav_goals = sorted(self.nav_goals, key=lambda x: x.get_score(), reverse=True)
```

스코어가 높은 순서대로 정렬하여, 가장 유망한 프런티어를 우선적으로 탐색합니다.

---

## 시각화 예시

```
전체 맵 (navigable_map):
████████████████████████
██░░░░░░░░░░░░░░░░░░░░██  █ = 장애물
██░░░░░░░░░░░░░░░░░░░░██  ░ = 탐색 가능
████████████████████████

Confidence Map:
████████████████████████
██░░░░░░░░░░░░░░░░░░░░██
██░░░░░░▒▒▒▒▒▒░░░░░░░░██  숫자 = confidence 값
██░░░░░░▒▒▒▒▒▒░░░░░░░░██  높을수록 많이 관측됨
██░░░░░░▒▒▒▒▒▒░░░░░░░░██
████████████████████████

Fully Explored Map (confidence 역수 < 0.2):
████████████████████████
██░░░░░░░░░░░░░░░░░░░░██
██░░░░░░░░░░░░░░░░░░░░██  ▒ = fully explored
██░░░░░░░░▒▒░░░░░░░░░░██  (confidence가 충분히 높음)
██░░░░░░░░░░░░░░░░░░░░██
████████████████████████

프런티어 (경계):
████████████████████████
██░░░░░░░░░░░░░░░░░░░░██
██░░░░░░░░░░░░░░░░░░░░██  ─ = 프런티어 경계
██░░░░░░░░▒─░░░░░░░░░░██  (fully explored ↔ confidence > 0)
██░░░░░░░░░░░░░░░░░░░░██
████████████████████████
```

---

## 코드 흐름 요약

```
Navigator.compute_frontiers_and_POIs()
  │
  ├─> detect_frontiers(
  │     navigable_map,           # 전체 탐색 가능 영역
  │     fully_explored_map,      # 충분히 탐색된 영역
  │     confidence_map > 0,      # 관측이 있는 영역
  │     area_thresh
  │   )
  │   │
  │   ├─> fully_explored_map의 컨투어 찾기
  │   ├─> 미탐색 영역 = navigable_map - fully_explored_map
  │   ├─> 미탐색 영역 중 confidence > 0인 부분만 프런티어로 선택
  │   └─> Bresenham 알고리즘으로 보간
  │
  ├─> 각 프런티어에 대해:
  │   │
  │   ├─> get_frontier_midpoint() - 프런티어 중점 계산
  │   │
  │   └─> compute_reachable_area_score(
  │         frontier_mp,              # 프런티어 중점
  │         confidence_map > 0,        # 도달 가능한 영역
  │         adjusted_score,           # similarity score 맵
  │         frontier_depth            # 최대 탐색 깊이
  │       )
  │       │
  │       └─> BFS로 도달 가능한 영역 탐색
  │           └─> 최대 similarity score 반환
  │
  └─> 프런티어를 스코어 순으로 정렬
```

---

## 주요 차이점 정리

| 항목 | 고전적인 프런티어 (VLFM) | Onemap 프런티어 |
|------|------------------------|-----------------|
| **정의** | 탐색됨 ↔ 미탐색 | 충분히 탐색됨 ↔ 관측은 있지만 충분하지 않음 |
| **분류** | 이진 (탐색됨/미탐색) | 3단계 (미탐색/부분 탐색/충분히 탐색) |
| **기준** | 한 번이라도 관측되면 탐색됨 | 충분히 관측되어야 탐색됨 |
| **활용 맵** | explored_map (이진) | confidence_map (연속값) |
| **스코어링** | 거리 기반 또는 단순 | 도달 가능한 영역의 similarity score 기반 |

---

## 핵심 포인트

1. **Confidence 기반 탐색**: 단순히 "봤다/안 봤다"가 아니라 "얼마나 많이 봤는가"를 고려

2. **세밀한 경계**: 충분히 탐색된 영역과 부분적으로 탐색된 영역의 경계를 찾음

3. **Semantic-aware 스코어링**: 프런티어 뒤에 있는 영역의 similarity score를 고려하여 목표 객체를 찾을 가능성이 높은 프런티어를 우선 선택

4. **점진적 탐색**: 한 번 관측된 영역도 충분히 탐색되지 않았다면 프런티어로 유지하여 더 자세히 탐색

---

## 참고 파일

1. **mapping/navigator.py:407-468**: 프런티어 계산 및 스코어링
2. **mapping/nav_goals/frontier.py:177-235**: 프런티어 경계면 감지
3. **mapping/feature_map.py:314-315**: fully_explored_map 계산
4. **planning/planning_utils.py:27-41**: 프런티어 스코어링 알고리즘

---

## 설정 파라미터

- **fully_explored_threshold**: `confidence_map`의 역수가 이 값보다 작으면 fully explored로 간주 (기본값: 0.2-0.3)
- **frontier_depth**: 프런티어 스코어링 시 탐색할 최대 깊이
- **area_thresh**: 최소 미탐색 영역 크기 (픽셀 단위)

