# Standard-Aware Prompt Templates (Design Document)

본 문서는 LLMRec의 LLM 증강 프롬프트(`P_I`, `P_U`, `P_UI`)를 schema.org/Movie 어휘 및 ISO 표준 코드 체계와 정합하도록 재설계한 결과를 정의한다. **본 텀페이퍼에서는 LLM 재호출을 수행하지 않고**, 본 템플릿을 정성적 분석과 졸업논문 후속 연구의 기반 자산으로 제시한다.

---

## 1. 기존 LLMRec 프롬프트의 표준학적 진단

### 1.1 Item 속성 프롬프트 (`P_I`, [gpt_i_attribute_generate_aug.py:40-57](../../LLMRec/LLM_augmentation_construct_prompt/gpt_i_attribute_generate_aug.py))

```text
You are now a search engines, and required to provide the inquired
information of the given movies bellow:
[332] 1993, Heart and Souls
The inquired information is : director, country, language.
And please output them in form of:
director::country::language
please output only the content in the form above ...
```

**문제점**:
- 출력 어휘에 표준 미부여 → "USA" / "United States" / "America" 자유 분산
- 컬럼 시프트 발생 시 (예: 빈 director) 후속 필드가 밀려 country 자리에 'language' 토큰이 들어가는 등 파싱 오류 (실측: 835건)
- contentRating, datePublished, duration 등 schema.org/Movie 필수 필드 결락

### 1.2 User 프로파일 프롬프트 (`P_U`)

```text
... output format: {age: , gender: , liked genre: , disliked genre: ,
liked directors: , country: , language: }
```

동일하게 country/language 자유 텍스트 출력, age/gender 단위 미정의.

---

## 2. Standard-Aware Item Prompt (`P_I*`)

### 2.1 설계 원칙
1. **Context 명시**: schema.org/Movie JSON-LD 컨텍스트 강제
2. **Coded Vocabulary**: countryOfOrigin → ISO 3166-1 alpha-2, inLanguage → BCP 47, contentRating → MPAA 표준 등급
3. **Failure Mode 명시**: 미상 시 `null` 반환을 강제하여 유사 토큰 환각 차단
4. **단일 응답 단위**: JSON 객체 단위 출력으로 컬럼 시프트 원천 차단

### 2.2 시스템 프롬프트
```text
You are a metadata extraction assistant strictly conforming to
schema.org/Movie. Output one JSON-LD object per movie with fields
encoded as machine-readable codes:
  - countryOfOrigin: ISO 3166-1 alpha-2 (e.g., "US", "JP", "FR")
  - inLanguage: BCP 47 short tag (e.g., "en", "ja", "fr")
  - genre: schema.org genre vocabulary or EIDR genre code
  - contentRating: MPAA standard ("G","PG","PG-13","R","NC-17","NR")
  - datePublished: ISO 8601 year (YYYY)
  - duration: ISO 8601 duration (e.g., "PT102M")
  - director: schema.org Person.name (free text)
If a field cannot be determined with confidence, output null.
Do NOT invent values. Output ONLY valid JSON-LD.
```

### 2.3 사용자 프롬프트
```text
{
  "@context": "https://schema.org",
  "@type": "Movie",
  "identifier": "[332]",
  "name": "Heart and Souls",
  "datePublished": 1993
}

Complete the missing schema.org/Movie fields for the above object.
Return one JSON-LD object only.
```

### 2.4 기대 출력 (예시)
```json
{
  "@context": "https://schema.org",
  "@type": "Movie",
  "identifier": "[332]",
  "name": "Heart and Souls",
  "datePublished": 1993,
  "director": {"@type": "Person", "name": "Ron Underwood"},
  "countryOfOrigin": "US",
  "inLanguage": "en",
  "genre": ["Comedy", "Fantasy"],
  "contentRating": "PG-13",
  "duration": "PT104M"
}
```

---

## 3. Standard-Aware User Profile Prompt (`P_U*`)

### 3.1 시스템 프롬프트
```text
You are a user profiling assistant. Infer a viewer profile from
viewing history and output ONE JSON object with these fields:
  - ageRange: one of {"<18","18-24","25-34","35-44","45-54","55-64","65+"}
  - gender: one of {"male","female","non-binary","unknown"}
  - preferredLanguages: array of BCP 47 tags
  - residenceCountry: ISO 3166-1 alpha-2 or null
  - likedGenres: array from schema.org genre vocabulary
  - dislikedGenres: array from schema.org genre vocabulary
  - likedDirectors: array of person names
Do not invent demographic facts that are not implied by the history.
Use null/[] for unknowns.
```

### 3.2 사용자 프롬프트
```text
History (schema.org/Movie objects):
[
  {"identifier":"[332]","name":"Heart and Souls","datePublished":1993,
   "genre":["Comedy","Fantasy"]},
  {"identifier":"[364]","name":"Men with Brooms","datePublished":2002,
   "genre":["Comedy","Drama","Romance"]}
]
Generate the user profile JSON.
```

### 3.3 기대 출력
```json
{
  "ageRange": "45-54",
  "gender": "female",
  "preferredLanguages": ["en"],
  "residenceCountry": "CA",
  "likedGenres": ["Comedy","Drama","Romance","Fantasy"],
  "dislikedGenres": ["Horror","Thriller"],
  "likedDirectors": ["Ron Underwood"]
}
```

---

## 4. Standard-Aware u-i Edge Prompt (`P_UI*`)

### 4.1 시스템 프롬프트
```text
You select likely positive and negative items from a candidate set
based on a user's prior history. Output ONE JSON object:
  {"positive": <id>, "negative": <id>}
Use the integer identifier from the candidate set. No prose.
```

### 4.2 사용자 프롬프트
```text
History: [332]Heart and Souls (1993, Comedy|Fantasy),
         [364]Men with Brooms (2002, Comedy|Drama|Romance)
Candidates:
  [121] The Vampire Lovers (1970, Horror)
  [155] Billabong Odyssey (2003, Documentary)
  [248] The Invisible Guest (2016, Crime|Drama|Mystery)
Return JSON only.
```

### 4.3 기대 출력
```json
{"positive": 248, "negative": 121}
```

---

## 5. 표준학적 효과 (정성)

| 효과 | 메커니즘 |
|---|---|
| **데이터 분산 억제** | ISO 코드 강제로 'USA'/'United States' 같은 표면 분산을 사전 차단. 임베딩 공간에서 동일 의미 토큰이 한 점으로 수렴. |
| **상호운용성 보장** | 출력이 곧 schema.org/Movie JSON-LD validator 통과 → 외부 추천/검색 시스템 직접 통합 가능. |
| **환각 가시화** | "country" 같은 placeholder 환각이 `null`로 강제 감소 → 다운스트림에서 결측치로 명시적 처리 가능. |
| **Long-tail 효율** | 희소 국가/언어가 코드 차원에서 동일 그룹화 → cold-start 시 전이 신호 강화. |

## 6. 본 텀페이퍼의 검증 범위

| 측면 | 본 연구에서 처리 | 후속 (졸업논문) |
|---|---|---|
| Prompt-time 표준 주입 | **설계만** (본 문서) | 실제 sLLM(Llama-3-8B)로 재증강 |
| Post-hoc 정규화 | **구현/실험** | 동일 모듈 재사용 |
| Standard Compliance Rate | **측정** | — |
| Recall/NDCG 트레이드오프 | **측정** | 멀티모달 결합 후 재측정 |
