# 3장 LLM 시작하기

## 3.1 랭체인이란?

* RAG 시스템을 구축하기 위해 설계된 다양한 툴이 존재하는데, 그중 가장 널리 활용되고 있는 툴

### 랭체인의 개념과 구성 요소

* Language + Chain
* 언어를 잇는 사슬
* 프롬프트 엔지니어링: LLM에게 원하는 답변을 받기 위해 프롬프트를 고도화하는 작업
* 질문을 잘할 수록 LLM의 답변 품질이 좋아진다는 전제를 바탕으로 등장
* 단순히 LLM에게 전달되는 프롬프트를 병형한다고 해서 위 문제를 해결할 수는 없다.
* 위 문제를 해결하기 위해 chain이라는 개념을 만들어서 하나의 템플릿을 거쳐 전달하도록 추가적인 '연결고리'를 만들어 원하는 답변을 이끌어 낼 수 있도록 처리하는 방식

   * Models: OpenAI, Google, Anthropic과 같은 LLM
   * Prompts: Prompt Template, Partiail 과 같은 사용자의 프롬프트를 재가공
   * Document Loaders: 백터 DB로 구축할 문서를 불러오는 역할
   * Text Splitters: 위 Document Loaders를 통해 만든 문서들을 여러 청크로 분할하는 역할
   * Vector Stores: 분할된 텍스트 청크들을 저장하는 역할
   * Output Parsers: 원하는 답변을 여러가지 형태(scv, json 등등)로 변경해주는 역할

   