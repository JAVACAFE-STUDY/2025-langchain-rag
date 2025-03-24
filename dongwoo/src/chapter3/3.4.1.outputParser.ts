/**
 * 프롬프트 엔지니어링: LLM API를 활용하여 구조화된 답변을 받고 싶을 때 사용할 수 있는 방법
 * 
프롬프트 엔지니어링(Prompt Engineering)은 생성형 인공지능(Generative AI)과 상호작용할 때, 원하는 결과를 얻기 위해 입력 명령어(프롬프트)를 설계하고 최적화하는 과정입니다. 
이 기술은 AI 모델의 성능을 극대화하고, 사용자의 의도를 정확히 반영하여 고품질의 결과를 생성하도록 돕는 데 핵심적인 역할을 합니다.

### **프롬프트 엔지니어링의 주요 개념**
1. **프롬프트란?**
   - AI 모델에 제공되는 입력값으로, 명령, 질문, 요청 등을 포함합니다. 예를 들어, "책 추천해줘" 같은 간단한 요청부터 "20대가 읽기 좋은 자기계발 도서 5권을 추천해줘"처럼 구체적인 지시까지 다양합니다.

2. **목적**
   - AI가 사용자의 의도를 정확히 이해하고, 관련성 높고 유용한 결과를 생성하도록 돕는 것입니다. 이를 통해 모델의 성능을 향상시키고 사용자 경험을 개선할 수 있습니다.

3. **구성 요소**
   - **명확성**: 프롬프트는 간결하고 명확해야 합니다.
   - **구체성**: 필요한 정보를 구체적으로 제공해야 합니다.
   - **문맥**: AI가 적절히 이해할 수 있도록 충분한 배경 정보를 제공합니다.
   - **의도**: 사용자의 목표를 명확히 전달해야 합니다.

### **활용 예시**
- 챗봇 및 대화형 AI 개발
- 콘텐츠 생성 (기사, 광고 카피 등)
- 코드 작성 및 오류 수정
- 언어 번역 및 교육 콘텐츠 제작 등.

### **중요성**
프롬프트 엔지니어링은 AI 모델의 잠재력을 최대한 활용하기 위한 핵심 기술로, 특히 대규모 언어 모델(LLM)의 등장과 함께 그 필요성이 더욱 강조되고 있습니다. 이를 통해 다양한 도메인에서 효율적이고 창의적인 결과를 얻을 수 있습니다.
 */
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';

import {CommaSeparatedListOutputParser} from '@langchain/core/output_parsers';

import 'dotenv/config';


const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
// CSV 로 받고 싶을 경우 해당 규칙을 LLM에 정확하게 요청해야 하는데 요청하는게 쉽지 않다.
// 랭체인에서는 해당 답변을 쉽게 출력할 수 있도록 Output Parser를 제공한다.
const outputParser = new CommaSeparatedListOutputParser();

// 이렇게 물어보면 어렵네..
const chatTemplate = ChatPromptTemplate.fromMessages([
    new SystemMessage("너는 영화 전문가 AI야. 사용자가 원하는 장르의 영화를 리스트 형태로 추천해줘."),
    new SystemMessage("ex) Query: SF 영화 3개 추천해줘 / 답변: ['인터스텔라', '스페이지오디세이', '혹성탈출']"),
    HumanMessagePromptTemplate.fromTemplate('{text}')
]);

const messages = await chatTemplate.formatMessages({
    text: "액션 영화 5개를 추천해줘."
});

const answer = await model.invoke(messages);

console.log(answer.content);


const prompt = PromptTemplate.fromTemplate("List {number} {subject}. answer in Korean.\n{formatInstructions}");

const chain = (await prompt
    .partial({formatInstructions: outputParser.getFormatInstructions()}))
    .pipe(model);

const result = await chain.invoke({subject: '공포 영화', number: '4'})

console.log(result.content);



