/**
 * LLM에서 Temperature는 답변의 일관성과 관련이 있습니다.
 * Temperature가 낮으면, LLM은 '안전한' 선택을 합니다. 가장 확률이 높은 단어를 고릅니다.
 * Temperature가 높으면, LLM은 더 '모험적인' 선택을 합니다. 확률이 좀 낮은 단어들도 성택할 가능성이 높아집니다.
 */
import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config'

const model1 = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0, maxTokens: 128 });
const model2 = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0, maxTokens: 128  });

const model3 = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 1, maxTokens: 128  });
const model4 = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 1, maxTokens: 128  });

const modelList = [model1, model2, model3, model4];

for (const model of modelList) {
    const answer = await model.invoke("왜 자바스크립트가 가장 인기 있는 프로그래밍 언어인지 한 문장으로 설명해줘");

    console.log("-".repeat(100));
    console.log(model.temperature, ">>>", answer.content);
}

export {};