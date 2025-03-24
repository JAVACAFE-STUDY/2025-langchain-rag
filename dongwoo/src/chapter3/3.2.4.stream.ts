/**
 * 스트리밍(Streaming): 사용자 질문에 마침 사람이 타이핑 하듯 답변하는 것을 볼 수 있는데 이런식으로 실시간으로 처리하는 방식을 스트리밍
 */
import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config'

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });



const streamResponse = async () => {
    const stream = await model.stream("달에 관한 시를 써줘");

    for await (const chunk of stream) {
        process.stdout.write(chunk.content as string);
    }
}

streamResponse();

export {};