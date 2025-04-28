// 1단계 LLM 모델 선택
import {ChatOpenAI} from '@langchain/openai';
import * as hub from "langchain/hub/node";
import type { ChatPromptTemplate } from '@langchain/core/prompts';
import { TavilySearch } from "@langchain/tavily";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";

import 'dotenv/config'

const tool = new TavilySearch({
    maxResults: 5,
    topic: "general",
    // includeAnswer: false,
    // includeRawContent: false,
    // includeImages: false,
    // includeImageDescriptions: false,
    // searchDepth: "basic",
    // timeRange: "day",
    // includeDomains: [],
    // excludeDomains: [],
});

// GPT-4o-mini 모델 사용
const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
});

// 프롬프트 불러오기
const prompt = await hub.pull<ChatPromptTemplate>("hwchase17/openai-tools-agent:c1867281");

const tools = [tool];
const agent = await createToolCallingAgent({ llm, tools, prompt });

console.log('prompt:', prompt);

// AgentExecutor 생성
const agentExecutor = new AgentExecutor({ agent, tools, verbose: true });

export async function chat(input: string, sessionId: string) {
    console.log('input:', input);
    console.log('sessionId:', sessionId);

    // const result = await agentExecutor.invoke({
    //     input: input,
    // });

    const result = await tool.invoke({
        query: input,       
    });

    console.log('result:', result);
    
    return result;
};