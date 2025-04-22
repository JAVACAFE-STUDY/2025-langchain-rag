// 1단계 LLM 모델 선택
import {ChatOpenAI} from '@langchain/openai';
// 2단계 Document Loader 선정
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// 3단계 text splitter을 선언
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// 4단계 임베딩 모델 선정
import {OpenAIEmbeddings} from '@langchain/openai';
// 5단계 벡터 스토어 선정
import { Chroma } from "@langchain/community/vectorstores/chroma";
// 6단계 Retriever 설정(Chroma DB에 있는 기능 사용)
import * as hub from  'langchain/hub';

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { BaseChatMessageHistory } from "@langchain/core/chat_history";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";

import 'dotenv/config'
import { BaseMessage } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';

console.log('Conversational RAG chain created');

const loader = new PDFLoader("/Users/user/Downloads/대한민국헌법(헌법)(제00010호)(19880225).pdf");
const pages = await loader.load();

// PDF 파일을 1000자 청크로 분리
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
});
const _docs = await splitter.splitDocuments(pages);
const docs = _docs.map(doc => ({
    ...doc,
    metadata: {source: doc.metadata.source, page: doc.metadata.loc.paggeNumber}
}));


//ChromaDB에 청크들을 저장
const vectorStore = await Chroma.fromDocuments(docs, new OpenAIEmbeddings({
    model: 'text-embedding-3-small'
}), {
    collectionName: 'qachartbot',
    collectionMetadata: {
        "hnsw:space": "cosine",
    }    
});

// GPT-4o-mini 모델 사용
const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
});

const retriever = vectorStore.asRetriever();

const contextualizeQSystemPrompt = `
    Given a chat history and the lastest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT the question,
    just reformulate it if needed and otherwise return it as is.`;

const contextualizeQprompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ["human", "{input}"],
]);

const historyAwareRetriever = await createHistoryAwareRetriever({
    llm, 
    retriever, 
    rephrasePrompt: contextualizeQprompt
});

const qaSystemPrompt = `
    You are an assistant for question-answering tasks.
    Use the following prices of retreeved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.


    {context}
`;

const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ["human", "{input}"],
]);

const quesiontAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
});
const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: quesiontAnswerChain,
});

const store = {} as Record<string, BaseChatMessageHistory>;

function getSessionHistory(sessionId: string): BaseChatMessageHistory {
    if (!store[sessionId]) {
        store[sessionId] = new InMemoryChatMessageHistory();
    }
    return store[sessionId];
}

const conversationalRagChain = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: 'input',
    outputMessagesKey: 'answer',
    historyMessagesKey: 'chat_history',
});    

export async function addFile(file: Blob) {
    const loader = new PDFLoader(file);
    const pages = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });
    const _docs = await splitter.splitDocuments(pages);
    const docs = _docs.map(doc => ({
        ...doc,
        metadata: {source: doc.metadata.source, page: doc.metadata.loc.paggeNumber}
    }));

    await vectorStore.addDocuments(docs);
    console.log('Added documents to vector store:', docs);
}

export async function chat(input: string, sessionId: string) {
    console.log('input:', input);
    console.log('sessionId:', sessionId);

    if (!conversationalRagChain) {
        return {
            context: [],
            answer: 'Conversational RAG chain is not initialized'
        };
    }

    const result = await conversationalRagChain.invoke(
        {
            "input": input
        },
        {
            configurable: {"sessionId": sessionId},
        }
    );
    
    return result;
};

