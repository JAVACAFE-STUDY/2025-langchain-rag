/**
 * 매개변수를 따로따로 입력받게 하고 싶을 때 사용하는 방법
 */
import {PromptTemplate} from '@langchain/core/prompts';
import dayjs from 'dayjs';
import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config';


const prompt = PromptTemplate.fromTemplate('나이: {age}\n직업: {job}');

// 매개변수 age를 받고
const partialPrompt = await prompt.partial({age: '20'});
// 추후에 학생 매개변수를 받는다.
const result = await partialPrompt.format({job: "학생"});

console.log(result);

function getDatetime() {
    return dayjs().format('YYYY-MM-DD HH:mm:ss');
}

const prompt2 = PromptTemplate.fromTemplate('Tell me a {adjective} joke the day {date}.');


const partialPrompt2 = await prompt2.partial({date: getDatetime()});
const result2 = await partialPrompt2.format({adjective: "funny"});

console.log(result2);




