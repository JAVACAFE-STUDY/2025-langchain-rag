import {Request, Response} from 'express';
import {chat} from './chat.ts';

// import { ChatOpenAI } from "@langchain/openai";
// import { HumanMessage, SystemMessage } from "@langchain/core/messages";
// import 'dotenv/config'

// const model = new ChatOpenAI({ model: "gpt-4o-mini" });

// type DeepChatMessage = {
//     role: string;
//     text: string;
// }    

// async function chat(messages: DeepChatMessage[]) {
//     const result = await model.invoke(messages.map(message => new HumanMessage(message.text)));
//     return result;
    
// }

export class Custom {
    public static async chat(body: Request['body'], res: Response, req: Request) {
        // Text messages are stored inside request body using the Deep Chat JSON format:
        // https://deepchat.dev/docs/connect
        // @ts-ignore
        const result = await chat(body.messages?.map(message => message.text)?.join('\n') || '', 'test');
        // Sends response back to Deep Chat using the Response format:
        // https://deepchat.dev/docs/connect/#Response
        res.json({text: result.answer});
    }

    public static async chatStream(body: Request['body'], res: Response) {
        // Text messages are stored inside request body using the Deep Chat JSON format:
        // https://deepchat.dev/docs/connect
        console.log(body);
        const responseChunks = 'This is a respone from ExpressJs server. Thankyou for your message!'.split(' ');
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*');
        Custom.sendStream(res, responseChunks);
    }

    private static sendStream(res: Response, responseChunks: string[], chunkIndex = 0) {
        setTimeout(() => {
        const chunk = responseChunks[chunkIndex];
        if (chunk) {
            // Sends response back to Deep Chat using the Response format:
            // https://deepchat.dev/docs/connect/#Response
            res.write(`data: ${JSON.stringify({text: `${chunk} `})}\n\n`);
            Custom.sendStream(res, responseChunks, chunkIndex + 1);
        } else {
            res.end();
        }
        }, 70);
    }

    public static async files(req: Request, res: Response) {
        // Files are stored inside a form using Deep Chat request FormData format:
        // https://deepchat.dev/docs/connect
        if (req.files as Express.Multer.File[]) {
        console.log('Files:');
        console.log(req.files);
        }
        // When sending text along with files, it is stored inside the request body using the Deep Chat JSON format:
        // https://deepchat.dev/docs/connect
        if (Object.keys(req.body).length > 0) {
        console.log('Text messages:');
        // message objects are stored as strings and they will need to be parsed (JSON.parse) before processing
        console.log(req.body);
        }
        // Sends response back to Deep Chat using the Response format:
        // https://deepchat.dev/docs/connect/#Response
        res.json({text: 'This is a respone from ExpressJs server. Thankyou for your message!'});
    }
}