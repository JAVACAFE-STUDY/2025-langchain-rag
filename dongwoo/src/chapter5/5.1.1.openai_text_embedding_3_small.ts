import { OpenAIEmbeddings } from "@langchain/openai";
import 'dotenv/config'

const embeddingModel = new OpenAIEmbeddings({model: 'text-embedding-3-small'});
const embeddings = await embeddingModel.embedDocuments([
    'Hi,there!',
    'Oh, hello!',
    `What's your name?`,
    `My friends call me World`,
    `Hello World!`,
]);

console.log(embeddings.length, embeddings[0].length);


