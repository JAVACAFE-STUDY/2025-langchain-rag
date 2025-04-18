import express, {Express, NextFunction, Request, Response} from 'express';
import {ErrorUtils} from './utils/errorUtils/index.ts';
import {Custom} from './service/custom/index.ts';
import dotenv from 'dotenv';
import multer from 'multer';
import session from 'express-session';
import cors from 'cors';

// ------------------ SETUP ------------------
dotenv.config();

// this is used for parsing FormData
const upload = multer();

const app: Express = express();
const port = 8080;

// this will need to be reconfigured before taking the app to production
app.use(cors());

app.use(express.json());
app.use(session({
    secret: 'keyboard cat',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: true }    
}));

// ------------------ CUSTOM API ------------------

app.post('/chat', async (req: Request, res: Response) => {
    Custom.chat(req.body, res, req);
});

app.post('/chat-stream', async (req: Request, res: Response) => {
    Custom.chatStream(req.body, res);
});

app.post('/files', upload.array('files'), async (req: Request, res: Response) => {
    Custom.files(req, res);
});

// ------------------ START SERVER ------------------

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});

// ------------------ ERROR HANDLER ------------------

app.use(ErrorUtils.handle);