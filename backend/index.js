const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const cors = require('cors');
const fs = require('fs');

const app = express();
app.use(cors());
const upload = multer({ dest: 'uploads/' });

// The route that connects to your Python AI
app.post('/api/detect', upload.single('video'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: "No file uploaded" });

        // Prepare the data to send to Python FastAPI (Port 8000)
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path), req.file.originalname);

        const aiResponse = await axios.post('http://127.0.0.1:8000/analyze', formData, {
            headers: formData.getHeaders(),
        });

        // Delete the temp file after processing
        fs.unlinkSync(req.file.path);

        res.json(aiResponse.data);
    } catch (error) {
        console.error("AI Service Error:", error.message);
        res.status(500).json({ error: "AI Service is offline or failed." });
    }
});

app.listen(5001, () => console.log('Backend Bridge running on port 5001'));