//This has the complete backend part

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const inf = require("@huggingface/inference");
const cors = require('cors');

async function imageToBlob(filePath) {
    try {
      const data = await fs.promises.readFile(".\\"+filePath);

      const blob = Buffer.from(data, 'binary');
  
      return blob;
    } catch (error) {
      throw error;
    }
}

async function rendering(blb){
    const hflnference = new inf.HfInference("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");
    const result = await hflnference.imageClassification({
        data: blb,
        model: "Heem2/bone-fracture-detection-using-xray",
    })

    return result;
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
      const ext = path.extname(file.originalname);
      cb(null, file.fieldname + '-' + Date.now() + ext);
    }
  });
const upload = multer({ storage: storage});

const app = express();
app.use(cors());

app.post('/upload', upload.single('file'), async (req, res) => {
    
    imageToBlob(req.file.path).then((blob) => {
        rendering(blob).then((result) => {
            console.log(result);
            res.json(result);
        }).catch((err) => {
            console.log(err);
            res.json({error: err});
        });
    }).catch((err) => {
        res.json({error: err});
    });
});



app.listen(3000, () => {
    console.log('Server is running on port 3000');
});