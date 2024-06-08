//This has the complete backend part

const e_x_p_r_e_ss = require('express');
const m_u_l_t_e_r = require('multer');
const p_a_t_h = require('path');
const f_s = require('fs');
const i_n_f = require("@huggingface/inference");
const c_o_r_s = require('cors');

async function image_To_Blob(file_Path) {
    try {
      const d_a_t_a = await f_s.promises.readFile(".\\"+file_Path);

      const b_l_o_b = Buffer.from(d_a_t_a, 'binary');
  
      return b_l_o_b;
    } catch (error) {
      throw error;
    }
}

async function ren_der_ing(b_l_b){
    const hf_lnfer_ence = new i_n_f.HfInference("HF-access-token");
    const res_ult = await hf_lnfer_ence.imageClassification({
        data: b_l_b,
        model: "Heem2/bone-fracture-detection-using-xray",
    })

    return res_ult;
}

const sto_rage = m_u_l_t_e_r.diskStorage({
    destination: function (req, file, cb) {
      cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
      const ext = p_a_t_h.extname(file.originalname);
      cb(null, file.fieldname + '-' + Date.now() + ext);
    }
  });
const up_load = m_u_l_t_e_r({ storage: sto_rage});

const a_p_p = e_x_p_r_e_ss();
a_p_p.use(c_o_r_s());

a_p_p.post('/upload', up_load.single('file'), async (req, res) => {
    
    image_To_Blob(req.file.path).then((blob) => {
        ren_der_ing(blob).then((result) => {
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



a_p_p.listen(3000, () => {
    console.log('Server is running on port 3000');
});
