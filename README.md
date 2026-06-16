# FibroCheck — Fibroid Risk Prediction Web App

This app wraps the trained XGBoost model from `fibroid-risk-model-main`
(specifically `src/train.py` + `data/augmented_fibroid_data.csv`) into a
ready-to-deploy Flask web application with a friendly form.

---

## 📁 What's in this folder

```
fibroid-app/
├── app.py                      ← Flask web server (the brain)
├── train_and_save.py           ← Trains the model & saves fibroid_model.pkl
├── data_loader.py               ← Feature engineering (copied from src/)
├── augmented_fibroid_data.csv  ← The training data (200 balanced rows)
├── requirements.txt            ← Python libraries needed
├── Procfile                     ← Tells Render how to start the app
├── .gitignore
└── templates/
    └── index.html              ← The web form users see
```

✅ **This is everything needed.** The app automatically trains and saves
the model the first time it runs — no manual training step required.

---

## ✅ PART A — Run it on your own computer (recommended first step)

### A1. Install Python
If you don't have Python, download it from **python.org/downloads** (get
version 3.11 or 3.12) and install it. During install, tick **"Add Python to PATH"**.

### A2. Open a terminal in this folder
- **Windows:** open the `fibroid-app` folder in File Explorer → click the
  address bar → type `cmd` → press Enter
- **Mac:** open Terminal → type `cd ` (with a space) → drag the `fibroid-app`
  folder into the terminal window → press Enter

### A3. Install the required libraries
Type this and press Enter:
```bash
pip install -r requirements.txt
```
Wait for it to finish (1–2 minutes).

### A4. Run the app
```bash
python app.py
```
You'll see something like:
```
* Running on http://127.0.0.1:5000
```

### A5. Open it in your browser
Go to: **http://127.0.0.1:5000**

You'll see the FibroCheck form. Fill it in and click **"Assess My Risk"** —
you should get an instant prediction. 🎉

To stop the server, go back to the terminal and press `CTRL + C`.

---

## ✅ PART B — Deploy it online for free (Render.com)

This gives you a real shareable link, e.g.
`https://fibroid-risk-app.onrender.com`

### B1. Create a GitHub account (if you don't have one)
Go to **github.com** → Sign up.

### B2. Create a new repository
1. Click the **"+"** icon (top right) → **"New repository"**
2. Name it `fibroid-app`
3. Leave it **Public**
4. Click **"Create repository"**

### B3. Upload the files to GitHub
On the new repo's page:
1. Click **"uploading an existing file"** (a link in the middle of the page)
2. Drag **all the files and folders** from your `fibroid-app` folder into
   the browser window (including the `templates` folder)
3. Scroll down → click **"Commit changes"**

> ⚠️ Make sure `templates/index.html` ends up inside a `templates` folder
> on GitHub — not loose in the root. If GitHub flattens it, create the
> `templates` folder manually first (click "Create new file", type
> `templates/index.html` as the filename — GitHub auto-creates the folder).

### B4. Sign up for Render
1. Go to **render.com** → **"Get Started"** → sign up with your **GitHub account**
   (this makes linking repos automatic)

### B5. Create the Web Service
1. On the Render dashboard, click **"New +"** (top right) → **"Web Service"**
2. Find and select your `fibroid-app` repository → click **"Connect"**
3. Fill in these settings:

| Field | Value |
|---|---|
| **Name** | `fibroid-risk-app` (or anything you like) |
| **Region** | Choose the one closest to you |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | `Free` |

4. Scroll down → click **"Create Web Service"**

### B6. Wait for deployment
Render will show a live log. Wait for it to say:
```
==> Your service is live 🎉
```
This takes about 2–5 minutes.

### B7. Open your live app
At the top of the Render page, you'll see a URL like:
```
https://fibroid-risk-app.onrender.com
```
Click it — your app is now live for anyone to use!

> 💡 **Free tier note:** Render's free apps "sleep" after 15 minutes of no
> traffic and take ~30–60 seconds to wake up on the next visit. This is
> normal and fine for a final-year project demo.

---

## 🧪 How to test it works correctly

Try these two examples (taken directly from the training data) to confirm
the model gives sensible results:

**Example 1 — should show HIGH risk:**
- Age: 29, Height: 1.72, Weight: 49, BP: 110/50
- Symptoms: Bleeding ✅, Lower Abdominal Pain ✅, General Abdominal Pain ✅

**Example 2 — should show LOW risk:**
- Age: 34, Height: 1.4, Weight: 107, BP: 109/96
- Symptoms: none ticked

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `pip` not recognized | Reinstall Python and tick "Add to PATH" |
| Port 5000 already in use | Close other running apps, or change `port` in `app.py` |
| Render build fails on xgboost | Check the build logs — usually a version mismatch; the `requirements.txt` here is tested and known to work |
| App shows "Application Error" on Render | Click "Logs" in Render dashboard to see the exact error, then share it here |

---

## 📊 Model Summary

- **Algorithm:** XGBoost Classifier (100 trees, max depth 4)
- **Preprocessing:** MinMax scaling
- **Training data:** 200 samples (balanced 100/100 via augmentation)
- **Features (16):** age, height, weight, BMI, systolic/diastolic BP,
  9 symptom flags, has_any_symptom

## ⚠️ Disclaimer

This is a final-year academic research project and **not a certified
medical device**. It should never replace professional medical advice.
