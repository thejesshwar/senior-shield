from gtts import gTTS
gTTS("Warning. Scam detected. Do not click any links.", lang='en').save("alert_scam_en.mp3")
gTTS("This message is safe.", lang='en').save("alert_safe_en.mp3")
gTTS("सावधान। यह एक धोखा है। लिंक पर क्लिक न करें।", lang='hi').save("alert_scam_hi.mp3")
gTTS("यह संदेश सुरक्षित है।", lang='hi').save("alert_safe_hi.mp3")

print("Audio files generated successfully!")