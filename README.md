# SAIG-Project (Sep. 2023)
"For applying SAIG Lab" ผลงานเกี่ยวกับ AI ที่ตนเองภูมิใจ 1 ชิ้นงาน (ปวิช รุ่งฐานอุดม 66010473 วิศวกรรมคอมพิวเตอร์ ชั้นปีที่ 1)
![SAIG-Project](https://github.com/pawito236/SAIG-Project/assets/44425803/3b93e8b3-f126-46a5-886d-0d9c19d23f34)

# จุดเริ่มต้นไอเดีย (Aug. - Sep. 2020)
ณ ตอนสมัยชั้น ม.ปลาย ได้เริ่มต้นทำ IoT โปรเจคด้วย Arduino - ESP32 ซึ่งตอนนั้นทีมก็ยังไม่ได้มีประสบการณ์ด้านนี้เท่าไร เราก็ได้คุยไอเดียกันจนได้ไอเดียที่ว่า จะเป็นไปได้ไหมที่เราจะคุยกับต้นไม้ ต้นไม้ที่เราปลูกจะสามารถสื่อสารกับเราได้ ทำให้เรารู้ในสิ่งที่เขาต้องการ เราสามารถดูแลเขาได้แม้เราจะไม่มีความรู้การปลูกต้นไม้ ... "จะเป็นไปได้ไหมที่เราจะเป็นเพื่อนกับต้นไม้" 

จนเกิดเป็น Concept "Plant Buddy ต้นไม้เพื่อนรัก" คือการทำให้ต้นไม้สามารถเป็นเพื่อนกับเราได้ โดยสร้างกระถางต้นไม้ Iot ด้วย Esp32 และ Sensor ต่าง ๆ เพื่อเก็บข้อมูลต้นไม้ไปยัง google sheet  และผู้ใช้สามารถสนทนากับต้นไม้ได้ผ่าน Line bot ที่เชื่อมกับ dialogflow บทสนทนาจะเปลี่ยนไปตามสถานะที่แปลผลมาจากข้อมูลใน Google sheet

แต่ในสมัยนั้น ผมที่เป็นฝ่าย Tech ยังไม่มีความรู้ด้าน Ai สักอย่าง ไม่รู้แม้กระทั่ง NLP คืออะไร ก็ได้แต่ค้นหาว่า จะทำแชทบอทยังไง ๆ ก็ได้ไปเจอ Dialogflow ที่เราสามารถพัฒนาแชทบอทและเชื่อม Webhook กับ line ได้ ก็เลยเลือกใช้ dialogflow ในการทำแชทบอท้ ซึ่งผมก็ยังไม่รู้เลยว่า dialogflow มันจะ Match intent ได้ยังไง =_= สมัยนั้น ChatGPT ก็ยังไม่มี ก็เรียกว่าทำออกมาเป็นแชทบอทได้ก็สุดยอดสำหรับผมมากแล้ว

ซึ่งมันมีหลายจุดมากที่ผมเชื่อว่าเราสามารถทำให้ดีกว่านี้ได้ แล้วด้วยเทคโนโลยีปัจจุบัน กับความสามารถที่ผมมีในตอนนี้ ผมอยากจะพัฒนาส่วนของ Software ให้มีประสิทธิภาพมากขึ้น และทำให้ Concept "Plant Buddy ต้นไม้เพื่อนรัก" ใกล้เคียงความเป็นจริงมากขึ้น

# การพัฒนาเพิ่มเติม (Sep. 2023)
จากที่กล่าวมา ในตอนนี้ผมมีความรู้มากขึ้น มีประสบการณ์ทำงานในด้าน Ai กว่าเมื่อก่อน ทำให้อยากที่จะพัฒนาส่วนของ Software ให้ดีขึ้น โดยได้ทำความรู้ด้าน NLP + Speech เข้ามาใช้พํฒนา
- Q&A ปรับเปลี่ยนจาก Dialogflow เป็น ChatGPT และทำ Prompt Engineering ในการปรับแต่งบุคลิก ลักษณะนิสัยการตอบกลับของ Chatbot
- เพิ่ม Feature การโต้ตอบด้วยเสียง โดย STT (Speech-to-Text) ใช้งาน Whisper และ TTS eng ใช้ Speech-T5 / TTS th ใช้ google api
- ปรับ Database เป็น firestore เพื่อให้การจัดเก็บข้อมูลมีประสิทธิภาพมากขึ้น
- ใช้งาน Pinecone Vector database ในการเก็บ Context ที่ใช้ในการตอบคำถาม

By ปวิช รุ่งฐานอุดม 66010473 วิศวกรรมคอมพิวเตอร์ ชั้นปีที่ 1
![3](https://github.com/pawito236/SAIG-Project/assets/44425803/38a989ca-1ee9-4490-9368-975a7cbc354f)

# Award & Credit (Aug. - Sep. 2020)
![PlantBuddy-resize](https://github.com/pawito236/SAIG-Project/assets/44425803/ff408dea-ace4-4992-8e04-1aa001483888)
