import os
import json
import requests
import time
from flask import Flask, render_template, request, jsonify
import base64
import random
import csv
from datetime import datetime

# 初始化 Flask 應用
app = Flask(__name__)

# --- API 配置 ---
API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_TEXT_MODEL = "gemini-2.5-flash-preview-09-2025" 
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

# --- 核心 AI 呼叫函式 (保持原樣) ---

def call_gemini_api(prompt: str, system_instruction: str) -> str:
    """呼叫 Gemini API，加入重試機制解決 429 錯誤。"""
    if not API_KEY:
        return "回饋失敗：AI 服務未配置 (API Key 缺失)。"

    url = f"{GEMINI_API_BASE}{GEMINI_TEXT_MODEL}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{ "text": system_instruction }]},
        "generationConfig": {"temperature": 0.5}
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 429:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
            return generated_text.strip() if generated_text else "回饋失敗：內容生成空值。"
        except Exception as e:
            if attempt == max_retries - 1:
                return "回饋失敗：AI 老師連線異常，請稍後再試。"
            time.sleep(1)
    return "回饋失敗。"

def call_gemini_image_api(user_sentence: str) -> str:
    """呼叫生圖：保持原樣。"""
    if not user_sentence:
        return None
    try:
        seed = int(time.time())
        style_prompt = f"children's book illustration style, simple, cute, {user_sentence}"
        safe_prompt = requests.utils.quote(style_prompt)
        img_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=512&height=512&nologo=true&seed={seed}&model=stable-diffusion-xl"
        response = requests.get(img_url, timeout=30)
        if response.status_code == 200 and response.content:
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        pass
    return None

# --- 修改後的儲存記錄功能 ---
def save_to_csv(data_dict):
    file_path = 'record.csv'
    # 根據需求新增四個評分欄位
    fieldnames = [
        'timestamp', 'level', 'feedback_round', 'selected_words', 'accuracy', 
        'user_sentence', 'ai_feedback', 'word_score', 'sentence_score', 
        'image_score', 'total_score'
    ]
    
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print(f"CSV 寫入失敗: {e}")

# --- AI 輔助功能 (保持原樣) ---
def get_sentence_analysis(user_sentence: str, correct_selected: list, wrong_selected: list, missing_words: list, target_answers: list, sentence_prompt: str) -> str:
    if len(missing_words) == 0 and len(wrong_selected) == 0:
        status_msg = "🌟 太厲害了！你完全觀察正確，找齊了所有單字！"
    else:
        status_msg = "⚠️ 圖片裡還有一些東西你沒發現喔！"

    system_instruction = (
        "你是一位國中一年級英文老師。請根據『原始圖片包含的正確單字』進行回饋。\n"
        "1. 禁止使用任何 Markdown 符號（如 ** 或 __）。\n"
        "2. 每一點 (1., 2., 3.) 之前必須換行。\n"
        "3. 單字提示：請專注於針對『學生漏選的正確單字』提供外觀、特徵或位置線索，不准說出英文單字本身。\n"
        "4. 畫面引導：必須嚴格參考『原始圖片正確單字』。如果學生造句與圖中事實不符（例如圖中是鴨子，學生寫貓），請禮貌指出。每次只建議增加一個簡單細節，引導學生慢慢改進。"
    )

    prompt = (
        f"【事實參考】\n"
        f"圖片中真實存在的正確單字: {', '.join(target_answers)}\n"
        f"學生選中的正確單字: {', '.join(correct_selected)}\n"
        f"學生選錯的單字: {', '.join(wrong_selected)}\n"
        f"學生遺漏的單字: {', '.join(missing_words)}\n"
        f"學生目前造句: 『{user_sentence}』\n"
        f"要求句型: 『{sentence_prompt}』\n\n"
        "請依照此格式回報：\n\n"
        "1. 單字提示：(若有漏選，提供其特徵線索；若有選錯，溫和糾正。請勿列出正確單字拼法)\n\n"
        "2. 文法修正：(分析造句文法，並檢查是否『符合圖片事實』)\n\n"
        "3. 畫面引導建議：(根據圖片內容，引導學生下一步可以加入的一個小細節，例如顏色或大小)"
    )

    ai_critique = call_gemini_api(prompt, system_instruction)
    ai_critique = ai_critique.replace("2. ", "\n\n2. ").replace("3. ", "\n\n3. ")

    final_feedback = (
        f"{status_msg}\n\n"
        f"{ai_critique}"
    )
    return final_feedback

# --- Flask 路由 ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/easy")
def easy_mode():
    return render_template("easy_mode.html")

@app.route("/hard")
def hard_mode():
    return render_template("hard_mode.html")

@app.route("/api/ai_feedback", methods=["POST"])
def get_ai_feedback():
    try:
        data = request.get_json()
        level_idx = data.get('level', 1)
        user_sentence = data.get('user_sentence', '').strip()
        sentence_prompt = data.get('sentence_prompt', '').strip()
        selected_cards = data.get('correct_words', []) 
        round_index = data.get('feedback_count', 0)
        feedback_round_text = f"第{round_index + 1}次回饋"

        with open('static/data/easy_mode.json', 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        current_level_data = next((item for item in full_data if item["level"] == int(level_idx)), None)
        if not current_level_data:
            return jsonify({"feedback": "找不到關卡資料"}), 400
        
        standard_answers = [a.lower() for a in current_level_data["answer"]]

        correct_count = 0
        for word in selected_cards:
            if word.lower() in standard_answers:
                correct_count += 1
        
        accuracy_val = round(correct_count / 3, 2)
        accuracy_str = f"{accuracy_val:.6f}"

        user_selected_lower = [w.lower() for w in selected_cards]
        correct_selected = [w for w in selected_cards if w.lower() in standard_answers]
        wrong_selected = [w for w in selected_cards if w.lower() not in standard_answers]
        missing_words = [w for w in standard_answers if w not in user_selected_lower]

        if not user_sentence:
            return jsonify({"feedback": "請先輸入造句。"})

        feedback = get_sentence_analysis(
            user_sentence, 
            correct_selected, 
            wrong_selected, 
            missing_words, 
            standard_answers, 
            sentence_prompt
        )

        # 紀錄回饋：評分相關欄位記為 nan
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'level': level_idx,
            'feedback_round': feedback_round_text,
            'selected_words': ",".join(selected_cards),
            'accuracy': accuracy_str,
            'user_sentence': user_sentence,
            'ai_feedback': feedback.replace('\n', ' '),
            'word_score': 'nan',
            'sentence_score': 'nan',
            'image_score': 'nan',
            'total_score': 'nan'
        }
        save_to_csv(log_data)

        return jsonify({"feedback": feedback})
    except Exception as e:
        print(f"Error in ai_feedback: {e}")
        return jsonify({"feedback": "伺服器處理錯誤。"}), 500

@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        level_idx = data.get('level', 1)
        user_sentence = data.get('user_sentence', '').strip()
        selected_cards = data.get('correct_words', []) # 從前端傳入目前選的單字
        
        if not user_sentence:
            return jsonify({"error": "無輸入句子"}), 400

        # 1. 生成圖片
        image_b64 = call_gemini_image_api(user_sentence)
        if not image_b64:
            return jsonify({"error": "圖片生成失敗"}), 500

        # 2. 評分邏輯
        with open('static/data/easy_mode.json', 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        current_level_data = next((item for item in full_data if item["level"] == int(level_idx)), None)
        standard_answers = current_level_data["answer"] if current_level_data else []
        
        # 單字分數 (0-3)
        word_score = sum(1 for w in selected_cards if w.lower() in [a.lower() for a in standard_answers])
        accuracy_str = f"{round(word_score / 3, 2):.6f}"

        # 呼叫 AI 評分造句與圖片
        grading_instruction = "你是一位專業的英文老師。請根據要求評分並僅回傳 JSON 格式。"
        grading_prompt = (
            f"請針對以下學生的表現給分：\n"
            f"目標單字：{', '.join(standard_answers)}\n"
            f"學生造句：『{user_sentence}』\n\n"
            "評分準則：\n"
            "1. 造句分數 (sentence_score, 0-4分)：評估文法、內容豐富度、是否包含目標單字。\n"
            "2. 圖片分數 (image_score, 0-3分)：評估此句子生成的畫面是否與目標單字內容語意相符（最高3分）。\n"
            "請嚴格回傳此格式：{\"sentence_score\": 分數, \"image_score\": 分數}"
        )
        
        grading_result = call_gemini_api(grading_prompt, grading_instruction)
        try:
            # 移除 Markdown 標籤並解析 JSON
            clean_json = grading_result.replace('```json', '').replace('```', '').strip()
            scores = json.loads(clean_json)
            sentence_score = int(scores.get("sentence_score", 0))
            image_score = int(scores.get("image_score", 0))
        except:
            sentence_score, image_score = 0, 0

        total_score = word_score + sentence_score + image_score

        # 3. 紀錄到 CSV
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'level': level_idx,
            'feedback_round': 'nan',
            'selected_words': ",".join(selected_cards),
            'accuracy': accuracy_str,
            'user_sentence': user_sentence,
            'ai_feedback': 'nan', # 生成圖片時無回饋文字
            'word_score': word_score,
            'sentence_score': sentence_score,
            'image_score': image_score,
            'total_score': total_score
        }
        save_to_csv(log_data)

        return jsonify({"image_data": image_b64})
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)