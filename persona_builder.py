"""
Persona Builder — Extract habits, facts, personality, and communication style
================================================================================
All signals are derived from actual conversation data (keyword counts, regex
matches, statistical analysis).  No guessing.
"""

import json
import re
import os
import sys
from collections import Counter


# ── Keyword / Pattern Banks ────────────────────────────────────────────
SLEEP_LATE   = re.compile(r'\b(night owl|stay up late|late sleeper|can\'t sleep|insomnia|up all night|late night)\b', re.I)
SLEEP_EARLY  = re.compile(r'\b(early bird|morning person|wake up early|early riser|6 am|7 am)\b', re.I)
FOOD_PATTERNS = {
    "coffee drinker":  re.compile(r'\bcoffee\b', re.I),
    "tea drinker":     re.compile(r'\btea\b', re.I),
    "likes cooking":   re.compile(r'\b(cook|bake|recipe|baking)\b', re.I),
    "likes pizza":     re.compile(r'\bpizza\b', re.I),
    "vegetarian/vegan": re.compile(r'\b(vegan|vegetarian|plant.based)\b', re.I),
    "enjoys dining out": re.compile(r'\b(restaurant|dine|dining out|eat out)\b', re.I),
}
EXERCISE_PATTERNS = {
    "does yoga":       re.compile(r'\byoga\b', re.I),
    "runs/jogging":    re.compile(r'\b(running|jogging|marathon|jog)\b', re.I),
    "goes to gym":     re.compile(r'\bgym\b', re.I),
    "plays basketball": re.compile(r'\bbasketball\b', re.I),
    "plays soccer":    re.compile(r'\bsoccer\b', re.I),
    "does hiking":     re.compile(r'\bhik(e|ing)\b', re.I),
    "swims":           re.compile(r'\b(swim|swimming|pool)\b', re.I),
    "cycles":          re.compile(r'\b(cycl|biking|bike)\b', re.I),
}
HOBBY_PATTERNS = {
    "likes reading":   re.compile(r'\b(reading books|read a novel|reading a book|love to read)\b', re.I),
    "plays video games": re.compile(r'\b(video game|gaming|gamer|xbox|playstation|nintendo)\b', re.I),
    "enjoys music":    re.compile(r'\b(music|guitar|piano|band|sing|concert)\b', re.I),
    "likes movies/TV": re.compile(r'\b(movie|film|netflix|tv show|watching)\b', re.I),
    "enjoys travel":   re.compile(r'\b(travel|trip|vacation|abroad)\b', re.I),
    "likes photography": re.compile(r'\b(photo|photography|camera)\b', re.I),
    "enjoys art":      re.compile(r'\b(art|painting|drawing|sketch)\b', re.I),
}

# Relationship patterns
REL_PATTERNS = {
    "has a partner":   re.compile(r'\b(my girlfriend|my boyfriend|my husband|my wife|my partner|my fiancee?)\b', re.I),
    "close to family": re.compile(r'\b(my mom|my dad|my parents|my mother|my father|my brother|my sister)\b', re.I),
    "has children":    re.compile(r'\b(my son|my daughter|my kid|my children)\b', re.I),
    "has pets":        re.compile(r'\b(my dog|my cat|my pet|my puppy|my kitten)\b', re.I),
}

# Events patterns
EVENT_PATTERNS = {
    "moving/relocated":  re.compile(r'\b(moving to|moved to|relocat)\b', re.I),
    "new job/career":    re.compile(r'\b(new job|got hired|started working|career change|got promoted)\b', re.I),
    "studying/student":  re.compile(r'\b(studying|student|college|university|school|degree|graduate)\b', re.I),
    "got married":       re.compile(r'\b(got married|wedding|engagement|engaged)\b', re.I),
    "traveling soon":    re.compile(r'\b(going to visit|planning a trip|going on vacation)\b', re.I),
}

# Location extraction
LOCATION_BLOCKLIST = {
    'Star Wars', 'The', 'My', 'It', 'That', 'This', 'But', 'And',
    'What', 'How', 'Who', 'Where', 'When', 'Why', 'Yes', 'No',
    'Thanks', 'Thank', 'Good', 'Great', 'Nice', 'Cool', 'Wow',
}
LOCATION_RE = re.compile(r'(?:moving to|living in|moved to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})')

# Humor / emotion signals
HUMOR_RE  = re.compile(r'\b(lol|lmao|haha|hahaha|rofl|😂|🤣|funny|hilarious|joke)\b', re.I)
EMOJI_RE  = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0001F1E0-\U0001F1FF]')
EMOTION_RE = re.compile(r'\b(love|miss|hate|angry|sad|happy|excited|worried|anxious|scared|afraid|grateful|thankful|sorry)\b', re.I)
POSITIVE_RE = re.compile(r'\b(love|happy|excited|amazing|awesome|great|wonderful|fantastic|grateful|thankful)\b', re.I)
NEGATIVE_RE = re.compile(r'\b(hate|angry|sad|worried|anxious|scared|afraid|terrible|horrible|awful|frustrated)\b', re.I)


def count_pattern(texts, pattern):
    return sum(1 for t in texts if pattern.search(t))


def build_persona(messages_file='data_cache/messages.json', output_file='data_cache/persona.json'):
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("  Persona Builder")
    print("=" * 60)

    with open(messages_file, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    # Profile the top 2 dominant speakers dynamically
    speaker_counts = Counter(m['speaker'] for m in messages)
    top_speakers = [s for s, c in speaker_counts.most_common(2)]
    user_1_name = top_speakers[0] if top_speakers else "User 1"
    user_2_name = top_speakers[1] if len(top_speakers) > 1 else "User 2"

    user1_msgs = [m for m in messages if m['speaker'] == user_1_name]
    user2_msgs = [m for m in messages if m['speaker'] == user_2_name]
    all_texts_1 = [m['text'] for m in user1_msgs]
    all_texts_2 = [m['text'] for m in user2_msgs]

    print(f"  {user_1_name} messages: {len(user1_msgs)}")
    print(f"  {user_2_name} messages: {len(user2_msgs)}")

    def profile_user(texts, label):
        n = len(texts)
        if n == 0:
            return {}

        # ── Habits ─────────────────────────────────────────────────
        sleep = []
        if count_pattern(texts, SLEEP_LATE) > 2:
            sleep.append(f"late sleeper (mentioned ~{count_pattern(texts, SLEEP_LATE)} times)")
        if count_pattern(texts, SLEEP_EARLY) > 2:
            sleep.append(f"early riser (mentioned ~{count_pattern(texts, SLEEP_EARLY)} times)")

        food = {}
        for label_f, pat in FOOD_PATTERNS.items():
            c = count_pattern(texts, pat)
            if c > 2:
                food[label_f] = c
        food_list = [f"{k} ({v} mentions)" for k, v in sorted(food.items(), key=lambda x: -x[1])[:5]]

        exercise = {}
        for label_e, pat in EXERCISE_PATTERNS.items():
            c = count_pattern(texts, pat)
            if c > 1:
                exercise[label_e] = c
        exercise_list = [f"{k} ({v} mentions)" for k, v in sorted(exercise.items(), key=lambda x: -x[1])[:5]]

        hobbies = {}
        for label_h, pat in HOBBY_PATTERNS.items():
            c = count_pattern(texts, pat)
            if c > 2:
                hobbies[label_h] = c
        hobby_list = [f"{k} ({v} mentions)" for k, v in sorted(hobbies.items(), key=lambda x: -x[1])[:6]]

        # ── Personal Facts ─────────────────────────────────────────
        relationships = []
        for label_r, pat in REL_PATTERNS.items():
            c = count_pattern(texts, pat)
            if c > 0:
                relationships.append(f"{label_r} ({c} mentions)")

        locations = []
        for t in texts:
            for m in LOCATION_RE.finditer(t):
                loc = m.group(1).strip()
                if loc not in LOCATION_BLOCKLIST and len(loc) > 2:
                    locations.append(loc)
        locations = [f"{loc} ({cnt}x)" for loc, cnt
                     in Counter(locations).most_common(8)]

        events = []
        for label_ev, pat in EVENT_PATTERNS.items():
            c = count_pattern(texts, pat)
            if c > 0:
                events.append(f"{label_ev} ({c} mentions)")

        # ── Personality Traits (signal-based) ──────────────────────
        humor_count = count_pattern(texts, HUMOR_RE)
        emotion_count = count_pattern(texts, EMOTION_RE)
        positive_count = count_pattern(texts, POSITIVE_RE)
        negative_count = count_pattern(texts, NEGATIVE_RE)

        humor_level = "high" if humor_count / n > 0.05 else "moderate" if humor_count / n > 0.01 else "low"
        emotional_level = "high" if emotion_count / n > 0.1 else "moderate" if emotion_count / n > 0.03 else "reserved"
        sentiment = "mostly positive" if positive_count > negative_count * 2 else "balanced" if positive_count > negative_count else "often negative"

        # ── Communication Style ────────────────────────────────────
        total_words = sum(len(t.split()) for t in texts)
        avg_len = round(total_words / n, 1)
        question_count = sum(1 for t in texts if '?' in t)
        exclaim_count = sum(1 for t in texts if '!' in t)
        emoji_count = sum(len(EMOJI_RE.findall(t)) for t in texts)

        question_ratio = round(question_count / n, 3)
        exclaim_ratio = round(exclaim_count / n, 3)
        emoji_per_msg = round(emoji_count / n, 3)

        if avg_len < 6:
            msg_style = "very short / terse"
        elif avg_len < 12:
            msg_style = "concise"
        else:
            msg_style = "descriptive / verbose"

        if exclaim_ratio > 0.5:
            tone = "enthusiastic"
        elif question_ratio > 0.35:
            tone = "inquisitive"
        elif avg_len > 15:
            tone = "expressive / descriptive"
        else:
            tone = "casual"

        emoji_usage = "frequent" if emoji_per_msg > 0.1 else "occasional" if emoji_per_msg > 0.01 else "rare/none"

        return {
            "habits": {
                "sleep": sleep if sleep else ["no strong signal detected"],
                "food": food_list if food_list else ["no strong signal detected"],
                "exercise": exercise_list if exercise_list else ["no strong signal detected"],
                "hobbies_interests": hobby_list if hobby_list else ["no strong signal detected"],
            },
            "personal_facts": {
                "relationships": relationships if relationships else ["no explicit mentions"],
                "locations": locations if locations else ["no explicit mentions"],
                "events": events if events else ["no explicit mentions"],
            },
            "personality_traits": {
                "humor": f"{humor_level} ({humor_count} humor signals in {n} messages)",
                "emotional_expressiveness": f"{emotional_level} ({emotion_count} emotion words)",
                "sentiment": f"{sentiment} ({positive_count} positive, {negative_count} negative)",
                "directness": "high (short, to-the-point messages)" if avg_len < 8 else "moderate",
            },
            "communication_style": {
                "avg_words_per_message": avg_len,
                "message_style": msg_style,
                "tone": tone,
                "question_ratio": f"{round(question_ratio*100, 1)}% of messages are questions",
                "exclamation_ratio": f"{round(exclaim_ratio*100, 1)}% of messages use exclamation marks",
                "emoji_usage": emoji_usage,
                "emoji_per_message": emoji_per_msg,
                "total_messages_analysed": n,
            },
        }

    persona = {
        "user_1": profile_user(all_texts_1, user_1_name),
        "user_2": profile_user(all_texts_2, user_2_name),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(persona, f, indent=2)

    print(f"\n✓ Persona saved to {output_file}")
    # Quick preview
    for user_key in ["user_1", "user_2"]:
        p = persona[user_key]
        if not p:
            continue
        print(f"\n  {user_key.upper()} snapshot:")
        print(f"    Tone:      {p['communication_style']['tone']}")
        print(f"    Avg words: {p['communication_style']['avg_words_per_message']}")
        print(f"    Humor:     {p['personality_traits']['humor']}")
        print(f"    Emoji:     {p['communication_style']['emoji_usage']}")


if __name__ == "__main__":
    build_persona()
