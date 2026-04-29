import re
import random
import nltk
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words("english"))

NOISE = {
    "i", "we", "it", "its", "they", "he", "she", "this", "that", "these",
    "those", "their", "our", "us", "you", "your", "my", "which", "who",
    "what", "where", "when", "how", "why", "example", "following", "given",
    "however", "therefore", "thus", "hence", "also", "well", "still", "even",
    "just", "like", "much", "many", "more", "most", "such", "same", "other",
    "another", "both", "each", "every", "any", "all", "few", "some",
    "simple", "words", "short", "long", "good", "bad", "new", "old",
    "certain", "general", "common", "basic", "main", "first", "last",
    "means", "way", "ways", "form", "forms", "terms", "term",
    "concept", "concepts", "idea", "ideas", "type", "types", "kind",
    "number", "part", "case", "note", "step", "thing", "things"
}

SKIP_LABELS = {"PERSON", "GPE", "LOC", "DATE", "TIME", "ORDINAL", "CARDINAL"}


# ══════════════════════════════════════════════════════════════
# STEP 1 — CLEAN TEXT
# ══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove URLs and emails
        line = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', line)
        
        line = re.sub(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', '', line)

        # Remove unicode box/bullet characters that browsers can't render
        line = re.sub(r'[^\x00-\x7F]+', ' ', line)
        # Skip professor/author/meta lines
        if re.search(r'\b(dr|prof|mr|mrs|ms|sr|jr|phd|mtech|btech'
                     r'|assistant professor|associate professor'
                     r'|prepared by|presented by|authored by)\b',
                     line, re.IGNORECASE):
            continue
        
        # Skip lines that are purely numbers/symbols
        if re.match(r'^[\W\d\s]+$', line):
            continue

        # Skip single character lines
        if len(line.strip()) <= 1:
            continue
        
        # Fix broken words from PDF (single letters with spaces like "Y ou", "T he")
        line = re.sub(r'(?<![a-zA-Z])([A-Z])\s+(?=[a-z])', r'\1', line)

        # Remove standalone special characters (keep c++, pointer*, .NET)
        line = re.sub(r'(?<!\w)[^\w\s](?!\w)', ' ', line)
        # Strip bullet characters from start of line only
        line = re.sub(r'^[\•\-\*\✓\➢\►\▪\◦\–]+\s*', '', line)

        # Remove single standalone characters
        line = re.sub(r'\b\w\b', ' ', line)

        # Remove stopwords
        words = line.split()
        words = [w for w in words if w.lower() not in STOP]
        line = ' '.join(words)

        # Collapse spaces
        line = re.sub(r' +', ' ', line).strip()

        if not line:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ══════════════════════════════════════════════════════════════
# STEP 2 — SENTENCE TOKENIZATION
# ══════════════════════════════════════════════════════════════
def get_sentences(text: str) -> list:
    # Use original text (before stopword removal) for sentences
    # so they read naturally as questions
    sentences = sent_tokenize(text)
    result = []
    for s in sentences:
        s = s.strip()
        # Remove dates from sentence
        s = re.sub(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', '', s)
        s = s.strip()
        words = s.split()
        if not (4 <= len(words) <= 100):
            continue
        if not re.search(r'[a-zA-Z]{3,}', s):
            continue
        # Skip meta/name lines
        if re.search(r'\b(dr|prof|mr|mrs|ms|sr|jr|phd'
                     r'|assistant professor|associate professor'
                     r'|prepared by|presented by)\b',
                     s, re.IGNORECASE):
            continue
        # Skip code lines
        if re.search(r'\b(import|public|private|class |void|static)\b.*[;{}]', s):
            continue
        result.append(s)
    return result


# ══════════════════════════════════════════════════════════════
# STEP 3 — LDA TOPIC MODELING
# Finds main topics in the document automatically
# Returns: list of (topic_name, [top sentences for this topic])
# ══════════════════════════════════════════════════════════════
def extract_topics_and_sentences(raw_text: str, sentences: list,
                                  num_topics: int = 5,
                                  sentences_per_topic: int = 8):
    """
    Uses LDA to find the main topics in the document.
    Then maps each sentence to its dominant topic.
    Returns sentences grouped by topic, each topic labelled
    with its most representative keyword.
    """

    # --- Prepare cleaned tokens for LDA ---
    # LDA needs tokenized words, no stopwords
    def tokenize_for_lda(text):
        tokens = simple_preprocess(text, deacc=True)  # lowercase, remove accents
        tokens = [t for t in tokens
                  if t not in STOP
                  and t not in NOISE
                  and len(t) > 2]
        return tokens

    # Use cleaned text lines as LDA documents
    cleaned = clean_text(raw_text)
    lda_docs = [tokenize_for_lda(line)
                for line in cleaned.split("\n")
                if line.strip()]
    lda_docs = [d for d in lda_docs if len(d) >= 3]

    if len(lda_docs) < num_topics:
        # Not enough content for LDA — fall back to TF-IDF ranking
        return _tfidf_fallback(sentences, num_topics, sentences_per_topic)

    # --- Build dictionary and corpus ---
    dictionary = corpora.Dictionary(lda_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in lda_docs]

    if not any(corpus):
        return _tfidf_fallback(sentences, num_topics, sentences_per_topic)

    # --- Train LDA ---
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,          # more passes = better topics
        iterations=100,
        random_state=42,
        alpha='auto',
        eta='auto'
    )

    # --- Get topic name = top 2 keywords of each topic ---
    topic_names = {}
    for i in range(num_topics):
        top_words = lda_model.show_topic(i, topn=2)
        name = " & ".join([w for w, _ in top_words]).title()
        topic_names[i] = name

    # --- Assign each sentence to its dominant topic ---
    topic_sentences = {i: [] for i in range(num_topics)}

    for sent in sentences:
        tokens = tokenize_for_lda(sent)
        if not tokens:
            continue
        bow = dictionary.doc2bow(tokens)
        if not bow:
            continue
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_sentences[dominant_topic].append((sent, topic_dist[dominant_topic][1]))

    # --- For each topic, return top N sentences by probability ---
    result = []
    for topic_id in range(num_topics):
        sents = topic_sentences[topic_id]
        if not sents:
            continue
        # Sort by topic probability — highest first
        sents.sort(key=lambda x: x[1], reverse=True)
        top_sents = [s for s, _ in sents[:sentences_per_topic]]
        result.append({
            "topic": topic_names[topic_id],
            "sentences": top_sents
        })

    return result


# ══════════════════════════════════════════════════════════════
# FALLBACK — TF-IDF sentence ranking if LDA can't run
# ══════════════════════════════════════════════════════════════
def _tfidf_fallback(sentences: list, num_topics: int,
                    sentences_per_topic: int) -> list:
    if not sentences:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).flatten()
        ranked = [s for _, s in sorted(zip(scores, sentences), reverse=True)]
        # Group into fake topics
        chunk = sentences_per_topic
        result = []
        for i in range(num_topics):
            chunk_sents = ranked[i * chunk:(i + 1) * chunk]
            if chunk_sents:
                result.append({"topic": f"Topic {i+1}", "sentences": chunk_sents})
        return result
    except Exception:
        return [{"topic": "General", "sentences": sentences}]


# ══════════════════════════════════════════════════════════════
# STEP 4 — TF-IDF SCORES FOR WHOLE DOCUMENT
# ══════════════════════════════════════════════════════════════
def get_tfidf_scores(sentences: list) -> dict:
    if len(sentences) < 2:
        return {}
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(sentences)
        names = vec.get_feature_names_out()
        scores = np.asarray(matrix.max(axis=0)).flatten()
        return {n: float(s) for n, s in zip(names, scores)}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# STEP 5 — VALIDATE WORD
# ══════════════════════════════════════════════════════════════
def is_valid(word: str) -> bool:
    if not word:
        return False
    w = word.lower().strip()
    if len(w) < 3:
        return False
    if len(word.split()) > 4:
        return False
    if w in STOP or w in NOISE:
        return False
    if not re.search(r'[a-zA-Z]{2,}', word):
        return False
    if re.search(r'[{};=<>()#\\@$]', word):
        return False
    if re.search(r'\b(dr|prof|mr|mrs|ms|sr|jr|phd)\b', word, re.IGNORECASE):
        return False
    return True


def clean_phrase(text: str) -> str:
    return re.sub(
        r'^(a|an|the|this|that|these|those|its|their|our|my|your)\s+',
        '', text.strip(), flags=re.IGNORECASE
    ).strip()


# ══════════════════════════════════════════════════════════════
# STEP 6 — EXTRACT KEYWORD FROM SENTENCE
# ══════════════════════════════════════════════════════════════
def extract_keyword(sentence: str, tfidf_scores: dict) -> str | None:
    doc = nlp(sentence)
    candidates = []

    for ent in doc.ents:
        if ent.label_ in SKIP_LABELS:
            continue
        w = ent.text.strip()
        if is_valid(w):
            score = tfidf_scores.get(w.lower(), 0.001)
            candidates.append((score + 2.0, w))

    for chunk in doc.noun_chunks:
        w = clean_phrase(chunk.text)
        if is_valid(w):
            score = tfidf_scores.get(w.lower(), 0.001) + \
                    tfidf_scores.get(chunk.root.lemma_.lower(), 0.001)
            candidates.append((score + 1.0, w))

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            w = token.text.strip()
            if is_valid(w):
                score = tfidf_scores.get(w.lower(), 0.001) + \
                        tfidf_scores.get(token.lemma_.lower(), 0.001)
                candidates.append((score, w))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ══════════════════════════════════════════════════════════════
# STEP 7 — BUILD DISTRACTOR POOL
# ══════════════════════════════════════════════════════════════
def build_noun_frequency_map(raw_text: str) -> dict:
    """
    POS tagging on every line.
    Counts how many lines each NOUN/PROPN appears in.
    Only keeps nouns that appear in actual content lines (not just headings).
    """
    lines = raw_text.split("\n")
    heading_words = set()
    content_word_lines = {}  # word -> set of line indices it appears in

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        doc = nlp(line)
        is_heading = len(line.split()) <= 6  # short lines = likely headings

        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                w = token.lemma_.lower().strip()
                if not is_valid(w):
                    continue
                if is_heading:
                    heading_words.add(w)
                else:
                    if w not in content_word_lines:
                        content_word_lines[w] = set()
                    content_word_lines[w].add(i)

    # Only keep words that appear in content lines
    # Heading words are included ONLY if they also appear in content
    freq_map = {}
    for word, line_set in content_word_lines.items():
        freq_map[word] = len(line_set)

    # Sort by frequency
    return dict(sorted(freq_map.items(), key=lambda x: x[1], reverse=True))

def get_distractors(answer: str, pool: list, n: int = 3) -> list:
    al = answer.lower()
    candidates = [
        w for w in pool
        if w.lower() != al
        and al not in w.lower()
        and w.lower() not in al
        and is_valid(w)
    ]
    random.shuffle(candidates)
    return candidates[:n]


# ══════════════════════════════════════════════════════════════
# STEP 8 — MAIN
# ══════════════════════════════════════════════════════════════
def generate_mcqs(raw_text: str, num_questions: int = 15) -> list:

    # 1. Get sentences from original text (before stopword removal)
    #    so questions read naturally
    sentences = get_sentences(raw_text)
    if len(sentences) < 5:
        sentences = [s.strip() for s in sent_tokenize(raw_text)
                     if len(s.split()) >= 4]
    if not sentences:
        return []

    # 2. TF-IDF scores
    tfidf_scores = get_tfidf_scores(sentences)

    # 3. LDA — find topics and best sentences per topic
    #    num_topics scales with document size
    num_topics = min(5, max(2, len(sentences) // 10))
    sentences_per_topic = max(6, (num_questions // num_topics) + 3)
    topic_groups = extract_topics_and_sentences(
        raw_text, sentences,
        num_topics=num_topics,
        sentences_per_topic=sentences_per_topic
    )

    # 4. Build noun frequency map
    freq_map = build_noun_frequency_map(raw_text)
    
    # Top nouns from freq map = our answer candidates + distractor pool
    pool = [w for w, count in freq_map.items() if count >= 1]

    # 5. Build MCQs — take proportionally from each topic
    mcqs = []
    used_answers = set()
    used_sents = set()
    qs_per_topic = max(1, num_questions // len(topic_groups))

    for group in topic_groups:
        topic_name = group["topic"]
        topic_sents = group["sentences"]
        count = 0

        for sent in topic_sents:
            if count >= qs_per_topic:
                break
            if len(mcqs) >= num_questions:
                break

            sid = sent[:50]
            if sid in used_sents:
                continue

            keyword = None
            sent_doc = nlp(sent)
            sent_nouns = []
            for token in sent_doc:
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                    w = token.lemma_.lower()
                    if w in freq_map and is_valid(token.text):
                        sent_nouns.append((freq_map[w], token.text))
            if sent_nouns:
                sent_nouns.sort(key=lambda x: x[0], reverse=True)
                keyword = sent_nouns[0][1]
            if not keyword or keyword.lower() in used_answers:
                continue

            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            if not pattern.search(sent):
                continue

            question_text = pattern.sub("________", sent, count=1)

            distractors = get_distractors(keyword, pool, n=3)
            if len(distractors) < 3:
                extras = [w for w in pool
                          if w.lower() != keyword.lower()
                          and w not in distractors][:3 - len(distractors)]
                distractors += extras
            if len(distractors) < 3:
                continue

            options = [keyword] + distractors[:3]
            random.shuffle(options)
            correct = next((o for o in options if o.lower() == keyword.lower()), keyword)

            mcqs.append({
                "question": question_text,
                "options":  options,
                "answer":   correct,
                "topic":    topic_name
            })

            used_answers.add(keyword.lower())
            used_sents.add(sid)
            count += 1

    # Fill remaining slots if any topic didn't produce enough
    if len(mcqs) < num_questions:
        for group in topic_groups:
            for sent in group["sentences"]:
                if len(mcqs) >= num_questions:
                    break
                sid = sent[:50]
                if sid in used_sents:
                    continue
                keyword = None
                sent_doc = nlp(sent)
                sent_nouns = []
                for token in sent_doc:
                    if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                        w = token.lemma_.lower()
                        if w in freq_map and is_valid(token.text):
                            sent_nouns.append((freq_map[w], token.text))
                if sent_nouns:
                    sent_nouns.sort(key=lambda x: x[0], reverse=True)
                    keyword = sent_nouns[0][1]
                if not keyword or keyword.lower() in used_answers:
                    continue
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                if not pattern.search(sent):
                    continue
                question_text = pattern.sub("________", sent, count=1)
                distractors = get_distractors(keyword, pool, n=3)
                if len(distractors) < 3:
                    extras = [w for w in pool
                              if w.lower() != keyword.lower()
                              and w not in distractors][:3 - len(distractors)]
                    distractors += extras
                if len(distractors) < 3:
                    continue
                options = [keyword] + distractors[:3]
                random.shuffle(options)
                correct = next((o for o in options if o.lower() == keyword.lower()), keyword)
                mcqs.append({
                    "question": question_text,
                    "options":  options,
                    "answer":   correct,
                    "topic":    group["topic"]
                })
                used_answers.add(keyword.lower())
                used_sents.add(sid)

    random.shuffle(mcqs)
    return mcqs[:num_questions]