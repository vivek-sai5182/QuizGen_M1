// ── STATE ──
let selectedFile = null;
let allQuestions = [];
let currentIndex = 0;
let score = 0;
let answered = false;
let userAnswers = [];  // {topic, correct: bool}

// ── ELEMENTS ──
const screens = {
    upload:  document.getElementById("screen-upload"),
    loading: document.getElementById("screen-loading"),
    quiz:    document.getElementById("screen-quiz"),
    result:  document.getElementById("screen-result")
};

function showScreen(name) {
    Object.entries(screens).forEach(([k, el]) => {
        el.classList.toggle("active", k === name);
    });
}

// ── UPLOAD LOGIC ──
const dropArea    = document.getElementById("drop-area");
const fileInput   = document.getElementById("fileInput");
const browseBtn   = document.getElementById("browseBtn");
const fileInfo    = document.getElementById("file-info");
const fileNameTxt = document.getElementById("file-name-text");
const removeFile  = document.getElementById("removeFile");
const submitBtn   = document.getElementById("submitBtn");
const errorMsg    = document.getElementById("error-msg");
const numQSelect  = document.getElementById("numQuestions");

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => handleFile(e.target.files[0]));

["dragenter","dragover","dragleave","drop"].forEach(ev =>
    dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
);
["dragenter","dragover"].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.add("highlight"))
);
["dragleave","drop"].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.remove("highlight"))
);
dropArea.addEventListener("drop", e => handleFile(e.dataTransfer.files[0]));
dropArea.addEventListener("click", e => {
    if (e.target === browseBtn) return;
    fileInput.click();
});

removeFile.addEventListener("click", e => {
    e.stopPropagation();
    selectedFile = null;
    fileInfo.classList.add("hidden");
    dropArea.style.display = "flex";
    submitBtn.disabled = true;
    fileInput.value = "";
});

function handleFile(file) {
    if (!file) return;
    const allowed = ["pdf","docx","pptx","txt"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showError("Unsupported file type. Please upload PDF, DOCX, PPTX, or TXT.");
        return;
    }
    selectedFile = file;
    fileNameTxt.textContent = file.name;
    fileInfo.classList.remove("hidden");
    dropArea.style.display = "none";
    submitBtn.disabled = false;
    hideError();
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove("hidden");
}
function hideError() {
    errorMsg.classList.add("hidden");
}

// ── SUBMIT ──
submitBtn.addEventListener("click", () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("num_questions", numQSelect.value);

    showScreen("loading");
    animateLoadingSteps();

    fetch("/generate", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showScreen("upload");
                showError("Error: " + data.error);
                return;
            }
            allQuestions = data.questions;
            currentIndex = 0;
            score = 0;
            userAnswers = [];
            setTimeout(() => startQuiz(), 800);
        })
        .catch(() => {
            showScreen("upload");
            showError("Server error. Make sure the Flask server is running.");
        });
});

// ── LOADING ANIMATION ──
function animateLoadingSteps() {
    const steps = ["step-1","step-2","step-3","step-4","step-5"];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove("active","done");
    });
    let i = 0;
    const interval = setInterval(() => {
        if (i > 0) {
            document.getElementById(steps[i-1]).classList.remove("active");
            document.getElementById(steps[i-1]).classList.add("done");
        }
        if (i < steps.length) {
            document.getElementById(steps[i]).classList.add("active");
            i++;
        } else {
            clearInterval(interval);
        }
    }, 700);
}

// ── QUIZ LOGIC ──
const questionText = document.getElementById("question-text");
const optionsGrid  = document.getElementById("options-grid");
const nextBtn      = document.getElementById("nextBtn");
const progressBar  = document.getElementById("progress-bar");
const qCounter     = document.getElementById("q-counter");
const topicBadge   = document.getElementById("topic-badge");

function startQuiz() {
    showScreen("quiz");
    renderQuestion();
}

function renderQuestion() {
    answered = false;
    nextBtn.disabled = true;

    const q = allQuestions[currentIndex];
    const total = allQuestions.length;

    // Progress
    progressBar.style.width = `${(currentIndex / total) * 100}%`;
    qCounter.textContent = `${currentIndex + 1} / ${total}`;
    topicBadge.textContent = q.topic || "General";

    // Question
    questionText.textContent = q.question;

    // Options
    optionsGrid.innerHTML = "";
    const labels = ["A", "B", "C", "D"];
    q.options.forEach((opt, i) => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.innerHTML = `<strong>${labels[i]}.</strong> ${opt}`;
        btn.dataset.value = opt;
        btn.addEventListener("click", () => selectOption(btn, q.answer, q.topic));
        optionsGrid.appendChild(btn);
    });

    // Animate card
    const card = document.getElementById("question-card");
    card.style.animation = "none";
    card.offsetHeight; // reflow
    card.style.animation = "fadeUp 0.3s ease";
}

function selectOption(selected, correctAnswer, topic) {
    if (answered) return;
    answered = true;

    const allBtns = optionsGrid.querySelectorAll(".option-btn");
    const isCorrect = selected.dataset.value === correctAnswer;

    allBtns.forEach(btn => {
        btn.disabled = true;
        if (btn.dataset.value === correctAnswer) {
            btn.classList.add("correct");
        }
    });

    if (!isCorrect) {
        selected.classList.add("wrong");
    } else {
        score++;
    }

    userAnswers.push({ topic: topic || "General", correct: isCorrect });
    nextBtn.disabled = false;
}

nextBtn.addEventListener("click", () => {
    currentIndex++;
    if (currentIndex < allQuestions.length) {
        renderQuestion();
    } else {
        showResult();
    }
});

// ── RESULT ──
const scoreNum    = document.getElementById("score-num");
const scoreDenom  = document.getElementById("score-denom");
const resultTitle = document.getElementById("result-title");
const resultSub   = document.getElementById("result-sub");
const ringFill    = document.getElementById("ring-fill");
const topicBreakdown = document.getElementById("topic-breakdown");

function showResult() {
    showScreen("result");

    const total = allQuestions.length;
    const pct = score / total;

    // Score ring (circumference = 2π×50 ≈ 314)
    const circumference = 314;
    const offset = circumference - pct * circumference;
    setTimeout(() => { ringFill.style.strokeDashoffset = offset; }, 100);

    scoreNum.textContent = score;
    scoreDenom.textContent = `/ ${total}`;

    if (pct >= 0.8) {
        resultTitle.textContent = "Excellent Work! 🎉";
        resultSub.textContent = "You have a strong grasp of the material.";
    } else if (pct >= 0.5) {
        resultTitle.textContent = "Good Effort! 👍";
        resultSub.textContent = "Review the topics you missed and try again.";
    } else {
        resultTitle.textContent = "Keep Studying 📚";
        resultSub.textContent = "Read through the material again before retrying.";
    }

    // Topic breakdown
    const topicStats = {};
    userAnswers.forEach(({ topic, correct }) => {
        if (!topicStats[topic]) topicStats[topic] = { correct: 0, total: 0 };
        topicStats[topic].total++;
        if (correct) topicStats[topic].correct++;
    });

    topicBreakdown.innerHTML = "<strong style='font-size:13px;color:var(--text-muted);'>Topic Breakdown</strong>";
    Object.entries(topicStats).forEach(([topic, stat]) => {
        const pctTopic = stat.correct / stat.total;
        const row = document.createElement("div");
        row.className = "topic-row";
        row.innerHTML = `
            <span class="topic-name">${topic}</span>
            <div class="topic-bar-wrap">
                <div class="topic-bar-fill" style="width:0%"></div>
            </div>
            <span class="topic-score-label">${stat.correct}/${stat.total}</span>
        `;
        topicBreakdown.appendChild(row);
        setTimeout(() => {
            row.querySelector(".topic-bar-fill").style.width = `${pctTopic * 100}%`;
        }, 200);
    });

    progressBar.style.width = "100%";
}

// ── RESTART ──
document.getElementById("retryBtn").addEventListener("click", () => {
    currentIndex = 0;
    score = 0;
    userAnswers = [];
    startQuiz();
});

document.getElementById("newFileBtn").addEventListener("click", () => {
    selectedFile = null;
    fileInfo.classList.add("hidden");
    dropArea.style.display = "flex";
    submitBtn.disabled = true;
    fileInput.value = "";
    hideError();
    showScreen("upload");
});