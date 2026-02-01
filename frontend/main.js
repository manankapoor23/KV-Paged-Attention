/**
 * Main application logic with multi-chat support
 */

const chatHistory = [];
let currentSimulationIndex = -1;

function truncatePrompt(text, maxLen = 30) {
    return text.length > maxLen ? text.substring(0, maxLen) + "..." : text;
}

function renderChatList() {
    const list = document.getElementById("chat-list");
    list.innerHTML = "";

    chatHistory.forEach((sim, idx) => {
        const item = document.createElement("div");
        item.className = "chat-item";
        if (idx === currentSimulationIndex) item.classList.add("active");

        const label = document.createElement("div");
        label.style.flex = "1";
        label.textContent = truncatePrompt(sim.prompt);

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "delete-btn";
        deleteBtn.textContent = "×";
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            chatHistory.splice(idx, 1);
            if (currentSimulationIndex === idx) {
                currentSimulationIndex = chatHistory.length > 0 ? 0 : -1;
            } else if (currentSimulationIndex > idx) {
                currentSimulationIndex--;
            }
            renderChatList();
            loadSimulation(currentSimulationIndex);
        };

        item.appendChild(label);
        item.appendChild(deleteBtn);

        item.onclick = () => {
            loadSimulation(idx);
        };

        list.appendChild(item);
    });
}

function loadSimulation(idx) {
    if (idx < 0 || idx >= chatHistory.length) return;

    currentSimulationIndex = idx;
    const sim = chatHistory[idx];
    const vizArea = document.getElementById("viz-area");

    // Compute prefix with previous simulation if available
    let prefixTokens = [];
    if (idx > 0) {
        const prevTokens = chatHistory[idx - 1].data.tokens || [];
        const currTokens = sim.data.tokens || [];
        const minLen = Math.min(prevTokens.length, currTokens.length);
        for (let i = 0; i < minLen; i++) {
            if (prevTokens[i] === currTokens[i]) {
                prefixTokens.push(i);
            } else {
                break;
            }
        }
    }

    visualizer.setData(sim.data, prefixTokens);
    const timelineSlider = document.getElementById("timeline-slider");
    const timelineLabel = document.getElementById("timeline-label");
    timelineSlider.max = Math.max(0, sim.data.events.length - 1);
    timelineSlider.value = 0;
    timelineLabel.textContent = `Step 0 / ${Math.max(0, visualizer.maxSteps - 1)}`;

    const prefixInfo = prefixTokens.length > 0 ? ` (${prefixTokens.length} prefix from previous)` : "";
    document.getElementById("request-display").textContent = truncatePrompt(sim.prompt, 50) + prefixInfo;
    document.getElementById("event-count").textContent = `${sim.data.summary.total_events} events`;

    vizArea.style.display = "flex";
    visualizer.render(0);
    renderChatList();
}

document.addEventListener("DOMContentLoaded", async () => {
    const promptInput = document.getElementById("prompt-input");
    const runBtn = document.getElementById("run-btn");
    const statusDiv = document.getElementById("status");
    const timelineSlider = document.getElementById("timeline-slider");
    const timelineLabel = document.getElementById("timeline-label");
    const playBtn = document.getElementById("play-btn");
    const vizArea = document.getElementById("viz-area");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    let playTimer = null;

    function setPlayState(isPlaying) {
        if (!playBtn) return;
        playBtn.textContent = isPlaying ? "Pause" : "Play";
    }

    function stopPlaying() {
        if (playTimer) {
            clearInterval(playTimer);
            playTimer = null;
        }
        setPlayState(false);
    }

    // Check backend health
    async function checkBackend() {
        try {
            const isHealthy = await api.health();
            if (!isHealthy) {
                statusDiv.textContent =
                    "Backend not running. Start with: python -m uvicorn backend.main:app --reload";
                statusDiv.style.color = "red";
                runBtn.disabled = true;
            }
        } catch (e) {
            statusDiv.textContent =
                "Cannot connect to backend. Make sure it's running on http://localhost:8000";
            statusDiv.style.color = "orange";
            runBtn.disabled = true;
        }
    }

    await checkBackend();

    // Run simulation
    runBtn.addEventListener("click", async () => {
        stopPlaying();

        const prompt = promptInput.value.trim();
        if (!prompt) {
            statusDiv.textContent = "Please enter a prompt";
            statusDiv.style.color = "red";
            return;
        }

        statusDiv.textContent = "Running simulation...";
        statusDiv.style.color = "#9ca3af";
        runBtn.disabled = true;
        if (playBtn) playBtn.disabled = true;

        try {
            const result = await api.simulate(prompt);

            // Add to history
            chatHistory.push({ prompt, data: result });
            currentSimulationIndex = chatHistory.length - 1;

            // Load the new simulation
            loadSimulation(currentSimulationIndex);

            statusDiv.textContent = `Simulation complete: ${result.summary.total_events} events`;
            statusDiv.style.color = "#9ca3af";
        } catch (error) {
            statusDiv.textContent = `Error: ${error.message}`;
            statusDiv.style.color = "red";
            console.error(error);
        } finally {
            runBtn.disabled = false;
            if (playBtn) playBtn.disabled = false;
        }
    });

    // Clear history
    clearHistoryBtn.addEventListener("click", () => {
        if (confirm("Clear all simulations?")) {
            chatHistory.length = 0;
            currentSimulationIndex = -1;
            vizArea.style.display = "none";
            renderChatList();
            statusDiv.textContent = "History cleared";
        }
    });

    // Timeline slider
    timelineSlider.addEventListener("input", (e) => {
        stopPlaying();
        const step = parseInt(e.target.value);
        timelineLabel.textContent = `Step ${step} / ${Math.max(0, visualizer.maxSteps - 1)}`;
        visualizer.render(step);
    });

    // Play / Pause
    if (playBtn) {
        playBtn.addEventListener("click", () => {
            if (!visualizer.maxSteps || visualizer.maxSteps <= 1) return;

            if (playTimer) {
                stopPlaying();
                return;
            }

            setPlayState(true);
            playTimer = setInterval(() => {
                const max = Math.max(0, visualizer.maxSteps - 1);
                const current = parseInt(timelineSlider.value);
                const next = Math.min(max, current + 1);

                timelineSlider.value = next;
                timelineLabel.textContent = `Step ${next} / ${max}`;
                visualizer.render(next);

                if (next >= max) {
                    stopPlaying();
                }
            }, 320);
        });
    }

    // Enter key to run
    promptInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            runBtn.click();
        }
    });

    // Initial chat list render
    renderChatList();
});
