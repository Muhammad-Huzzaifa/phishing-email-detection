const emailInput = document.getElementById("emailInput");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const resultsBody = document.getElementById("resultsBody");

const finalLabel = document.getElementById("finalLabel");
const voteBreakdown = document.getElementById("voteBreakdown");
const avgProbability = document.getElementById("avgProbability");
const totalInference = document.getElementById("totalInference");
const consensusCard = document.getElementById("consensusCard");

function setStatus(text) {
  statusText.textContent = text;
}

function resetTable() {
  resultsBody.innerHTML = `
    <tr>
      <td colspan="4" class="placeholder">No prediction yet.</td>
    </tr>
  `;
}

function resetConsensus() {
  finalLabel.textContent = "Final Label: -";
  voteBreakdown.textContent = "Votes: -";
  avgProbability.textContent = "Average Probability: -";
  totalInference.textContent = "Total Inference Time: -";
  consensusCard.classList.remove("safe", "phishing");
  consensusCard.classList.add("neutral");
}

function renderResults(predictions) {
  const rows = Object.entries(predictions)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([model, data]) => {
      return `
        <tr>
          <td>${model}</td>
          <td>${data.predicted_label}</td>
          <td>${Number(data.probability).toFixed(4)}</td>
          <td>${Number(data.inference_time_ms).toFixed(3)}</td>
        </tr>
      `;
    })
    .join("");

  resultsBody.innerHTML = rows;
}

function renderConsensus(consensus) {
  finalLabel.textContent = `Final Label: ${consensus.final_label}`;
  voteBreakdown.textContent = `Votes: Phishing ${consensus.phishing_votes} | Safe ${consensus.safe_votes}`;
  avgProbability.textContent = `Average Probability: ${Number(consensus.average_probability).toFixed(4)}`;
  totalInference.textContent = `Total Inference Time: ${Number(consensus.total_inference_time_ms).toFixed(3)} ms`;

  consensusCard.classList.remove("neutral", "safe", "phishing");
  consensusCard.classList.add(consensus.final_label === "Phishing" ? "phishing" : "safe");
}

async function runPrediction() {
  const text = emailInput.value.trim();
  if (!text) {
    setStatus("Please enter email text.");
    return;
  }

  predictBtn.disabled = true;
  setStatus("Running inference across all models...");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email_text: text }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    renderResults(payload.predictions);
    renderConsensus(payload.consensus);
    setStatus("Done. Prediction complete.");
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    predictBtn.disabled = false;
  }
}

predictBtn.addEventListener("click", runPrediction);

clearBtn.addEventListener("click", () => {
  emailInput.value = "";
  resetTable();
  resetConsensus();
  setStatus("Cleared.");
});

resetTable();
resetConsensus();
