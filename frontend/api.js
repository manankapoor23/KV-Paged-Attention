/**
 * API client for backend communication
 */

const API_BASE = "http://localhost:8000";

class APIClient {
    async simulate(prompt) {
        try {
            const response = await fetch(`${API_BASE}/simulate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt }),
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error("API request failed:", error);
            throw error;
        }
    }

    async health() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            return response.ok;
        } catch {
            return false;
        }
    }
}

const api = new APIClient();
