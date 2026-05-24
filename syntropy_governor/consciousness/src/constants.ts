import { Task } from "./types";

export const INITIAL_TASKS: Task[] = [
  { id: 1, name: "Crawling URL: /wiki/Quantum_Entanglement", status: "Running" },
  { id: 2, name: "Processing Document: 'aetherium_whitepaper.pdf'", status: "Queued" },
  { id: 3, name: "Evolving Concepts: Cellular Automata", status: "Queued" },
  { id: 4, name: "Reticulating Splines...", status: "Complete" },
];

export const generateChartData = () => {
  const data: { time: string; stability: number; variance: number; chaos: number }[] = [];
  for (let i = 30; i >= 0; i -= 1) {
    data.push({
      time: `-${i}s`,
      stability: Math.random() * 0.2 + 0.75,
      variance: Math.random() * 0.1 + 0.05,
      chaos: Math.random() * 0.1,
    });
  }
  return data;
};
