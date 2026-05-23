import React from "react";
import BinaryStreamBackground from "./components/BinaryStreamBackground";
import ChatInterface from "./components/ChatInterface";
import NeuralArchives from "./components/NeuralArchives";
import KnowledgeWorkbench from "./components/KnowledgeWorkbench";
import { ThemeProvider } from "./context/ThemeContext";

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <main className="app-root">
        <BinaryStreamBackground />
        <div className="app-layout">
          <ChatInterface />
          <NeuralArchives />
          <KnowledgeWorkbench />
        </div>
      </main>
    </ThemeProvider>
  );
};

export default App;