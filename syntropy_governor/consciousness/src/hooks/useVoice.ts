import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type SpeechRecognitionEventLike = Event & {
  results: ArrayLike<{
    isFinal: boolean;
    length: number;
    item: (index: number) => { transcript: string };
    [index: number]: { transcript: string };
  }>;
};

type SpeechRecognitionLike = EventTarget & {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type VoiceApi = {
  supported: boolean;
  listening: boolean;
  speaking: boolean;
  transcript: string;
  error: string | null;
  startListening: () => void;
  stopListening: () => void;
  speak: (text: string) => void;
  stopSpeaking: () => void;
  clearTranscript: () => void;
};

function getSpeechRecognitionCtor(): SpeechRecognitionConstructor | null {
  const w = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function useVoice(): VoiceApi {
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  const supported = useMemo(
    () => typeof window !== "undefined" && !!getSpeechRecognitionCtor() && !!window.speechSynthesis,
    [],
  );

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      window.speechSynthesis.cancel();
    };
  }, []);

  const startListening = useCallback(() => {
    setError(null);
    if (!supported) {
      setError("Speech recognition not supported in this browser.");
      return;
    }

    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      setError("Speech recognition unavailable.");
      return;
    }

    if (!recognitionRef.current) {
      recognitionRef.current = new Ctor();
    }

    const recognition = recognitionRef.current;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const parts: string[] = [];
      for (let i = 0; i < event.results.length; i += 1) {
        const alt = event.results[i]?.[0] ?? event.results[i]?.item(0);
        if (alt?.transcript) {
          parts.push(alt.transcript);
        }
      }
      setTranscript(parts.join(" ").trim());
    };

    recognition.onerror = () => {
      setError("Voice capture failed. Please try again.");
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
    };

    try {
      recognition.start();
      setListening(true);
    } catch {
      setError("Unable to start voice capture.");
      setListening(false);
    }
  }, [supported]);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setListening(false);
  }, []);

  const speak = useCallback((text: string) => {
    if (!text.trim()) {
      return;
    }
    if (!window.speechSynthesis) {
      setError("Speech synthesis not supported in this browser.");
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => {
      setSpeaking(false);
      setError("Voice playback failed.");
    };
    window.speechSynthesis.speak(utterance);
  }, []);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, []);

  const clearTranscript = useCallback(() => {
    setTranscript("");
  }, []);

  return {
    supported,
    listening,
    speaking,
    transcript,
    error,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    clearTranscript,
  };
}
