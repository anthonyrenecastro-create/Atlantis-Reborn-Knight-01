export type WindowType =
  | "CHAT"
  | "UNIFIED_CHAT"
  | "FIELD_PANEL"
  | "TRAINING_DASHBOARD"
  | "IMAGE_GENERATOR"
  | "VIDEO_GENERATOR"
  | "BROWSER"
  | "DOCUMENT_VIEWER";

export interface BaseWindow {
  id: number;
  type: WindowType;
  title: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  zIndex: number;
}

export interface ChatWindow extends BaseWindow {
  type: "CHAT";
}

export interface ImageGeneratorWindow extends BaseWindow {
  type: "IMAGE_GENERATOR";
  initialPrompt?: string;
}

export interface UnifiedChatWindow extends BaseWindow {
  type: "UNIFIED_CHAT";
}

export interface FieldPanelWindow extends BaseWindow {
  type: "FIELD_PANEL";
}

export interface TrainingDashboardWindow extends BaseWindow {
  type: "TRAINING_DASHBOARD";
}

export interface VideoGeneratorWindow extends BaseWindow {
  type: "VIDEO_GENERATOR";
  initialPrompt?: string;
}

export interface BrowserWindow extends BaseWindow {
  type: "BROWSER";
  initialQuery: string;
}

export interface DocumentViewerWindow extends BaseWindow {
  type: "DOCUMENT_VIEWER";
  file: File;
}

export type WindowInstance =
  | ChatWindow
  | UnifiedChatWindow
  | FieldPanelWindow
  | TrainingDashboardWindow
  | ImageGeneratorWindow
  | VideoGeneratorWindow
  | BrowserWindow
  | DocumentViewerWindow;

export interface ChatMessage {
  id: number;
  sender: "user" | "aura";
  text: string;
}

export type TaskStatus = "Running" | "Queued" | "Complete" | "Error";

export interface Task {
  id: number;
  name: string;
  status: TaskStatus;
}

export enum Screen {
  COGNITIVE_DASHBOARD = "COGNITIVE_DASHBOARD",
  COMMAND_CONSOLE = "COMMAND_CONSOLE",
  TOOLS_PANEL = "TOOLS_PANEL",
}

export type FeedItemType = "info" | "summary" | "error" | "autonomous";

export interface FeedItem {
  id: number;
  type: FeedItemType;
  text: string;
  sources?: { uri: string; title: string }[];
}
