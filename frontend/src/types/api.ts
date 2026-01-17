export interface DiseaseResponse {
  predicted_disease_en: string;
  predicted_disease_translated: string;
  target_language: string;
}

export interface RagResponse {
  answer: string;
  sources?: string[];
}

export interface MarketResponse {
  response: string;
  raw_data: any;
}

export interface TranscriptionResponse {
  transcription: string;
}