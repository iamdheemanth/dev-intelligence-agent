export interface Project {
  project_id: string;
  status: string;
  meta?: {
    project_id?: string;
    chunks_indexed?: number;
    indexed_at?: string;
    [key: string]: any;
  };
  created_at?: string;
  repo_url?: string;
}

export interface RepoSearchResult {
  path?: string;
  file_path?: string;
  text?: string;
  content?: string;
  line_number?: number;
  score?: number;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

export interface User {
  id: string;
  email: string;
}
