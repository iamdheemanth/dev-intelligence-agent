import { supabase } from './supabase';
import type { Project, RepoSearchResult } from '../types';

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;
const API_TOKEN = import.meta.env.VITE_API_TOKEN;

async function getAuthHeaders(): Promise<HeadersInit> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
  };

  // Add Bearer token if available
  if (API_TOKEN) {
    headers['Authorization'] = `Bearer ${API_TOKEN}`;
  }

  return headers;
}

async function getAuthHeadersMultipart(): Promise<HeadersInit> {
  const headers: HeadersInit = {
    'ngrok-skip-browser-warning': 'true',
  };

  // Add Bearer token if available
  if (API_TOKEN) {
    headers['Authorization'] = `Bearer ${API_TOKEN}`;
  }

  return headers;
}

export const api = {
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async getProjects(): Promise<Project[]> {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/projects`, { headers });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Get projects failed:', response.status, errorText);
        throw new Error(`Failed to fetch projects: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Backend returns {projects: [...]} not just [...]
      const projects = data.projects || data;
      
      // Ensure we return an array
      if (!Array.isArray(projects)) {
        console.error('Projects API returned non-array:', data);
        return [];
      }
      
      return projects;
    } catch (error: any) {
      console.error('Get projects error:', error);
      throw error;
    }
  },

  async createProject(gitUrl: string): Promise<Project> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ git_url: gitUrl })
    });
    if (!response.ok) throw new Error(`Failed to create project: ${response.status}`);
    return response.json();
  },

  async uploadProject(file: File): Promise<Project> {
    const headers = await getAuthHeadersMultipart();
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/projects/upload`, {
      method: 'POST',
      headers,
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Failed to upload project: ${response.status}`);
    }
    
    return response.json();
  },

  async repoSearch(projectId: string, query: string): Promise<RepoSearchResult[]> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/repo_search`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query })
    });
    if (!response.ok) throw new Error('Failed to search repository');
    return response.json();
  },

  async summarizeProject(projectId: string, topic: string = "project overview"): Promise<any> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/summarize`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ topic })
    });
    if (!response.ok) throw new Error('Failed to summarize project');
    return response.json();
  },

  async recommendImprovements(projectId: string, query: string = "recommend improvements"): Promise<any> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/recommend`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query, top_k: 5 })
    });
    if (!response.ok) throw new Error('Failed to get recommendations');
    return response.json();
  },

  async triageIssues(projectId: string, title: string = "Code review", body: string = ""): Promise<any> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/triage`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ title, body })
    });
    if (!response.ok) throw new Error('Failed to triage issues');
    return response.json();
  },

  async reviewCode(projectId: string, path?: string, code?: string): Promise<any> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/review`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ 
        path: path || null,
        code: code || null,
        use_llm: true 
      })
    });
    if (!response.ok) throw new Error('Failed to review code');
    return response.json();
  },

  async refactorCode(projectId: string, scope?: string, limitFiles: number = 10): Promise<any> {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/refactor`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ 
        scope: scope || null,
        limit_files: limitFiles
      })
    });
    if (!response.ok) throw new Error('Failed to get refactor suggestions');
    return response.json();
  }
};
