import { useEffect, useState } from 'react';
import { api } from '../../lib/api';
import { Button } from '../ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Spinner } from '../ui/Spinner';
import { Modal } from '../ui/Modal';
import { ResultDisplay } from '../ui/ResultDisplay';
import type { Project } from '../../types';
import { FileText, Lightbulb, AlertTriangle } from 'lucide-react';

interface ProjectDetailProps {
  projectId: string;
}

export const ProjectDetail = ({ projectId }: ProjectDetailProps) => {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [actionResult, setActionResult] = useState<any>(null);
  const [error, setError] = useState('');
  
  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [modalConfig, setModalConfig] = useState<{
    title: string;
    description?: string;
    defaultValue?: string;
    placeholder?: string;
    onSubmit: (value: string) => void;
    multiline?: boolean;
  } | null>(null);

  useEffect(() => {
    const loadProject = async () => {
      try {
        const projects = await api.getProjects();
        const found = projects.find(p => p.project_id === projectId);
        if (found) {
          setProject(found);
        } else {
          setError(`Project ${projectId} not found. It may still be processing.`);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load project');
      } finally {
        setLoading(false);
      }
    };

    loadProject();
    
    // Poll for updates if project is processing
    const interval = setInterval(loadProject, 5000);
    return () => clearInterval(interval);
  }, [projectId]);

  const handleAction = async (
    action: 'summarize' | 'recommend' | 'triage' | 'refactor',
    apiCall: () => Promise<any>
  ) => {
    setActionLoading(action);
    setActionResult(null);
    setError('');

    try {
      const result = await apiCall();
      setActionResult({ action, data: result });
    } catch (err: any) {
      setError(err.message || `Failed to ${action}`);
    } finally {
      setActionLoading(null);
    }
  };

  const openModal = (config: typeof modalConfig) => {
    setModalConfig(config);
    setModalOpen(true);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <Spinner size={32} />
      </div>
    );
  }

  if (error && !project) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  if (!project) return null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Project Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status</span>
              <span className={`px-3 py-1 rounded-full text-sm ${
                project.status === 'completed' || project.status.includes('indexed')
                  ? 'bg-green-100 text-green-800' 
                  : project.status.includes('processing') || project.status.includes('received')
                  ? 'bg-blue-100 text-blue-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {project.status}
              </span>
            </div>
            {project.meta?.chunks_indexed && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Chunks Indexed</span>
                <span className="text-sm">{project.meta.chunks_indexed}</span>
              </div>
            )}
            {project.meta?.indexed_at && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Indexed At</span>
                <span className="text-sm">
                  {new Date(project.meta.indexed_at).toLocaleString()}
                </span>
              </div>
            )}
            {project.repo_url && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Repository</span>
                <a
                  href={project.repo_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-primary hover:underline"
                >
                  {project.repo_url}
                </a>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>AI Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Button
              onClick={() => {
                openModal({
                  title: 'Summarize Project',
                  description: 'What topic would you like to summarize?',
                  defaultValue: 'project overview',
                  placeholder: 'e.g., API endpoints, database schema',
                  onSubmit: (topic) => handleAction('summarize', () => api.summarizeProject(projectId, topic))
                });
              }}
              disabled={actionLoading !== null}
              variant="secondary"
              className="h-auto py-4 flex-col items-start"
            >
              <div className="flex items-center space-x-2 mb-1">
                <FileText size={20} />
                <span className="font-semibold">Summarize</span>
              </div>
              <span className="text-xs text-muted-foreground">
                Get project overview
              </span>
            </Button>

            <Button
              onClick={() => {
                openModal({
                  title: 'Get Recommendations',
                  description: 'What would you like recommendations for?',
                  defaultValue: 'data processing',
                  placeholder: 'e.g., testing frameworks, API libraries',
                  onSubmit: (query) => handleAction('recommend', () => api.recommendImprovements(projectId, query))
                });
              }}
              disabled={actionLoading !== null}
              variant="secondary"
              className="h-auto py-4 flex-col items-start"
            >
              <div className="flex items-center space-x-2 mb-1">
                <Lightbulb size={20} />
                <span className="font-semibold">Recommend</span>
              </div>
              <span className="text-xs text-muted-foreground">
                Get library recommendations
              </span>
            </Button>

            <Button
              onClick={() => {
                openModal({
                  title: 'Triage Issue',
                  description: 'Enter issue title and description (separate with new line)',
                  defaultValue: 'Code Review\n\nPlease review this code for best practices',
                  placeholder: 'Issue title\n\nIssue description...',
                  multiline: true,
                  onSubmit: (value) => {
                    const lines = value.split('\n');
                    const title = lines[0] || 'Code Review';
                    const body = lines.slice(1).join('\n').trim();
                    handleAction('triage', () => api.triageIssues(projectId, title, body));
                  }
                });
              }}
              disabled={actionLoading !== null}
              variant="secondary"
              className="h-auto py-4 flex-col items-start"
            >
              <div className="flex items-center space-x-2 mb-1">
                <AlertTriangle size={20} />
                <span className="font-semibold">Triage</span>
              </div>
              <span className="text-xs text-muted-foreground">
                Classify issue priority
              </span>
            </Button>

            <Button
              onClick={() => {
                openModal({
                  title: 'Refactor Code',
                  description: 'Enter refactor scope (leave empty for all files)',
                  placeholder: 'e.g., tests, utils, src/components',
                  onSubmit: (scope) => handleAction('refactor', () => api.refactorCode(projectId, scope || undefined))
                });
              }}
              disabled={actionLoading !== null}
              variant="secondary"
              className="h-auto py-4 flex-col items-start"
            >
              <div className="flex items-center space-x-2 mb-1">
                <FileText size={20} />
                <span className="font-semibold">Refactor</span>
              </div>
              <span className="text-xs text-muted-foreground">
                Get refactoring suggestions
              </span>
            </Button>
          </div>

          {actionLoading && (
            <div className="mt-6 flex items-center justify-center space-x-2">
              <Spinner size={20} />
              <span className="text-sm text-muted-foreground">
                Processing {actionLoading}...
              </span>
            </div>
          )}

          {error && (
            <p className="mt-4 text-sm text-red-600">{error}</p>
          )}

          {actionResult && (
            <ResultDisplay action={actionResult.action} data={actionResult.data} />
          )}
        </CardContent>
      </Card>

      {/* Modal for inputs */}
      {modalConfig && (
        <Modal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          title={modalConfig.title}
          description={modalConfig.description}
          defaultValue={modalConfig.defaultValue}
          placeholder={modalConfig.placeholder}
          onSubmit={modalConfig.onSubmit}
          multiline={modalConfig.multiline}
        />
      )}
    </div>
  );
};
