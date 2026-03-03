import { useEffect, useState } from 'react';
import { api } from '../../lib/api';
import { ProjectCard } from './ProjectCard';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';
import type { Project } from '../../types';
import { RefreshCw, Package } from 'lucide-react';

export const ProjectList = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  const loadProjects = async () => {
    try {
      setError('');
      const data = await api.getProjects();
      // Ensure data is an array
      if (Array.isArray(data)) {
        setProjects(data);
      } else {
        console.error('Projects API returned non-array:', data);
        setProjects([]);
      }
    } catch (err: any) {
      console.error('Load projects error:', err);
      setError(err.message || 'Failed to load projects');
      setProjects([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    loadProjects();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <Spinner size={32} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error}</p>
        <Button onClick={loadProjects}>Try Again</Button>
      </div>
    );
  }

  if (projects.length === 0) {
    return (
      <div className="text-center py-12">
        <Package className="mx-auto text-muted-foreground mb-4" size={48} />
        <h3 className="text-lg font-medium text-foreground mb-2">No projects yet</h3>
        <p className="text-muted-foreground">Upload a project to get started</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold">Your Projects</h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw size={16} className={`mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.isArray(projects) && projects.map((project) => (
          <ProjectCard key={project.project_id} project={project} />
        ))}
      </div>
    </div>
  );
};
