import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import type { Project } from '../../types';
import { ChevronRight, Folder, Clock } from 'lucide-react';

interface ProjectCardProps {
  project: Project;
}

export const ProjectCard = ({ project }: ProjectCardProps) => {
  const navigate = useNavigate();

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  return (
    <Card className="hover:shadow-md transition-shadow cursor-pointer">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Folder className="text-primary" size={24} />
            </div>
            <div>
              <CardTitle className="text-lg">{project.project_id}</CardTitle>
              <div className="flex items-center space-x-2 mt-1">
                <span className={`text-xs px-2 py-1 rounded-full ${
                  project.status === 'completed' 
                    ? 'bg-green-100 text-green-800' 
                    : project.status === 'processing'
                    ? 'bg-blue-100 text-blue-800'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {project.status}
                </span>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate(`/project/${project.project_id}`)}
          >
            <ChevronRight size={20} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 text-sm text-muted-foreground">
          {project.created_at && (
            <div className="flex items-center space-x-2">
              <Clock size={14} />
              <span>Created {formatDate(project.created_at)}</span>
            </div>
          )}
          {project.meta?.indexed_at && (
            <div className="flex items-center space-x-2">
              <Clock size={14} />
              <span>Indexed {formatDate(project.meta.indexed_at)}</span>
            </div>
          )}
          {project.meta?.chunks_indexed && (
            <div>
              <span className="font-medium">Chunks:</span> {project.meta.chunks_indexed}
            </div>
          )}
          {project.repo_url && (
            <div className="truncate">
              <span className="font-medium">Repo:</span> {project.repo_url}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
