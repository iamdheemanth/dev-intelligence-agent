import { useParams, useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { ProjectDetail } from '../components/projects/ProjectDetail';
import { RepoSearch } from '../components/projects/RepoSearch';
import { Button } from '../components/ui/Button';
import { ArrowLeft } from 'lucide-react';

export const Project = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  if (!id) {
    return (
      <Layout>
        <div className="text-center py-12">
          <p className="text-red-600">Invalid project ID</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/dashboard')}
          >
            <ArrowLeft size={16} className="mr-2" />
            Back to Dashboard
          </Button>
        </div>

        <div>
          <h1 className="text-3xl font-bold mb-2">Project: {id}</h1>
          <p className="text-muted-foreground">
            Analyze and search your project code
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <ProjectDetail projectId={id} />
          </div>
          <div>
            <RepoSearch projectId={id} />
          </div>
        </div>
      </div>
    </Layout>
  );
};
