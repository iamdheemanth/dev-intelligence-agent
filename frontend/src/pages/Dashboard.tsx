import { Layout } from '../components/layout/Layout';
import { ProjectList } from '../components/projects/ProjectList';
import { UploadProject } from '../components/projects/UploadProject';

export const Dashboard = () => {
  return (
    <Layout>
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
          <p className="text-muted-foreground">
            Manage and analyze your development projects
          </p>
        </div>

        <UploadProject />
        <ProjectList />
      </div>
    </Layout>
  );
};
