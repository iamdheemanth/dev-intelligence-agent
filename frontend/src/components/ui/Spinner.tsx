import { Loader2 } from 'lucide-react';

interface SpinnerProps {
  size?: number;
  className?: string;
}

export const Spinner = ({ size = 24, className = '' }: SpinnerProps) => {
  return (
    <Loader2 
      size={size} 
      className={`animate-spin text-primary ${className}`}
    />
  );
};

export const LoadingScreen = ({ message = 'Loading...' }: { message?: string }) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <Spinner size={48} />
      <p className="mt-4 text-muted-foreground">{message}</p>
    </div>
  );
};
