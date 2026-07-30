import { AuthProvider } from "@/lib/auth";
import { Workspace } from "@/components/workspace";

export default function Home() {
  return (
    <AuthProvider>
      <Workspace />
    </AuthProvider>
  );
}
