"""Check DNS resolution for Supabase hostname."""
import socket

host = 'db.hibjmylxyfhcizjtmspi.supabase.co'
print(f"Testing DNS resolution for: {host}")
try:
    ip = socket.gethostbyname(host)
    print(f"[OK] DNS resolution: SUCCESS")
    print(f"  {host} resolves to {ip}")
except socket.gaierror as e:
    print(f"[ERROR] DNS resolution: FAILED")
    print(f"  Error: {e}")
    print("\nThis usually means:")
    print("  1. Project is paused (check Supabase dashboard)")
    print("  2. Hostname is incorrect")
    print("  3. Network/DNS issue")
    print("\nTo check if project is paused:")
    print("  Go to: https://supabase.com/dashboard/project/hibjmylxyfhcizjtmspi")
    print("  If you see 'PAUSED', click 'Restore Project'")

