# 💻 Podman Guide: Running a Containerized On-Premises LLM on Windows

## Prerequisites

- Windows 10/11 with WSL2 enabled
- Administrator privileges for installation


## Install Podman for Windows

1. Download and install Podman Desktop from the [official releases page](https://github.com/containers/podman-desktop/releases)
2. Alternatively, install via the command line following the [Podman for Windows tutorial](https://github.com/containers/podman/blob/main/docs/tutorials/podman-for-windows.md)
3. Initialize the Podman machine:
   ```powershell
   podman machine init
   podman machine start
   ```

## Dealing with Corporate Firewalls

Corporate firewalls may cause complications when building containers. Follow these steps:

### 1. Obtain Certificate
Obtain your company's custom certificate (e.g., `company.crt`)

### 2. Copy Certificate
Copy `company.crt` to the `onprem` root folder (parent directory of the `docker` folder)

### 3. Update Podman Machine Trust Store
Update the trust store to allow container operations:
```bash
podman machine ssh
sudo mkdir -p /etc/pki/ca-trust/source/anchors/
sudo vi /etc/pki/ca-trust/source/anchors/company.crt # copy company.crt contents and save
sudo update-ca-trust
exit
```

### 4. Modify Dockerfile
Edit `onprem/docker/Dockerfile-cpu` and add certificate handling after the `FROM` statement:
```dockerfile
FROM python:3.10
# Add corporate certificate
COPY company.crt /usr/local/share/ca-certificates/company.crt
RUN update-ca-certificates
```

## Build Container Image

Navigate to the `onprem/docker` directory and build the image:
```powershell
cd onprem\docker
podman build -t onprem:cpu -f Dockerfile-cpu ..
```

**Build time:** Approximately 10-15 minutes depending on your system and network speed.

## Usage Examples

### Run Interactive REPL
```powershell
podman run --rm -it -v C:\Users\%USERNAME%\onprem_data:/root/onprem_data -v C:\Users\%USERNAME%\.cache:/root/.cache onprem:cpu ipython
```

### Run Web Application
```powershell
podman run --rm -it ^
  -v C:\Users\%USERNAME%\onprem_data:/root/onprem_data ^
  -v C:\Users\%USERNAME%\.cache:/root/.cache ^
  -p 8000:8000 ^
  -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt ^
  -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt ^
  onprem:cpu onprem --port 8000 --address 0.0.0.0
```

Access the web application at: http://localhost:8000

## Volume Mounts Explained

- `onprem_data`: Stores models, embeddings, and application data
- `.cache`: Caches downloaded models and dependencies for faster subsequent runs

## Troubleshooting

### Common Issues

1. **Permission denied errors:**
   ```powershell
   podman machine stop
   podman machine start
   ```

2. **Certificate errors behind corporate firewall:**
   - Verify the certificate is correctly copied to both the host and container
   - Ensure `update-ca-trust` was run successfully

3. **Port already in use:**
   - Change the port mapping: `-p 8001:8000` (access via http://localhost:8001)

4. **Volume mount issues:**
   - Ensure the local directories exist and have proper permissions
   - Use absolute paths for volume mounts

### Performance Tips

- Allocate more resources to the Podman machine:
  ```powershell
  podman machine stop
  podman machine rm
  podman machine init --memory 8192 --cpus 4
  podman machine start
  ```

## Next Steps

After successfully running the container:
1. Check the application logs for any initialization messages
2. Upload your documents to start building your knowledge base
3. Configure model settings based on your hardware capabilities

---

## Podman Command Reference

### Container Management

**Stop a running container:**
```bash
podman stop onprem-container
```

**Start a stopped container:**
```bash
podman start onprem-container
```

**Remove a container:**
```bash
podman rm onprem-container
```

**View container logs:**
```bash
podman logs onprem-container
```

**Check running containers:**
```bash
podman ps
```

**Check all containers (including stopped):**
```bash
podman ps -a
```

### Image Management

**List all images:**
```bash
podman images
```

**Remove an image:**
```bash
podman rmi IMAGE_ID
# or by name
podman rmi onprem:cpu
```

**Transfer image to another machine:**
```bash
# Export image on source machine
podman image save onprem:cpu > onprem-cpu.tar

# Import image on destination machine
podman image load < onprem-cpu.tar
```

### Network and Port Management

**Check what's using a specific port:**
```bash
sudo lsof -i :8000
# or
sudo netstat -tlnp | grep 8000
```

**Kill process using a port:**
```bash
# Find the process ID (PID) from above command, then:
sudo kill <PID>
# or forcefully:
sudo kill -9 <PID>
```

**Use host networking (alternative to port mapping):**
```bash
podman run --rm -it --network=host onprem:cpu onprem --port 8000 --address 0.0.0.0
```

### Troubleshooting Commands

**Build with alternative cgroup manager (for WSL2 issues):**
```bash
podman build --cgroup-manager=cgroupfs -t onprem:cpu -f Dockerfile-cpu ..
```

**Reset podman network (if network issues persist):**
```bash
podman system reset --force
```

**Use alternative networking:**
```bash
# slirp4netns networking
podman run --network=slirp4netns:port_handler=slirp4netns -p 8000:8000 onprem:cpu
```

### Resource Management

**Configure Podman machine resources:**
```bash
podman machine stop
podman machine rm
podman machine init --memory 8192 --cpus 4
podman machine start
```

**Check system info:**
```bash
podman system info
```

**Clean up unused data:**
```bash
podman system prune -a
```

### Interactive Container Access

**Execute commands in running container:**
```bash
podman exec -it onprem-container bash
```

**Run container with shell access:**
```bash
podman run --rm -it onprem:cpu bash
```
