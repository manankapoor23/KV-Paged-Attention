# KV-Paged Visualizer - Hosting Guide

## Quick Start (Local)

The visualizer is already running locally. Access it at:
- **Frontend**: http://localhost:9000
- **Backend**: http://localhost:8000

Start everything with:
```bash
./start_visualizer.sh
```

---

## Option 1: Docker (Recommended for Portability)

### Build Docker Image

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose ports
EXPOSE 8000 9000

# Start both services
CMD bash -c "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &" && \
    "cd frontend && python -m http.server 9000 --bind 0.0.0.0"
```

Build & run:
```bash
docker build -t kv-paged .
docker run -p 8000:8000 -p 9000:9000 kv-paged
```

---

## Option 2: Cloud Platforms

### **Heroku** (Free tier deprecated, but still an option)

```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login
heroku login

# Create app
heroku create kv-paged-visualizer

# Push code
git push heroku main

# Open
heroku open
```

### **Railway.app** (Easy, Recommended)

1. Go to https://railway.app
2. Connect GitHub repo
3. Add two services:
   - **Backend**: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
   - **Frontend**: `cd frontend && python -m http.server 9000 --bind 0.0.0.0`
4. Deploy!

### **Render.com**

1. Push code to GitHub
2. Create two web services:
   - Backend: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
   - Frontend: `cd frontend && python -m http.server 9000 --bind 0.0.0.0`
3. Set environment variables & deploy

### **Vercel** (Frontend only)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend
vercel
```

(Then deploy backend separately to any Python host)

---

## Option 3: VPS (Self-Hosted)

Recommended providers: DigitalOcean, Linode, AWS EC2, Hetzner

### On Ubuntu/Debian VPS:

```bash
# SSH into your server
ssh root@your_server_ip

# Install Python & git
apt update && apt install -y python3 python3-pip git

# Clone repo
git clone https://github.com/yourusername/KV-Paged.git
cd KV-Paged

# Install dependencies
pip install -r requirements.txt

# Run with systemd (persistent)
sudo cp systemd/kv-paged.service /etc/systemd/system/
sudo systemctl enable kv-paged
sudo systemctl start kv-paged
```

Create `/etc/systemd/system/kv-paged.service`:

```ini
[Unit]
Description=KV-Paged Visualizer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/kv-paged
ExecStart=/usr/bin/python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 & \
          bash -c 'cd frontend && python3 -m http.server 9000 --bind 0.0.0.0'
Restart=always

[Install]
WantedBy=multi-user.target
```

### With Nginx (Reverse Proxy)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api {
        proxy_pass http://localhost:8000;
    }

    location / {
        proxy_pass http://localhost:9000;
    }
}
```

Enable HTTPS:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Option 4: Environment Variables

Create `.env` for production:

```bash
BACKEND_URL=https://api.yourdomain.com
FRONTEND_URL=https://yourdomain.com
```

Update `frontend/api.js`:

```javascript
const API_BASE = process.env.BACKEND_URL || "http://localhost:8000";
```

---

## Checklist Before Deploying

- [ ] Update `backend/main.js` CORS to allow your domain:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://yourdomain.com"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

- [ ] Set production-level logging
- [ ] Configure environment variables
- [ ] Test all features in production environment
- [ ] Set up monitoring/alerts
- [ ] Enable HTTPS/SSL
- [ ] Set up auto-scaling if needed

---

## Performance Tips

1. **Cache static files** (frontend) with 1-year expiry
2. **Use CDN** for frontend assets
3. **Database** for storing simulation results (future enhancement)
4. **Load balancing** if traffic is high
5. **Monitor** API response times

---

## Troubleshooting

**CORS errors**: Update backend allowed origins
**Port conflicts**: Change port in `start_visualizer.sh`
**Connection refused**: Ensure both services are running

---

## Questions?

For issues, check:
- Backend logs: `/tmp/kvpaged-backend.log`
- Frontend logs: `/tmp/kvpaged-frontend.log`
