# DigitalOcean VPS Setup (OpenClaw Gateway)

- **Droplet:** Ubuntu 24.04, Basic, Regular, $12/month (2 GB RAM / 1 CPU). 1 GB may be insufficient.
- **SSH:** Add your SSH key during creation.

## On the VPS

```bash
ssh root@YOUR_VPS_IP

# Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# OpenClaw
npm install -g openclaw

# Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Run `openclaw onboard`: set LLM endpoint to Jetson Tailscale IP (e.g. `http://100.x.x.x:8080` or 8081 if RAG is on Jetson), connect Telegram, configure SOUL.md.

```bash
openclaw gateway install
sudo systemctl enable openclaw-gateway
sudo systemctl start openclaw-gateway
openclaw gateway status
```

Checkpoint: **v0.8.0** when production stack is live.
