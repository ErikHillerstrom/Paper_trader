# AWS EC2 Setup — Automated Paper Trader

Runs `paper_trader_1.py` on a free-tier AWS EC2 instance with cron scheduling.
The evening scan fires at 22:30 and the morning open at 15:35 (Stockholm time) on weekdays.

---

## Prerequisites

- An AWS account (free tier covers 750 hrs/month for 12 months)
- The `.pem` key file downloaded during instance creation
- Windows 10/11 with OpenSSH (built-in)
- Gmail App Password set up (see [Email notifications](README.md#email-notifications) in the main README)

---

## Step 1 — Launch an EC2 instance

1. Log in to the [AWS Management Console](https://console.aws.amazon.com)
2. Search for **EC2** → click **Launch instance**
3. Configure:

| Setting | Value |
|---------|-------|
| Name | `paper-trader` |
| AMI | Ubuntu Server 24.04 LTS (Free tier eligible) |
| Instance type | `t3.micro` (Free tier eligible in most regions) |
| Key pair | Create new → name `paper-trader-key` → RSA → `.pem` → **Download and save securely** |
| Network settings | Default (SSH port 22 open) |
| Storage | 8 GB gp3 (default) |

4. Click **Launch instance**
5. Note the **Public IPv4 address** from the Instances page — you will need it for all SSH commands

---

## Step 2 — Prepare the key file (Windows)

Copy the downloaded `.pem` file to your `.ssh` folder for a clean path:

```powershell
Copy-Item "C:\Users\<you>\OneDrive\Dokument\paper-trader-key.pem" "C:\Users\<you>\.ssh\paper-trader-key.pem"
```

---

## Step 3 — Connect via SSH

```powershell
ssh -i "C:\Users\<you>\.ssh\paper-trader-key.pem" ubuntu@<YOUR-PUBLIC-IP>
```

Type `yes` when prompted about the host fingerprint. You are now on the server.

---

## Step 4 — Install Python and dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git

python3 -m venv ~/venv
source ~/venv/bin/activate

pip install yfinance pandas schedule colorama tabulate matplotlib
```

---

## Step 5 — Transfer project files

In a **new local PowerShell window** (keep the SSH session open), run from your project directory:

```powershell
# Create the remote directory first (run this in the SSH session)
# mkdir -p ~/paper_trader

# Then transfer all .py files from your local machine
scp -i "C:\Users\<you>\.ssh\paper-trader-key.pem" C:\Users\<you>\PycharmProjects\test_model\files\*.py ubuntu@<YOUR-PUBLIC-IP>:~/paper_trader/
```

---

## Step 6 — Set environment variables

In the SSH session, append credentials to `~/.bashrc`:

```bash
echo 'export NOTIFY_FROM="you@gmail.com"' >> ~/.bashrc
echo 'export NOTIFY_PASSWORD="xxxx xxxx xxxx xxxx"' >> ~/.bashrc
echo 'export NOTIFY_TO="you@gmail.com"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
echo $NOTIFY_FROM
echo $NOTIFY_TO
```

Both should print your Gmail address.

---

## Step 7 — Set up cron jobs

```bash
crontab -e
```

Select `nano` if prompted. Paste the following at the bottom of the file:

```
MAILTO=""
TZ=Europe/Stockholm
NOTIFY_FROM=you@gmail.com
NOTIFY_PASSWORD=xxxx xxxx xxxx xxxx
NOTIFY_TO=you@gmail.com

# Evening scan — weekdays at 22:30 Stockholm
30 22 * * 1-5 /home/ubuntu/venv/bin/python /home/ubuntu/paper_trader/paper_trader_1.py --evening >> /home/ubuntu/paper_trader/cron.log 2>&1

# Morning open — weekdays at 15:35 Stockholm
35 15 * * 1-5 /home/ubuntu/venv/bin/python /home/ubuntu/paper_trader/paper_trader_1.py --morning >> /home/ubuntu/paper_trader/cron.log 2>&1
```

Save and exit: **Ctrl+X** → **Y** → **Enter**

> The `TZ` line ensures cron runs on Stockholm time regardless of the server's UTC clock.
> Credentials are set directly in crontab because cron does not source `~/.bashrc`.

---

## Step 8 — Test manually

```bash
cd ~/paper_trader
mkdir -p data
source ~/venv/bin/activate

NOTIFY_FROM=you@gmail.com \
NOTIFY_PASSWORD="xxxx xxxx xxxx xxxx" \
NOTIFY_TO=you@gmail.com \
python paper_trader_1.py --evening
```

You should receive an email within a minute. If it errors, check the output for missing dependencies or path issues.

---

## Ongoing maintenance

**Check the run log:**
```bash
cat ~/paper_trader/cron.log
```

**Re-upload files after local edits:**
```powershell
scp -i "C:\Users\<you>\.ssh\paper-trader-key.pem" C:\Users\<you>\PycharmProjects\test_model\files\*.py ubuntu@<YOUR-PUBLIC-IP>:~/paper_trader/
```

**Do not stop the instance** from the AWS console — stopping it pauses the cron jobs and may assign a new public IP on restart. Leave it running continuously.

---

## Cost

| Period | Cost |
|--------|------|
| First 12 months | Free (750 hrs/month free tier) |
| After 12 months | ~$8–10/month for t3.micro running 24/7 |

Monitor usage in the **AWS Billing Dashboard** to avoid unexpected charges.

---

## Timezone reference

The cron jobs use `TZ=Europe/Stockholm`. UTC offsets:

| Season | Stockholm | UTC equivalent |
|--------|-----------|---------------|
| Winter (CET) | 22:30 / 15:35 | 21:30 / 14:35 |
| Summer (CEST) | 22:30 / 15:35 | 20:30 / 13:35 |

The `TZ` setting handles daylight saving automatically — no manual adjustment needed.
