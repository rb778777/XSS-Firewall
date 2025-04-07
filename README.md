# XSS Firewall

An advanced Web Application Firewall (WAF) designed to detect and mitigate Cross-Site Scripting (XSS) attacks using Machine Learning.

## Prerequisites

Before you start, ensure you have the following installed:

- Python 3.6 or above
- pip (Python package installer)
- Virtual Environment tools (optional but recommended)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/rb778777/XSS-Firewall
cd XSS-Firewall
```

### 2. Grant Execution Permissions
Run the following command to make the necessary scripts executable:
```bash
chmod +x config.sh firewall.py train_model.py
```

### 3. Install Dependencies
#### Direct Installation
If you do not wish to use a virtual environment, install dependencies directly:
```bash
pip3 install -r requirements.txt
```

#### Using a Virtual Environment
For a cleaner setup, create and activate a virtual environment:
```bash
sudo apt install python3-venv
python3 -m venv MLENVIR
source MLENVIR/bin/activate
pip3 install -r requirements.txt
```

### 4. Configure the WAF
Run the configuration script:
```bash
./config.sh
```

### 5. Train the Machine Learning Model
Train the ML-based XSS detection model:
```bash
python3 train_model.py
```

### 6. Start the Firewall
Finally, execute the firewall script to activate the WAF:
```bash
python3 firewall.py
```

## Usage

Once the firewall is running, it will monitor web application traffic and block malicious XSS payloads based on the trained ML model. Make sure to periodically update and retrain the model for optimal performance.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request.

## Developed by Rashik

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For more information or to report an issue, feel free to open an issue on the GitHub repository.

