Commands to run this WAF.

1.
chmod +x config.sh && chmod +x firewall.py && chmod +x train_model.py

2.
pip3 install -r requirnments.txt
(if it's not working then make virtual Envirnment
"sudo apt install python3-venv && python3 -m venv MLENVIR && source MLENVIR/bin/activate && pip3 install -r requirnments.txt")

3.
./config.sh

4.
python3 train_model.py
python3 firewall.py
