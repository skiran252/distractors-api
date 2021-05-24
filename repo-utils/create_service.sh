sudo cp servicetemplate.txt /etc/systemd/system/app.service
sudo systemctl daemon-reload
sudo systemctl enable app.service
sudo service app start